// gemm.cu (week 3)
// C <- alpha * op(A) * op(B) + beta * C
// - Support optional transpose: transA, transB ∈ {N, T}
// - In-place update C (no longer using D as separate output)
// - Keep timing/GFLOPs/CPU verification and other features
// Build: nvcc -O3 -use_fast_math -lineinfo -o gemm gemm.cu
// Run:   ./gemm --M 512 --N 512 --K 512 --alpha 1.25 --beta 0.75 --kernel tiled --reps 50 --transA N --transB T

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

enum Op { OP_N=0, OP_T=1 };

// -------------------- helpers --------------------

static void fill_matrix(std::vector<float>& a, float vmin=-1.f, float vmax=1.f) {
    unsigned int seed = 42u;
    for (size_t i = 0; i < a.size(); ++i) {
        seed = 1664525u * seed + 1013904223u;
        float r = (seed & 0x00FFFFFF) / float(0x01000000);
        a[i] = vmin + (vmax - vmin) * r;
    }
}

// row-major memory layout: A stored as MxK, B stored as KxN, C stored as MxN
// The following two functions return op(A)[i,p] and op(B)[p,j]
__device__ __forceinline__ float A_at(const float* A, int M, int K, int i, int p, Op transA) {
    // A: MxK (row-major)
    // opN: A[i, p] => A[i*K + p]
    // opT: A^T[i, p] = A[p, i] => A[p*M + i] (A is MxK)
    return (transA == OP_N) ? A[i * K + p] : A[p * M + i];
}
__device__ __forceinline__ float B_at(const float* B, int K, int N, int p, int j, Op transB) {
    // B: KxN (row-major)
    // opN: B[p, j] => B[p*N + j]
    // opT: B^T[p, j] = B[j, p] => B[j*K + p] (B is KxN)
    return (transB == OP_N) ? B[p * N + j] : B[j * K + p];
}

// -------------------- CPU reference --------------------
// C <- alpha * op(A) * op(B) + beta * C
static void gemm_ref_op(const float* A, const float* B, float* C,
                        int M, int N, int K, float alpha, float beta,
                        Op transA, Op transB)
{
    // A: stored MxK, B: stored KxN, C: MxN
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int p = 0; p < K; ++p) {
                double a = (transA == OP_N) ? (double)A[i*K + p]
                                            : (double)A[p*M + i];
                double b = (transB == OP_N) ? (double)B[p*N + j]
                                            : (double)B[j*K + p];
                acc += a * b;
            }
            int idx = i * N + j;
            C[idx] = (float)(alpha * acc + beta * (double)C[idx]);
        }
    }
}

// ---------------------- Kernels ----------------------
// naive: each thread computes one C[i,j]
__global__ void gemm_naive_op(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              float alpha, float beta,
                              Op transA, Op transB)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)
    if (i >= M || j >= N) return;

    float acc = 0.f;
    #pragma unroll
    for (int p = 0; p < K; ++p) {
        float a = (transA == OP_N) ? A[i * K + p] : A[p * M + i];
        float b = (transB == OP_N) ? B[p * N + j] : B[j * K + p];
        acc += a * b;
    }
    int idx = i * N + j;
    C[idx] = alpha * acc + beta * C[idx];
}

// tiled(shared memory)
template<int TILE>
__global__ void gemm_tiled_op(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              float alpha, float beta,
                              Op transA, Op transB)
{
    __shared__ float As[TILE][TILE]; // tile: op(A)[row, a_col]
    __shared__ float Bs[TILE][TILE]; // tile: op(B)[b_row, col]

    int row = blockIdx.y * TILE + threadIdx.y; // i
    int col = blockIdx.x * TILE + threadIdx.x; // j

    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x; // p for A tile
        int b_row = t * TILE + threadIdx.y; // p for B tile

        // load op(A)[row, a_col]
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] =
                (transA == OP_N) ? A[row * K + a_col] : A[a_col * M + row];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.f;
        }

        // load op(B)[b_row, col]
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] =
                (transB == OP_N) ? B[b_row * N + col] : B[col * K + b_row];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE; ++p) {
            acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = alpha * acc + beta * C[idx];
    }
}

// ---------------------- Utilities ----------------------

static double gflops_theoretical(int M, int N, int K, bool include_axpby=true) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    if (include_axpby) flops += 2.0 * (double)M * (double)N;
    return flops / 1e9;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ---------------------- Arg parsing ----------------------
struct Args {
    int M=512, N=512, K=512;
    float alpha=1.f, beta=1.f;
    std::string kernel="tiled"; // "naive" or "tiled"
    int reps=50;
    int tile=16;                // 16 or 32
    Op transA = OP_N;           // N/T
    Op transB = OP_N;
};

static Op parse_op(const char* s) {
    if (s[0]=='N' || s[0]=='n') return OP_N;
    if (s[0]=='T' || s[0]=='t') return OP_T;
    fprintf(stderr, "Invalid transpose flag '%s' (use N or T)\n", s);
    exit(1);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--M") && i+1<argc) a.M = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--N") && i+1<argc) a.N = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--K") && i+1<argc) a.K = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--alpha") && i+1<argc) a.alpha = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--beta") && i+1<argc) a.beta = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--kernel") && i+1<argc) a.kernel = argv[++i];
        else if (!strcmp(argv[i],"--reps") && i+1<argc) a.reps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--tile") && i+1<argc) a.tile = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--transA") && i+1<argc) a.transA = parse_op(argv[++i]);
        else if (!strcmp(argv[i],"--transB") && i+1<argc) a.transB = parse_op(argv[++i]);
        else if (!strcmp(argv[i],"--help")) {
            printf("Usage: %s [--M int] [--N int] [--K int] [--alpha f] [--beta f] "
                   "[--kernel naive|tiled] [--tile 16|32] [--reps int] "
                   "[--transA N|T] [--transB N|T]\n", argv[0]);
            exit(0);
        }
    }
    return a;
}

// ---------------------- main ----------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const int M = args.M, N = args.N, K = args.K;
    const float alpha = args.alpha, beta = args.beta;

    printf("GEMM inplace: C <- alpha*op(A)op(B) + beta*C\n");
    printf("M=%d N=%d K=%d alpha=%.4f beta=%.4f kernel=%s tile=%d reps=%d transA=%c transB=%c\n",
           M,N,K,alpha,beta,args.kernel.c_str(), args.tile, args.reps,
           (args.transA==OP_N?'N':'T'), (args.transB==OP_N?'N':'T'));

    // host buffers (A: MxK, B: KxN, C: MxN)
    size_t szA = (size_t)M * K;
    size_t szB = (size_t)K * N;
    size_t szC = (size_t)M * N;

    std::vector<float> hA(szA), hB(szB), hC(szC), hC_ref(szC);
    fill_matrix(hA);
    fill_matrix(hB);
    fill_matrix(hC, -0.25f, 0.25f);
    hC_ref = hC; // Save original C for CPU reference use (in-place update)

    // Device alloc & copy
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    CHECK_CUDA(cudaMalloc(&dA, szA*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, szB*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, szC*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), szA*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), szB*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), szC*sizeof(float), cudaMemcpyHostToDevice));

    // launch config
    dim3 block( (args.tile==32)?32:16, (args.tile==32)?32:16 );
    dim3 grid( (N + block.x - 1)/block.x, (M + block.y - 1)/block.y );

    // Warm-up + timing
    cudaEvent_t evStart, evStop;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evStop));

    // Warmup
    if (args.kernel == "naive") {
        gemm_naive_op<<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta, args.transA, args.transB);
    } else {
        if (args.tile == 32) {
            gemm_tiled_op<32><<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta, args.transA, args.transB);
        } else {
            gemm_tiled_op<16><<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta, args.transA, args.transB);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed reps
    CHECK_CUDA(cudaEventRecord(evStart));
    for (int r = 0; r < args.reps; ++r) {
        if (args.kernel == "naive") {
            gemm_naive_op<<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta, args.transA, args.transB);
        } else {
            if (args.tile == 32) {
                gemm_tiled_op<32><<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta, args.transA, args.transB);
            } else {
                gemm_tiled_op<16><<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta, args.transA, args.transB);
            }
        }
    }
    CHECK_CUDA(cudaEventRecord(evStop));
    CHECK_CUDA(cudaEventSynchronize(evStop));

    float ms=0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, evStart, evStop));
    ms /= args.reps; // average per run (ms)

    // Copy back (C is updated in-place)
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, szC*sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference (update hC_ref in-place)
    gemm_ref_op(hA.data(), hB.data(), hC_ref.data(), M, N, K, alpha, beta, args.transA, args.transB);

    // Error and throughput
    float maxErr = max_abs_diff(hC_ref, hC);
    double gflops = gflops_theoretical(M,N,K,true);
    double gflops_per_s = gflops / (ms * 1e-3);

    printf("Avg time: %.6f ms | Throughput: %.2f GFLOP/s | Max |GPU-CPU|: %.6f\n",
           ms, gflops_per_s, maxErr);

    // cleanup
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
