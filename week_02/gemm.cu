// gemm.cu
// Single-file CUDA program: D = alpha * A * B + beta * C
// - Implements two kernels: naive and tiled(shared-memory)
// - Includes CPU reference, timing (cudaEvent), GFLOPs, and max error.
// Build: nvcc -O3 -use_fast_math -lineinfo -o gemm gemm.cu
// Run:   ./gemm --M 512 --N 512 --K 512 --alpha 1.25 --beta 0.75 --kernel tiled --reps 50

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

static void fill_matrix(std::vector<float>& a, float vmin=-1.f, float vmax=1.f) {
    // deterministic pseudo-random but simple
    unsigned int seed = 42u;
    for (size_t i = 0; i < a.size(); ++i) {
        seed = 1664525u * seed + 1013904223u;
        float r = (seed & 0x00FFFFFF) / float(0x01000000);
        a[i] = vmin + (vmax - vmin) * r;
    }
}

// CPU reference: D = alpha*A*B + beta*C
static void gemm_ref(const float* A, const float* B, const float* C, float* D,
                     int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            const float* Ai = A + i * K;
            const float* Bj = B + j; // column j (row-major B: idx k*N + j)
            for (int k = 0; k < K; ++k) {
                acc += (double)Ai[k] * (double)Bj[k * N];
            }
            D[i*N + j] = (float)(alpha * acc + beta * (double)C[i*N + j]);
        }
    }
}

// ---------------------- Kernels ----------------------

__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           const float* __restrict__ C,
                           float* __restrict__ D,
                           int M, int N, int K,
                           float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)
    if (row >= M || col >= N) return;

    float acc = 0.f;
    const float* Ai = A + row * K;
    for (int k = 0; k < K; ++k) {
        acc += Ai[k] * B[k * N + col];
    }
    D[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// TILE configurable: 16 or 32. 16 is more general, uses less registers/shared memory.
template<int TILE>
__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           const float* __restrict__ C,
                           float* __restrict__ D,
                           int M, int N, int K,
                           float alpha, float beta)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y; // row of C
    int col = blockIdx.x * TILE + threadIdx.x; // column of C

    float acc = 0.f;

    // Tiled accumulation: process one TILE width of K each time
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Load A sub-block to shared memory
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.f;
        // Load B sub-block to shared memory
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        D[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// ---------------------- Utilities ----------------------

static double gflops_theoretical(int M, int N, int K, bool include_axpby=true) {
    // A*B: 2*M*N*K FLOPs (mul+add)
    // + alpha* + beta*C: 2*M*N (approximately, one multiply and one add)
    double flops = 2.0 * (double)M * (double)N * (double)K;
    if (include_axpby) flops += 2.0 * (double)M * (double)N;
    return flops / 1e9; // GFLOPs
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Simple command line parsing
struct Args {
    int M=512, N=512, K=512;
    float alpha=1.f, beta=1.f;
    std::string kernel="tiled"; // "naive" or "tiled"
    int reps=50;
    int tile=16; // TILE size for tiled kernel: 16/32
};

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
        else if (!strcmp(argv[i],"--help")) {
            printf("Usage: %s [--M int] [--N int] [--K int] [--alpha f] [--beta f] [--kernel naive|tiled] [--tile 16|32] [--reps int]\n", argv[0]);
            exit(0);
        }
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const int M = args.M, N = args.N, K = args.K;
    const float alpha = args.alpha, beta = args.beta;

    printf("GEMM: D = alpha*A*B + beta*C\n");
    printf("M=%d N=%d K=%d alpha=%.4f beta=%.4f kernel=%s tile=%d reps=%d\n",
           M,N,K,alpha,beta,args.kernel.c_str(), args.tile, args.reps);

    size_t szA = (size_t)M * K;
    size_t szB = (size_t)K * N;
    size_t szC = (size_t)M * N;

    std::vector<float> hA(szA), hB(szB), hC(szC), hD(szC), hRef(szC);
    fill_matrix(hA);
    fill_matrix(hB);
    fill_matrix(hC, -0.25f, 0.25f);

    // Device alloc & copy
    float *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr;
    CHECK_CUDA(cudaMalloc(&dA, szA*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, szB*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, szC*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dD, szC*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), szA*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), szB*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), szC*sizeof(float), cudaMemcpyHostToDevice));

    // launch config
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    // Warm-up + timing
    cudaEvent_t evStart, evStop;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evStop));

    // Warmup
    if (args.kernel == "naive") {
        gemm_naive<<<grid, block>>>(dA, dB, dC, dD, M, N, K, alpha, beta);
    } else {
        if (args.tile == 32) {
            gemm_tiled<32><<<grid, dim3(32,32)>>>(dA, dB, dC, dD, M, N, K, alpha, beta);
        } else {
            gemm_tiled<16><<<grid, dim3(16,16)>>>(dA, dB, dC, dD, M, N, K, alpha, beta);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed reps
    CHECK_CUDA(cudaEventRecord(evStart));
    for (int r = 0; r < args.reps; ++r) {
        if (args.kernel == "naive") {
            gemm_naive<<<grid, block>>>(dA, dB, dC, dD, M, N, K, alpha, beta);
        } else {
            if (args.tile == 32) {
                gemm_tiled<32><<<grid, dim3(32,32)>>>(dA, dB, dC, dD, M, N, K, alpha, beta);
            } else {
                gemm_tiled<16><<<grid, dim3(16,16)>>>(dA, dB, dC, dD, M, N, K, alpha, beta);
            }
        }
    }
    CHECK_CUDA(cudaEventRecord(evStop));
    CHECK_CUDA(cudaEventSynchronize(evStop));

    float ms=0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, evStart, evStop));
    ms /= args.reps; // average per run (ms)

    // Copy back
    CHECK_CUDA(cudaMemcpy(hD.data(), dD, szC*sizeof(float), cudaMemcpyDeviceToHost));

    // Reference & error
    gemm_ref(hA.data(), hB.data(), hC.data(), hRef.data(), M, N, K, alpha, beta);
    float maxErr = max_abs_diff(hRef, hD);

    // Throughput
    double gflops = gflops_theoretical(M,N,K,true);
    double gflops_per_s = gflops / (ms * 1e-3);

    printf("Avg time: %.6f ms | Throughput: %.2f GFLOP/s | Max |GPU-CPU|: %.6f\n",
           ms, gflops_per_s, maxErr);

    // cleanup
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    return 0;
}
