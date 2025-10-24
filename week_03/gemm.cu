// gemm.cu (Week 3, fixed version)
// C <- alpha * op(A) * op(B) + beta * C
// Features:
// - Optional transpose A/B
// - In-place update of C
// - Timing, GFLOPs, CPU verification
// - Reset C each repetition to avoid accumulation
// - Shared memory padding and beta==0 fast path
//
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

// -------------------- CPU reference --------------------
static void gemm_ref_op(const float* A, const float* B, float* C,
                        int M, int N, int K, float alpha, float beta,
                        Op transA, Op transB)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int p = 0; p < K; ++p) {
                double a = (transA == OP_N) ? (double)A[i*K + p] : (double)A[p*M + i];
                double b = (transB == OP_N) ? (double)B[p*N + j] : (double)B[j*K + p];
                acc += a * b;
            }
            int idx = i * N + j;
            double old = (beta != 0.0) ? (double)C[idx] : 0.0;
            C[idx] = (float)(alpha * acc + beta * old);
        }
    }
}

// ---------------------- Kernels ----------------------

// naive kernel
__global__ void gemm_naive_op(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              float alpha, float beta,
                              Op transA, Op transB)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) return;

    float acc = 0.f;
    #pragma unroll
    for (int p = 0; p < K; ++p) {
        float a = (transA == OP_N) ? A[i*K + p] : A[p*M + i];
        float b = (transB == OP_N) ? B[p*N + j] : B[j*K + p];
        acc += a * b;
    }
    int idx = i * N + j;
    float old = (beta != 0.f) ? C[idx] : 0.f;
    C[idx] = alpha * acc + beta * old;
}

// tiled kernel
template<int TILE>
__global__ void gemm_tiled_op(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              float alpha, float beta,
                              Op transA, Op transB)
{
    __shared__ float As[TILE][TILE + 1]; // padding to avoid bank conflicts
    __shared__ float Bs[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] =
                (transA == OP_N) ? A[row*K + a_col] : A[a_col*M + row];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] =
                (transB == OP_N) ? B[b_row*N + col] : B[col*K + b_row];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE; ++p)
            acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row * N + col;
        float old = (beta != 0.f) ? C[idx] : 0.f;
        C[idx] = alpha * acc + beta * old;
    }
}

// ---------------------- Utilities ----------------------
static double gflops_theoretical(int M, int N, int K, bool include_axpby=true) {
    double flops = 2.0 * M * N * K;
    if (include_axpby) flops += 2.0 * M * N;
    return flops / 1e9;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
        m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

// ---------------------- Arg parsing ----------------------
struct Args {
    int M=512, N=512, K=512;
    float alpha=1.f, beta=1.f;
    std::string kernel="tiled";
    int reps=50;
    int tile=16;
    Op transA=OP_N, transB=OP_N;
};

static Op parse_op(const char* s) {
    if (s[0]=='N'||s[0]=='n') return OP_N;
    if (s[0]=='T'||s[0]=='t') return OP_T;
    fprintf(stderr, "Invalid transpose flag '%s'\n", s);
    exit(1);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"--M") && i+1<argc) a.M=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--N") && i+1<argc) a.N=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--K") && i+1<argc) a.K=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--alpha") && i+1<argc) a.alpha=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--beta") && i+1<argc) a.beta=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--kernel") && i+1<argc) a.kernel=argv[++i];
        else if (!strcmp(argv[i],"--reps") && i+1<argc) a.reps=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--tile") && i+1<argc) a.tile=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--transA") && i+1<argc) a.transA=parse_op(argv[++i]);
        else if (!strcmp(argv[i],"--transB") && i+1<argc) a.transB=parse_op(argv[++i]);
    }
    return a;
}

// ---------------------- main ----------------------
int main(int argc, char** argv) {
    Args args=parse_args(argc, argv);
    int M=args.M,N=args.N,K=args.K;
    float alpha=args.alpha,beta=args.beta;

    printf("GEMM inplace: C <- alpha*op(A)op(B) + beta*C\n");
    printf("M=%d N=%d K=%d alpha=%.4f beta=%.4f kernel=%s tile=%d reps=%d transA=%c transB=%c\n",
           M,N,K,alpha,beta,args.kernel.c_str(),args.tile,args.reps,
           (args.transA==OP_N?'N':'T'),(args.transB==OP_N?'N':'T'));

    size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N;
    std::vector<float> hA(szA),hB(szB),hC(szC),hC_ref(szC);
    fill_matrix(hA); fill_matrix(hB); fill_matrix(hC,-0.25f,0.25f);
    hC_ref=hC;

    float *dA=nullptr,*dB=nullptr,*dC=nullptr,*dC0=nullptr;
    CHECK_CUDA(cudaMalloc(&dA,szA*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB,szB*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC,szC*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC0,szC*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA,hA.data(),szA*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB.data(),szB*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC,hC.data(),szC*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC0,hC.data(),szC*sizeof(float),cudaMemcpyHostToDevice));

    dim3 block((args.tile==32)?32:16,(args.tile==32)?32:16);
    dim3 grid((N+block.x-1)/block.x,(M+block.y-1)/block.y);

    cudaEvent_t evStart,evStop;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evStop));

    // warm-up
    if(args.kernel=="naive")
        gemm_naive_op<<<grid,block>>>(dA,dB,dC,M,N,K,alpha,beta,args.transA,args.transB);
    else if(args.tile==32)
        gemm_tiled_op<32><<<grid,block>>>(dA,dB,dC,M,N,K,alpha,beta,args.transA,args.transB);
    else
        gemm_tiled_op<16><<<grid,block>>>(dA,dB,dC,M,N,K,alpha,beta,args.transA,args.transB);
    CHECK_CUDA(cudaDeviceSynchronize());

    float total_ms=0.f;
    for(int r=0;r<args.reps;++r){
        CHECK_CUDA(cudaMemcpyAsync(dC,dC0,szC*sizeof(float),cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaEventRecord(evStart));
        if(args.kernel=="naive")
            gemm_naive_op<<<grid,block>>>(dA,dB,dC,M,N,K,alpha,beta,args.transA,args.transB);
        else if(args.tile==32)
            gemm_tiled_op<32><<<grid,block>>>(dA,dB,dC,M,N,K,alpha,beta,args.transA,args.transB);
        else
            gemm_tiled_op<16><<<grid,block>>>(dA,dB,dC,M,N,K,alpha,beta,args.transA,args.transB);
        CHECK_CUDA(cudaEventRecord(evStop));
        CHECK_CUDA(cudaEventSynchronize(evStop));
        float one_ms=0.f;
        CHECK_CUDA(cudaEventElapsedTime(&one_ms,evStart,evStop));
        total_ms+=one_ms;
    }
    float ms=total_ms/args.reps;

    CHECK_CUDA(cudaMemcpy(hC.data(),dC,szC*sizeof(float),cudaMemcpyDeviceToHost));
    gemm_ref_op(hA.data(),hB.data(),hC_ref.data(),M,N,K,alpha,beta,args.transA,args.transB);

    float maxErr=max_abs_diff(hC_ref,hC);
    double gflops=gflops_theoretical(M,N,K,true);
    double gflops_per_s=gflops/(ms*1e-3);
    printf("Avg time: %.6f ms | Throughput: %.2f GFLOP/s | Max |GPU-CPU|: %.6f\n",
           ms,gflops_per_s,maxErr);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC0);
    cudaEventDestroy(evStart); cudaEventDestroy(evStop);
    return 0;
}
