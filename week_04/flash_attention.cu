// flash_attention.cu
// nvcc -O3 -use_fast_math -lineinfo -o flash flash_attention.cu
// ./flash  # optional: B H S D (default 1 2 128 64)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

static inline float frand() {
    return (float)rand() / RAND_MAX * 2.f - 1.f; // [-1,1]
}

// --------------------------------------
// CPU naive attention (reference)
// --------------------------------------
void attention_naive_cpu(const float* Q, const float* K, const float* V,
                         float* O, int B, int H, int S, int D, float scale)
{
    // Layout: [B, H, S, D] row-major contiguous
    auto idx = [H,S,D](int b,int h,int s,int d){ return (((b*H + h)*S + s)*D + d); };

    std::vector<float> scores(S);
    for (int b=0;b<B;++b){
        for (int h=0;h<H;++h){
            for (int i=0;i<S;++i){
                // scores_i[j] = <q_i, k_j> * scale
                float maxv = -INFINITY;
                for (int j=0;j<S;++j){
                    double dot = 0.0;
                    for (int d=0; d<D; ++d){
                        dot += (double)Q[idx(b,h,i,d)] * (double)K[idx(b,h,j,d)];
                    }
                    float s = (float)(dot * scale);
                    scores[j] = s;
                    if (s > maxv) maxv = s;
                }
                // softmax
                double denom = 0.0;
                for (int j=0;j<S;++j) denom += exp((double)scores[j] - maxv);
                // output o_i = sum_j softmax_ij * v_j
                for (int d=0; d<D; ++d) {
                    double acc = 0.0;
                    for (int j=0;j<S;++j){
                        double w = exp((double)scores[j] - maxv) / denom;
                        acc += w * (double)V[idx(b,h,j,d)];
                    }
                    O[idx(b,h,i,d)] = (float)acc;
                }
            }
        }
    }
}

// --------------------------------------
// FlashAttention kernel (online softmax + KV tiling + shared memory)
// One block processes: TILE_Q queries from a fixed (b,h) group.
// Each thread handles one query (one i), scanning along K/V dimension in tiles.
// --------------------------------------

#ifndef TILE_Q
#define TILE_Q  64     // number of queries processed per block
#endif
#ifndef TILE_K
#define TILE_K  64     // number of key/value loaded at once (streaming tile size)
#endif

// Combine two segments (m,d): m = max(ma, mb), d = da*e^(ma-m) + db*e^(mb-m)
__device__ __forceinline__
void md_combine(float ma, float da, float mb, float db, float &m, float &d) {
    float m_new = fmaxf(ma, mb);
    float d_new = da * __expf(ma - m_new) + db * __expf(mb - m_new);
    m = m_new; d = d_new;
}

__global__ void flash_attn_forward(
    const float* __restrict__ Q, // [B,H,S,D]
    const float* __restrict__ K, // [B,H,S,D]
    const float* __restrict__ V, // [B,H,S,D]
    float* __restrict__ O,       // [B,H,S,D]
    int B, int H, int S, int D, float scale)
{
    // block index: each block handles one (b,h) group of query row tiles
    int bh = blockIdx.y;           // 0..(B*H-1)
    int b  = bh / H;
    int h  = bh % H;

    int q_block = blockIdx.x * TILE_Q;  // starting query row number for this block
    if (q_block >= S) return;

    int q_local = threadIdx.x;          // 0..(TILE_Q-1)
    int qi = q_block + q_local;         // query row this thread is responsible for (note boundaries)

    // Shared memory: one tile of K and one tile of V
    extern __shared__ float smem[];
    float* sK = smem;                   // [TILE_K, D]
    float* sV = sK + TILE_K * D;        // [TILE_K, D]

    auto idx = [H,S,D](int b,int h,int s,int d){ return (((b*H + h)*S + s)*D + d); };

    // For query i this thread is responsible for, prepare online softmax (m,l) & output vector acc
    float m_i = -INFINITY;  // running max
    float l_i = 0.f;        // running denom
    // Output accumulator (accumulated in registers in loops)
    // For general D, use loops instead of fixed register vectors
    // Could also be made into small vectorization (e.g., block processing of D), keeping simple here
    // Initialize to 0 first
    // Note: to avoid large stack, changed to process dimension by dimension (see for(d) loop writeback below)
    // Since we need to do weighted accumulation of V on each K tile, we need inner loop over d.
    // But to improve locality, we change to: first compute q·k_j scores on each K/V tile,
    // then do weighted accumulation for each dimension d (read sV[j*D + d]).
    // Temporarily cache scores for current tile (corresponding to qi, j in tile)
    float scores_tile[TILE_K];

    // Loop through K/V tiles
    for (int k0 = 0; k0 < S; k0 += TILE_K) {
        int tk = min(TILE_K, S - k0);

        // Load K, V to shared memory (all threads cooperate)
        // Read [tk, D] block into sK/sV
        for (int t = threadIdx.x; t < tk * D; t += blockDim.x) {
            int j  = t / D;
            int dd = t % D;
            sK[j*D + dd] = K[idx(b,h, k0 + j, dd)];
            sV[j*D + dd] = V[idx(b,h, k0 + j, dd)];
        }
        __syncthreads();

        // 1) This thread (query row qi) computes current tile scores and finds max within tile
        float m_tile = -INFINITY;
        if (qi < S) {
            for (int j = 0; j < tk; ++j) {
                double dot = 0.0;
                // dot(q_i, k_{k0+j})
                for (int dd=0; dd<D; ++dd) {
                    float qv = Q[idx(b,h,qi,dd)];
                    float kv = sK[j*D + dd];
                    dot += (double)qv * (double)kv;
                }
                float s = (float)(dot * (double)scale);
                scores_tile[j] = s;
                if (s > m_tile) m_tile = s;
            }
        }

        // 2) Online softmax merge: (m_i, l_i) with current tile (m_tile, d_tile)
        //    d_tile = sum_j e^{scores_j - m_new}
        if (qi < S) {
            float m_new, l_new;
            md_combine(m_i, l_i, m_tile, 0.f /*placeholder, compute d_tile first*/, m_new, l_new); // l_new meaningless for now
            // Compute tile exponential sum and O increment according to m_new
            float d_tile = 0.f;
            // Compute d_tile first
            for (int j = 0; j < tk; ++j) {
                d_tile += __expf(scores_tile[j] - m_new);
            }
            // Update l_i and O: O = O*exp(m_i - m_new) + sum_j e^{scores_j - m_new} * V_j
            float alpha = __expf(m_i - m_new);
            // Do weighted accumulation for each dimension d
            for (int dd=0; dd<D; ++dd) {
                // First decay previous output (multiply by alpha)
                float out_prev = (k0 == 0) ? 0.f : O[idx(b,h,qi,dd)];
                float out_new  = out_prev * alpha;
                // Current tile accumulation
                float accum = 0.f;
                for (int j=0; j<tk; ++j) {
                    float w = __expf(scores_tile[j] - m_new);
                    accum += w * sV[j*D + dd];
                }
                out_new += accum;
                O[idx(b,h,qi,dd)] = out_new;
            }
            // Update (m_i, l_i)
            m_i = m_new;
            l_i = l_i * alpha + d_tile;
        }
        __syncthreads();
    }

    // 3) Normalization: O_i /= l_i
    if (qi < S) {
        float inv = 1.f / fmaxf(l_i, 1e-20f);
        for (int dd=0; dd<D; ++dd) {
            O[idx(b,h,qi,dd)] *= inv;
        }
    }
}

// --------------------------------------
// host helper
// --------------------------------------
void flash_attention_launch(const float* dQ, const float* dK, const float* dV,
                            float* dO, int B, int H, int S, int D, float scale)
{
    dim3 grid((S + TILE_Q - 1)/TILE_Q, B*H);
    dim3 block(TILE_Q);  // one query per thread
    size_t smem = (size_t)(TILE_K * D * 2) * sizeof(float); // sK + sV
    flash_attn_forward<<<grid, block, smem>>>(dQ, dK, dV, dO, B, H, S, D, scale);
    CHECK_CUDA(cudaPeekAtLastError());
}

// --------------------------------------
// main: generate random data, run CPU/Flash comparison
// --------------------------------------
int main(int argc, char** argv) {
    int B = 1, H = 2, S = 128, D = 64;
    if (argc >= 5) {
        B = atoi(argv[1]);
        H = atoi(argv[2]);
        S = atoi(argv[3]);
        D = atoi(argv[4]);
    }
    float scale = 1.0f / std::sqrt((float)D);

    size_t N = (size_t)B*H*S*D;
    std::vector<float> hQ(N), hK(N), hV(N), hO_ref(N), hO(N);

    srand(0);
    for (size_t i=0;i<N;++i){ hQ[i]=frand(); hK[i]=frand(); hV[i]=frand(); }

    // CPU reference
    attention_naive_cpu(hQ.data(), hK.data(), hV.data(), hO_ref.data(), B,H,S,D, scale);

    // GPU
    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO, N*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    flash_attention_launch(dQ,dK,dV,dO, B,H,S,D, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, N*sizeof(float), cudaMemcpyDeviceToHost));

    // compare
    float max_err = 0.f, mae = 0.f;
    for (size_t i=0;i<N;++i){
        float e = fabsf(hO[i] - hO_ref[i]);
        max_err = std::max(max_err, e);
        mae += e;
    }
    mae /= (float)N;
    printf("B=%d H=%d S=%d D=%d  |  max |GPU-CPU| = %.6e,  MAE = %.6e\n", B,H,S,D, max_err, mae);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
