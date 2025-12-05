#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

//CUDA Error Checking Macro
#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

// function to generate random float 
static inline float frand() {
    float u = (float)rand() / RAND_MAX; //generate a random number and nornalize to [0,1]
    return u * 2.f - 1.f; // Set range to [-1,1]
}

//-----------------------------------------------
// CPU version of naive attention (for reference)
//-----------------------------------------------

void attention_naive_cpu(const float* Q, const float* K, const float* V,
                         float* O, int B, int H, int S, int D, float scale)
{
    // convert to 1D array index
    // H number of heads
    // S sequence length
    // D per-head hidden dimension
    auto idx = [H,S,D](int b,int h,int s,int d){ return (((b*H + h)*S + s)*D + d); };
    // Temporary buffer to store attention scores for one query row
    std::vector<float> scores(S);
    // Loop over all batches
    for (int b=0;b<B;++b){
        // Loop over all attention heads
        for (int h=0;h<H;++h){
            // Loop over each query position i
            for (int i=0;i<S;++i){
                // Compute all scores s(i,j) = <q_i, k_j> * scale
                // scores_i[j] = dot(q_i, k_j) * scale
                float maxv = -INFINITY;
                // Loop over all keys j
                for (int j=0;j<S;++j){
                    double dot = 0.0;
                    // Compute dot product q_i ⋅ k_j
                    for (int d=0; d<D; ++d){
                        dot += (double)Q[idx(b,h,i,d)] * (double)K[idx(b,h,j,d)];
                    }
                    // Apply scale (1/sqrt(D))
                    float s = (float)(dot * scale);
                    // Save score for softmax
                    scores[j] = s;
                    // Track maximum score for numerical stability
                    if (s > maxv) maxv = s;
                }
                // Compute softmax denominator:
                // denom = Σ_j exp(scores[j] - maxv)
                double denom = 0.0;
                for (int j=0;j<S;++j) denom += exp((double)scores[j] - maxv);
                // Output o_i = Σ_j softmax(scores[i][j]) * v_j
                for (int d=0; d<D; ++d)  {
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



#ifndef TILE_Q
#define TILE_Q  64     // number of queries processed per block
#endif
#ifndef TILE_K
#define TILE_K  64     // number of key/value loaded at once (streaming tile size)
#endif

//Merge partial results from two tiles
__device__ __forceinline__
void md_combine(float ma, float da, float mb, float db, float &m, float &d) {
    float m_new = fmaxf(ma, mb);
    float d_new = da * __expf(ma - m_new) + db * __expf(mb - m_new);
    m = m_new; d = d_new;
}

//Define GPU kernel
__global__ void flash_attn_forward(
    const float* __restrict__ Q, // [B,H,S,D]
    const float* __restrict__ K, // [B,H,S,D]
    const float* __restrict__ V, // [B,H,S,D]
    float* __restrict__ O,       // [B,H,S,D]
    int B, int H, int S, int D, float scale)
{
   //find index and query range
    int bh = blockIdx.y;           
    int b  = bh / H;
    int h  = bh % H;

    int q_block = blockIdx.x * TILE_Q;  
    if (q_block >= S) return;

    int q_local = threadIdx.x;          
    int qi = q_block + q_local;         

    //allocate shared memory
    extern __shared__ float smem[];
    float* sK = smem;                   
    float* sV = sK + TILE_K * D;       

    auto idx = [H,S,D](int b,int h,int s,int d){ return (((b*H + h)*S + s)*D + d); };

    float m_i = -INFINITY;  
    float l_i = 0.f;        

    float scores_tile[TILE_K];
    //kv tiling
    for (int k0 = 0; k0 < S; k0 += TILE_K) {
        int tk = min(TILE_K, S - k0);

        //load current K/V to shared memory
        for (int t = threadIdx.x; t < tk * D; t += blockDim.x) {
            int j  = t / D;
            int dd = t % D;
            sK[j*D + dd] = K[idx(b,h, k0 + j, dd)];
            sV[j*D + dd] = V[idx(b,h, k0 + j, dd)];
        }
        __syncthreads();
        //caculate dot(qᵢ, kⱼ)
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

   
        if (qi < S) {
            //online softmax combine
            float m_new, l_new;
            md_combine(m_i, l_i, m_tile, 0.f /*placeholder, compute d_tile first*/, m_new, l_new); // l_new meaningless for now
            // Compute tile exponential sum and O increment according to m_new
            float d_tile = 0.f;
            // Compute d_tile first
            for (int j = 0; j < tk; ++j) {
                d_tile += __expf(scores_tile[j] - m_new);
            }
            float alpha = __expf(m_i - m_new);
            for (int dd=0; dd<D; ++dd) {
                float out_prev = (k0 == 0) ? 0.f : O[idx(b,h,qi,dd)];
                float out_new  = out_prev * alpha;
                float accum = 0.f;
                for (int j=0; j<tk; ++j) {
                    float w = __expf(scores_tile[j] - m_new);
                    accum += w * sV[j*D + dd];
                }
                out_new += accum;
                O[idx(b,h,qi,dd)] = out_new;
            }
            m_i = m_new;
            l_i = l_i * alpha + d_tile;
        }
        __syncthreads();
    }
    //softmax normalize
    if (qi < S) {
        float inv = 1.f / fmaxf(l_i, 1e-20f);
        for (int dd=0; dd<D; ++dd) {
            O[idx(b,h,qi,dd)] *= inv;
        }
    }
}


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
    //define B/H/S/D
    if (argc >= 5) {
        B = atoi(argv[1]);
        H = atoi(argv[2]);
        S = atoi(argv[3]);
        D = atoi(argv[4]);
    }
    float scale = 1.0f / std::sqrt((float)D);
    //allocate cpu memory
    size_t N = (size_t)B*H*S*D;
    std::vector<float> hQ(N), hK(N), hV(N), hO_ref(N), hO(N);

    //randomly initialize Q/K/V
    srand(0);
    for (size_t i=0;i<N;++i){ hQ[i]=frand(); hK[i]=frand(); hV[i]=frand(); }

    // run CPU reference
    attention_naive_cpu(hQ.data(), hK.data(), hV.data(), hO_ref.data(), B,H,S,D, scale);

    // allocate GPU memory
    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO, N*sizeof(float)));
    // move Q/K/V from CPU to GPU
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    // run FlashAttention CUDA kernel
    flash_attention_launch(dQ,dK,dV,dO, B,H,S,D, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, N*sizeof(float), cudaMemcpyDeviceToHost));

    // compare cpu and gpu result
    float max_err = 0.f, mae = 0.f;
    for (size_t i=0;i<N;++i){
        float e = fabsf(hO[i] - hO_ref[i]);
        max_err = std::max(max_err, e);
        mae += e;
    }
    mae /= (float)N;
    printf("B=%d H=%d S=%d D=%d  |  max |GPU-CPU| = %.6e,  MAE = %.6e\n", B,H,S,D, max_err, mae);
    //compare first 10 result
    printf("\n--- compare first 10 result ---\n");
    int N_print = std::min((size_t)10, N); 
    
    for (int i = 0; i < N_print; ++i) {
        printf("O[%d]: \t CPU ref: %12.8f \t GPU: %12.8f \t ", 
               i, 
               hO_ref[i], // CPU result
               hO[i]     // GPU result
        );
    }
    printf("----------------------------------------\n\n");

    //release memory
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
