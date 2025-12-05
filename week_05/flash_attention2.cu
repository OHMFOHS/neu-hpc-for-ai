 #include <cuda_runtime.h>
 #include <cstdio>
 #include <cmath>
 #include <float.h>
 #include <vector>
 #include <algorithm>
 
 #define BLOCK_M 64
 #define BLOCK_N 64
 #define D 64
 #define WARP_SIZE 32
 
 // ---------------- Utility ----------------
 //return max number in array
 __device__ inline float rowmax(const float* s, int n){
     float m = -FLT_MAX;
     for(int i=0;i<n;i++) m = fmaxf(m, s[i]);
     return m;
 }
 // return row sum
 __device__ inline float rowsum(const float* s, int n){
     float x = 0.f;
     for(int i=0;i<n;i++) x += s[i];
     return x;
 }
 
 // flashattention 2 kernel
 __global__ void flashattn2_fwd_kernel(
     const float* __restrict__ Q,
     const float* __restrict__ K,
     const float* __restrict__ V,
     float* __restrict__ O,
     float* __restrict__ L,
     int B, int H, int N, int d, bool causal)
 {  
    // Row tiling
     int br_idx = blockIdx.x;
     int h = blockIdx.y;
     int b = blockIdx.z;
 
     int row_start = br_idx * BLOCK_M;
     if(row_start >= N) return;
     int br = min(BLOCK_M, N - row_start);
    // move pointer to current block 
     const float* Qbh = Q + ((b*H + h)*N*d);
     const float* Kbh = K + ((b*H + h)*N*d);
     const float* Vbh = V + ((b*H + h)*N*d);
     float* Obh = O + ((b*H + h)*N*d);
     float* Lbh = L + ((b*H + h)*N);
    // put tiled K/V in shared memory
     extern __shared__ float shared[];
     float* Ks = shared;
     float* Vs = Ks + BLOCK_N * d;
    //One thread is responsible for an entire row of queries
     int tid = threadIdx.x;
     if(tid >= br) return;
 
     float Qi[D];
     for(int k=0;k<d;k++)
         Qi[k] = Qbh[(row_start + tid)*d + k];
    //online softmax state
     float Oi_num[D] = {0.f};
     float mi = -FLT_MAX;
     float ell = 0.f;
    //iterate Column tiles
     for(int col_start=0; col_start<N; col_start+=BLOCK_N){
         int bc = min(BLOCK_N, N - col_start);
 
         // load K,V block into shared
         for(int i=threadIdx.x; i<bc*d; i+=blockDim.x){
             Ks[i] = Kbh[col_start*d + i];
             Vs[i] = Vbh[col_start*d + i];
         }
         __syncthreads();
 
         // compute attention scores S = Qi*K^T
         float S[BLOCK_N];
         for(int j=0;j<bc;j++){
             float dot = 0.f;
             for(int k=0;k<d;k++)
                 dot += Qi[k]*Ks[j*d + k];
             int col_idx = col_start + j;
             if(causal && col_idx > row_start + tid)
                 S[j] = -INFINITY;
             else
                 S[j] = dot;
         }
 
         // --- online softmax combine ---
         float m_new = fmaxf(mi, rowmax(S, bc));
         float scale = expf(mi - m_new);
         float e_sum = 0.f;
         float block_sum[BLOCK_N];
         for(int j=0;j<bc;j++){
             float e = expf(S[j] - m_new);
             block_sum[j] = e;
             e_sum += e;
         }
         //update Oi_num
         for(int k=0;k<d;k++)
             Oi_num[k] *= scale;
         for(int j=0;j<bc;j++){
             float e = block_sum[j];
             for(int k=0;k<d;k++)
                 Oi_num[k] += e * Vs[j*d + k];
         }
         ell = scale*ell + e_sum;
         mi = m_new;
         __syncthreads();
     }
     //normalize and logsumexp
     for(int k=0;k<d;k++)
         Obh[(row_start+tid)*d + k] = Oi_num[k] / ell;
     Lbh[row_start+tid] = logf(ell) + mi;
 }
 


 // CPU baseline for validation
 void cpu_attention_ref(const std::vector<float>& Q,
                        const std::vector<float>& K,
                        const std::vector<float>& V,
                        std::vector<float>& O,
                        int N, int d, bool causal)
 {
     for(int i=0;i<N;i++){
         std::vector<float> S(N);
         for(int j=0;j<N;j++){
             float dot=0;
             for(int k=0;k<d;k++) dot += Q[i*d+k]*K[j*d+k];
             if(causal && j>i) S[j]=-INFINITY;
             else S[j]=dot;
         }
         float m = *std::max_element(S.begin(), S.end());
         float sum=0;
         for(int j=0;j<N;j++) if(S[j]>-INFINITY) sum+=expf(S[j]-m);
         for(int k=0;k<d;k++){
             float num=0;
             for(int j=0;j<N;j++) if(S[j]>-INFINITY)
                 num += expf(S[j]-m)/sum * V[j*d+k];
             O[i*d+k]=num;
         }
     }
 }
 

 int main(){
     int B=1,H=1,N=64,d=D;
     size_t sz = B*H*N*d*sizeof(float);
     float *Q,*K,*V,*O,*L;
     cudaMallocManaged(&Q,sz); cudaMallocManaged(&K,sz);
     cudaMallocManaged(&V,sz); cudaMallocManaged(&O,sz);
     cudaMallocManaged(&L,B*H*N*sizeof(float));
 
     // init data
     for(int i=0;i<B*H*N*d;i++){
         Q[i]=(float)(i%11)/11.0f;
         K[i]=(float)((i*3)%17)/17.0f;
         V[i]=(float)((i*7)%19)/19.0f;
     }
     
     // define grid size and call kernel
     dim3 grid((N+BLOCK_M-1)/BLOCK_M,H,B);
     dim3 block(64);
     size_t smem = (BLOCK_N*d*2)*sizeof(float);
     flashattn2_fwd_kernel<<<grid,block,smem>>>(Q,K,V,O,L,B,H,N,d,true);
     cudaDeviceSynchronize();
 
     // CPU baseline
     std::vector<float> Qh(Q,Q+N*d), Kh(K,K+N*d), Vh(V,V+N*d);
     std::vector<float> O_cpu(N*d);
     cpu_attention_ref(Qh,Kh,Vh,O_cpu,N,d,true);
 
     // Compare
     double max_err=0;
     for(int i=0;i<N*d;i++){
         double err=fabs(O_cpu[i]-O[i]);
         if(err>max_err) max_err=err;
     }
 
     printf("\n=== Validation Result ===\n");
     for(int i=0;i<4;i++)
         printf("Row %d  GPU=%.6f  CPU=%.6f  Δ=%.6e\n",
                i,O[i*d],O_cpu[i*d],fabs(O[i*d]-O_cpu[i*d]));
     printf("Max abs error: %.6e\n",max_err);
     if(max_err<1e-6) printf("✅ SUCCESS: GPU matches CPU baseline.\n");
     else printf("❌ Mismatch detected!\n");
 
     cudaFree(Q);cudaFree(K);cudaFree(V);cudaFree(O);cudaFree(L);
     return 0;
 }
 