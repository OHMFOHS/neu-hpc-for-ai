#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) if((x)!=cudaSuccess){printf("CUDA error\n");MPI_Abort(MPI_COMM_WORLD,1);}
#define CHECK_NCCL(x) if((x)!=ncclSuccess){printf("NCCL error\n");MPI_Abort(MPI_COMM_WORLD,1);}
#define CHECK_MPI(x)  if((x)!=MPI_SUCCESS){printf("MPI error\n");MPI_Abort(MPI_COMM_WORLD,1);}

static const int D = 64;
static const int N = 1024;
static const int BLOCK_N = 64;
static const int BLOCK_M = 64;

__device__ float rowmax(const float* s, int n) {
    float m = -1e30f;
    for(int i=0;i<n;i++) m = fmaxf(m,s[i]);
    return m;
}

// ======================
// Local FlashAttention2
// ======================
__global__ void fa2_local_kernel(
    const float* __restrict__ Q,      // [N,d] full copy on each GPU
    const float* __restrict__ K_r,    // [N_local,d] shard
    const float* __restrict__ V_r,    // [N_local,d] shard
    float* __restrict__ m_local,      // [N]
    float* __restrict__ l_local,      // [N]
    float* __restrict__ num_local,    // [N,d]
    int N_local, int d, int rank
){
    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = Ks + BLOCK_N*d;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float q[D];
    for(int k=0;k<d;k++) q[k] = Q[row*d + k];

    float m_i = -1e30f;
    float l_i = 0.f;
    float num_i[D];
    for(int k=0;k<d;k++) num_i[k] = 0.f;

    for(int t = 0; t < N_local; t += BLOCK_N){
        int bc = min(BLOCK_N, N_local - t);

        for(int i=threadIdx.x; i<bc*d; i+=blockDim.x){
            Ks[i] = K_r[t*d + i];
            Vs[i] = V_r[t*d + i];
        }
        __syncthreads();

        float S[BLOCK_N];
        for(int j=0;j<bc;j++){
            float dot=0;
            for(int k=0;k<d;k++) dot += q[k]*Ks[j*d + k];
            S[j] = dot / sqrtf((float)d);
        }

        float m_new = fmaxf(m_i, rowmax(S,bc));
        float alpha = expf(m_i - m_new);

        float es = 0;
        float e_j[BLOCK_N];
        for(int j=0;j<bc;j++){
            float e = expf(S[j] - m_new);
            e_j[j] = e;
            es += e;
        }

        for(int k=0;k<d;k++) num_i[k] *= alpha;
        for(int j=0;j<bc;j++){
            for(int k=0;k<d;k++){
                num_i[k] += e_j[j] * Vs[j*d + k];
            }
        }

        l_i = alpha*l_i + es;
        m_i = m_new;
        __syncthreads();
    }

    m_local[row] = m_i;
    l_local[row] = l_i;
    for(int k=0;k<d;k++) num_local[row*d + k] = num_i[k];
}

//Align the local softmax results of each GPU to the global scale
__global__ void scale_local(
    float* l_local, float* num_local,
    const float* m_local, const float* m_global,
    int N, int d)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row>=N) return;

    float s = expf(m_local[row] - m_global[row]);
    l_local[row] *= s;

    for(int k=0;k<d;k++){
        num_local[row*d + k] *= s;
    }
}
//Calculate the final attention output O
__global__ void finalize(
    const float* num, const float* lsum,
    float* O, int N, int d)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row>=N) return;

    float inv = 1.f/(lsum[row] + 1e-12f);
    for(int k=0;k<d;k++){
        O[row*d + k] = num[row*d + k] * inv;
    }
}

// =========================================
// Main
// =========================================
int main(int argc,char** argv){
    //Initialize MPI to obtain rank/world size
    CHECK_MPI(MPI_Init(&argc,&argv));
    int rank,world;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD,&rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD,&world));

    //Bind the GPU, initialize NCCL communication, and create the CUDA stream
    CHECK_CUDA(cudaSetDevice(rank));
    
    // ======= NCCL init =======
    ncclUniqueId id;
    if(rank==0) CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_MPI(MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD));
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, world, id, rank));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    //Determine the sequence length handled by this GPU and initialize Q/K/V on the CPU
    const int N_local = N / world;

    // === Allocate & init data ===
    std::vector<float> hQ(N*D), hK(N_local*D), hV(N_local*D);
    for(int i=0;i<N;i++)
        for(int k=0;k<D;k++)
            hQ[i*D+k] = (float)((i+k)%17)/17;

    int offset = rank*N_local;
    for(int i=0;i<N_local;i++){
        for(int k=0;k<D;k++){
            hK[i*D+k] = (float)((offset+i+k)%19)/19;
            hV[i*D+k] = (float)((offset+i+k*3)%23)/23;
        }
    }
    //malloc all the required buffers on the GPU
    float *dQ,*dK,*dV,*d_m_local,*d_l_local,*d_num_local;
    float *d_m_global,*d_l_sum,*d_num_sum,*dO;

    CHECK_CUDA(cudaMalloc(&dQ,N*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK,N_local*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV,N_local*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_local,N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l_local,N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_num_local,N*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_global,N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l_sum,N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_num_sum,N*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO,N*D*sizeof(float)));
    //Copy the CPU's Q/K/V to the GPU to perform local FlashAttention2
    CHECK_CUDA(cudaMemcpyAsync(dQ,hQ.data(),N*D*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK_CUDA(cudaMemcpyAsync(dK,hK.data(),N_local*D*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK_CUDA(cudaMemcpyAsync(dV,hV.data(),N_local*D*sizeof(float),cudaMemcpyHostToDevice,stream));

    cudaStreamSynchronize(stream);

    // ========== 1) Local FlashAttn2 ==========
    dim3 grid((N+BLOCK_M-1)/BLOCK_M);
    dim3 block(BLOCK_M);
    size_t smem = BLOCK_N*D*2*sizeof(float);

    fa2_local_kernel<<<grid,block,smem,stream>>>(
        dQ,dK,dV,
        d_m_local,d_l_local,d_num_local,
        N_local,D,rank);

    // ========== 2) AllReduce max(m_local) ==========
    cudaMemcpyAsync(d_m_global, d_m_local, N*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    CHECK_NCCL(ncclAllReduce(d_m_global,d_m_global,N,ncclFloat,ncclMax,comm,stream));

    // ========== 3) scale local by exp(m_local - m_global) ==========
    scale_local<<<(N+255)/256,256,0,stream>>>(
        d_l_local,d_num_local,d_m_local,d_m_global,N,D);

    // ========== 4) AllReduce sum ==========
    cudaMemcpyAsync(d_l_sum,d_l_local,N*sizeof(float),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(d_num_sum,d_num_local,N*D*sizeof(float),cudaMemcpyDeviceToDevice,stream);
    cudaStreamSynchronize(stream);

    CHECK_NCCL(ncclAllReduce(d_l_sum,d_l_sum,N,ncclFloat,ncclSum,comm,stream));
    CHECK_NCCL(ncclAllReduce(d_num_sum,d_num_sum,N*D,ncclFloat,ncclSum,comm,stream));

    // ========== 5) finalize O ==========
    finalize<<<(N+255)/256,256,0,stream>>>(d_num_sum,d_l_sum,dO,N,D);

    cudaStreamSynchronize(stream);

    if(rank==0){
        std::vector<float> hO(N*D);
        CHECK_CUDA(cudaMemcpy(hO.data(),dO,N*D*sizeof(float),cudaMemcpyDeviceToHost));
        printf("O[0,0]=%.6f  O[100,0]=%.6f\n",hO[0],hO[100*D]);
        
    }

    MPI_Finalize();
    return 0;
}
