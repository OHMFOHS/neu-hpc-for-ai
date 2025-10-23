// distributed_flash_attention.cu
//
// Multi-GPU (single node) FlashAttention v2 Forward Only, Single Head
// Sequence parallelism over sequence length N.
// Each rank r owns:
//  - A block of Q rows:      Q_r      shape [N_r, d]
//  - A block of K/V columns: K_r, V_r shape [N_r, d]
// It computes local online-softmax stats (m_local, l_local, num_local) from its K/V
// then combines across GPUs via:
//   m_global = AllReduce_max(m_local)
//   scale_r  = exp(m_local - m_global)
//   l_sum    = AllReduce_sum(l_local * scale_r)
//   num_sum  = AllReduce_sum(num_local * scale_r)
//   O_r      = num_sum / l_sum
//
// Build (典型):
//   nvcc -O3 -std=c++17 -ccbin mpicc \
//        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
//        -lcudart -lnccl -lm -o output.bin distributed_flash_attention.cu
//
// Run (用 mpirun 指定进程数 = GPU 数):
//   mpirun --allow-run-as-root -np 4 ./output.bin 4
//
// Notes:
// - 要求 N 能被 world_size 整除（为了简化切分）。可自行扩展支持 remainder 切分。
// - 为了简洁，B=1、H=1；如需扩展，给 grid.z/y 维度加批次/头循环即可。

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <float.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F (FLT_MAX)
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>


#define CHECK_MPI(cmd) do {                                     \
    int e = (cmd);                                              \
    if (e != MPI_SUCCESS) {                                     \
      fprintf(stderr, "MPI error %d at %s:%d\n", e, __FILE__, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, e);                             \
    }                                                           \
  } while(0)

#define CHECK_CUDA(cmd) do {                                    \
    cudaError_t e = (cmd);                                      \
    if (e != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error %s at %s:%d\n",               \
        cudaGetErrorString(e), __FILE__, __LINE__);             \
      MPI_Abort(MPI_COMM_WORLD, 1);                             \
    }                                                           \
  } while(0)

#define CHECK_NCCL(cmd) do {                                    \
    ncclResult_t r = (cmd);                                     \
    if (r != ncclSuccess) {                                     \
      fprintf(stderr, "NCCL error %s at %s:%d\n",               \
        ncclGetErrorString(r), __FILE__, __LINE__);             \
      MPI_Abort(MPI_COMM_WORLD, 1);                             \
    }                                                           \
  } while(0)

// ===================== problem sizes=====================
static const int D  = 64;     // head dim
static const int N  = 4096;   // total seq len 
static const bool CAUSAL = true;

// ===================== block config =====================
static const int BLOCK_M = 64;   // rows per thread block (queries)
static const int BLOCK_N = 64;   // K/V columns tile
static const int THREADS = 64;   // threads per block

// ---------------- Utilities ----------------
__device__ inline float rowmax_local(const float* s, int n) {
  float m = -CUDART_INF_F;
  for (int i = 0; i < n; ++i) m = fmaxf(m, s[i]);
  return m;
}

// ---------------- FlashAttnV2-like local kernel ----------------
// Each block computes a tile of query rows [row_start, row_start+br)
// against *this-rank*'s K/V columns [0, N_local), in BLOCK_N tiles.
// Online softmax per row: keep (m_i, l_i, num_i[:]) locally.
__global__ void fa2_local_kernel(
    const float* __restrict__ Q_r,    // [N_r, d]   rows for this rank
    const float* __restrict__ K_r,    // [N_r, d]   cols for this rank
    const float* __restrict__ V_r,    // [N_r, d]
    float* __restrict__ m_local,      // [N_r]
    float* __restrict__ l_local,      // [N_r]
    float* __restrict__ num_local,    // [N_r, d]
    int N_local, int d,
    int rank, int world_size, bool causal)
{
  extern __shared__ float smem[];                 // size = (BLOCK_N*d)*2
  float* Ks = smem;                               // [BLOCK_N, d]
  float* Vs = Ks + BLOCK_N * d;                   // [BLOCK_N, d]

  const int row_block = blockIdx.x;
  const int tid = threadIdx.x;                    // one thread = one Q row in the block (<= BLOCK_M)

  const int row_start = row_block * BLOCK_M;
  if (row_start >= N_local) return;

  const int br = min(BLOCK_M, N_local - row_start);
  if (tid >= br) return;

  // load one row of Q to registers
  float q_reg[D];
#pragma unroll
  for (int k = 0; k < d; ++k) {
    q_reg[k] = Q_r[(row_start + tid) * d + k];
  }

  // accumulators
  float m_i = -CUDART_INF_F;
  float l_i = 0.0f;

  float num_i[D];
#pragma unroll
  for (int k = 0; k < d; ++k) num_i[k] = 0.0f;

  // global row index of this query (for causal)
  const int global_row = rank * N_local + (row_start + tid);

  // loop over K/V tiles (columns are local shard only)
  for (int col_start = 0; col_start < N_local; col_start += BLOCK_N) {
    const int bc = min(BLOCK_N, N_local - col_start);

    // load K/V tile into shared mem
    for (int i = threadIdx.x; i < bc * d; i += blockDim.x) {
      Ks[i] = K_r[col_start * d + i];
      Vs[i] = V_r[col_start * d + i];
    }
    __syncthreads();

    // compute dot(Q_i, K_j) for j in tile
    float S[BLOCK_N];
#pragma unroll
    for (int j = 0; j < bc; ++j) {
      float dot = 0.0f;
#pragma unroll
      for (int k = 0; k < d; ++k) {
        dot += q_reg[k] * Ks[j * d + k];
      }

      // global column index = rank*N_local + (col_start + j)
      const int global_col = rank * N_local + (col_start + j);
      if (causal && global_col > global_row) {
        S[j] = -CUDART_INF_F;
      } else {
        S[j] = dot / sqrtf((float)d);   // scale by 1/sqrt(d) like standard attention
      }
    }

    // online softmax merge with current tile
    float m_new = fmaxf(m_i, rowmax_local(S, bc));
    float alpha = __expf(m_i - m_new);

    float e_sum = 0.f;
    float e_j[BLOCK_N];
#pragma unroll
    for (int j = 0; j < bc; ++j) {
      float e = __expf(S[j] - m_new);
      e_j[j] = e;
      e_sum += e;
    }

    // update numerator
#pragma unroll
    for (int k = 0; k < d; ++k) {
      num_i[k] *= alpha;
    }
    for (int j = 0; j < bc; ++j) {
      float e = e_j[j];
#pragma unroll
      for (int k = 0; k < d; ++k) {
        num_i[k] += e * Vs[j * d + k];
      }
    }

    // update denominator (sum of exp)
    l_i = alpha * l_i + e_sum;
    m_i = m_new;

    __syncthreads();
  }

  // write local stats
  m_local[row_start + tid] = m_i;
  l_local[row_start + tid] = l_i;
#pragma unroll
  for (int k = 0; k < d; ++k) {
    num_local[(row_start + tid) * d + k] = num_i[k];
  }
}

// scale l_local & num_local by exp(m_local - m_global)
__global__ void scale_with_global_m(
    const float* __restrict__ m_local,
    const float* __restrict__ m_global,
    float* __restrict__ l_local,
    float* __restrict__ num_local,
    int N_local, int d)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N_local) return;

  float s = __expf(m_local[row] - m_global[row]);

  l_local[row] *= s;
  // scale numerator row
  for (int k = 0; k < d; ++k) {
    num_local[row * d + k] *= s;
  }
}

// normalize O = num_sum / l_sum
__global__ void finalize_output(
    const float* __restrict__ num_sum,
    const float* __restrict__ l_sum,
    float* __restrict__ O_r,
    int N_local, int d)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N_local) return;

  float denom = l_sum[row] + 1e-12f;
  float inv = 1.0f / denom;
  for (int k = 0; k < d; ++k) {
    O_r[row * d + k] = num_sum[row * d + k] * inv;
  }
}

int main(int argc, char** argv) {
  // =========== MPI Init ===========
  CHECK_MPI(MPI_Init(&argc, &argv));
  int rank = 0, world_size = 1;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  if (argc < 2) {
    if (rank == 0) {
      printf("Usage: %s <num_gpus 1..8>\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }
  int num_gpus = atoi(argv[1]);
  if (num_gpus < 1 || num_gpus > 8) {
    if (rank == 0) {
      printf("Error: num_gpus must be in [1, 8]\n");
    }
    MPI_Finalize();
    return 1;
  }
  if (world_size != num_gpus) {
    if (rank == 0) {
      printf("Error: mpirun -np must equal <num_gpus>. Got world_size=%d, num_gpus=%d\n",
             world_size, num_gpus);
    }
    MPI_Finalize();
    return 1;
  }

  if (N % world_size != 0) {
    if (rank == 0) {
      printf("Error: N=%d must be divisible by world_size=%d in this demo.\n", N, world_size);
    }
    MPI_Finalize();
    return 1;
  }


  CHECK_CUDA(cudaSetDevice(rank)); 

  // =========== NCCL Init (multi-process) ===========
  ncclUniqueId ncclId;
  if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&ncclId));
  CHECK_MPI(MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, ncclId, rank));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  const int d = D;
  const int N_local = N / world_size;


  std::vector<float> Q_r_host(N_local * d);
  std::vector<float> K_r_host(N_local * d);
  std::vector<float> V_r_host(N_local * d);
  for (int i = 0; i < N_local; ++i) {
    for (int k = 0; k < d; ++k) {
      int g = rank * N_local + i;         
      Q_r_host[i * d + k] = (float)((g + k) % 31) / 31.0f;
      K_r_host[i * d + k] = (float)((g * 7 + k) % 29) / 29.0f;
      V_r_host[i * d + k] = (float)((g * 13 + k) % 23) / 23.0f;
    }
  }

  // =========== Device buffers ===========
  float *d_Qr, *d_Kr, *d_Vr;
  float *d_m_local, *d_l_local, *d_num_local;
  float *d_m_global;   // will hold allreduced max(m_local)
  float *d_l_sum;      // allreduced sums
  float *d_num_sum;    // allreduced sums
  float *d_O;          // output rows for this rank

  CHECK_CUDA(cudaMalloc(&d_Qr, N_local * d * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_Kr, N_local * d * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_Vr, N_local * d * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_m_local, N_local * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_l_local, N_local * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_num_local, N_local * d * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_m_global, N_local * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_l_sum, N_local * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_num_sum, N_local * d * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_O, N_local * d * sizeof(float)));

  CHECK_CUDA(cudaMemcpyAsync(d_Qr, Q_r_host.data(), N_local * d * sizeof(float), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_Kr, K_r_host.data(), N_local * d * sizeof(float), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_Vr, V_r_host.data(), N_local * d * sizeof(float), cudaMemcpyHostToDevice, stream));

  CHECK_CUDA(cudaStreamSynchronize(stream));


  dim3 grid((N_local + BLOCK_M - 1) / BLOCK_M);
  dim3 block(THREADS);
  size_t smem_bytes = (size_t)(BLOCK_N * d * 2) * sizeof(float);

  fa2_local_kernel<<<grid, block, smem_bytes, stream>>>(
      d_Qr, d_Kr, d_Vr,
      d_m_local, d_l_local, d_num_local,
      N_local, d, rank, world_size, CAUSAL);
  CHECK_CUDA(cudaPeekAtLastError());

  // =========== 2) NCCL: m_global = allreduce_max(m_local) ===========
  CHECK_CUDA(cudaMemcpyAsync(d_m_global, d_m_local, N_local * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_NCCL(ncclAllReduce(
      (const void*)d_m_global, (void*)d_m_global,
      N_local, ncclFloat, ncclMax, comm, stream));

  // =========== 3) scale 本地 l/num by exp(m_local - m_global) ===========
  {
    dim3 g2((N_local + 255) / 256);
    dim3 b2(256);
    scale_with_global_m<<<g2, b2, 0, stream>>>(
        d_m_local, d_m_global, d_l_local, d_num_local, N_local, d);
    CHECK_CUDA(cudaPeekAtLastError());
  }

  // =========== 4) NCCL: sum-reduce l_local→l_sum, num_local→num_sum ===========
  CHECK_CUDA(cudaMemcpyAsync(d_l_sum, d_l_local, N_local * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_num_sum, d_num_local, N_local * d * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_NCCL(ncclAllReduce(
      (const void*)d_l_sum, (void*)d_l_sum,
      N_local, ncclFloat, ncclSum, comm, stream));
  CHECK_NCCL(ncclAllReduce(
      (const void*)d_num_sum, (void*)d_num_sum,
      N_local * d, ncclFloat, ncclSum, comm, stream));

  {
    dim3 g3((N_local + 255) / 256);
    dim3 b3(256);
    finalize_output<<<g3, b3, 0, stream>>>(d_num_sum, d_l_sum, d_O, N_local, d);
    CHECK_CUDA(cudaPeekAtLastError());
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<float> O_r_host;
  if (rank == 0) O_r_host.resize(N_local * d);
  else           O_r_host.resize(0);

  {
    std::vector<float> tmp(N_local * d);
    CHECK_CUDA(cudaMemcpy(tmp.data(), d_O, N_local * d * sizeof(float), cudaMemcpyDeviceToHost));
    if (rank == 0) {
      printf("Rank %d: O[0,0]=%.6f  O[last,0]=%.6f\n",
        rank, tmp[0], tmp[(N_local-1)*d + 0]);
    }
  }

  CHECK_CUDA(cudaFree(d_Qr));
  CHECK_CUDA(cudaFree(d_Kr));
  CHECK_CUDA(cudaFree(d_Vr));
  CHECK_CUDA(cudaFree(d_m_local));
  CHECK_CUDA(cudaFree(d_l_local));
  CHECK_CUDA(cudaFree(d_num_local));
  CHECK_CUDA(cudaFree(d_m_global));
  CHECK_CUDA(cudaFree(d_l_sum));
  CHECK_CUDA(cudaFree(d_num_sum));
  CHECK_CUDA(cudaFree(d_O));

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_MPI(MPI_Finalize());
  return 0;
}
