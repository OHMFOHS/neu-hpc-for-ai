#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <chrono>
#include <cstdlib>

// --------------------- Tunables ---------------------
#ifndef BLOCK_M
#define BLOCK_M 64          // rows per block (queries)
#endif
#ifndef BLOCK_N
#define BLOCK_N 64          // cols per tile (keys/values)
#endif
#ifndef MAX_D
#define MAX_D   64          // head dim upper bound (single head, <= MAX_D)
#endif

// --------------------- Helpers ---------------------
#define CUDA_CHECK(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

#define NCCL_CHECK(cmd)                                                          \
    do {                                                                         \
        ncclResult_t r = (cmd);                                                  \
        if (r != ncclSuccess) {                                                  \
            fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,        \
                    ncclGetErrorString(r));                                      \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

// --------------------- Kernels ---------------------
// 计算每个 GPU 的局部 (m_i, l_i, num_i)
// Q_local: [N_local, d], K_local: [N_local, d], V_local: [N_local, d]
// out_num_local: [N_local, d]  (分子累加, 未与其他GPU合并)
// m_local: [N_local]           (每行最大分数)
// l_local: [N_local]           (∑_j exp(S_ij - m_local[i]))
__global__ void local_partials_kernel(
    const float* __restrict__ Q_local,
    const float* __restrict__ K_local,
    const float* __restrict__ V_local,
    float* __restrict__ out_num_local,
    float* __restrict__ m_local,
    float* __restrict__ l_local,
    int N_local, int N_global, int d,
    int gpu_id, int num_gpus,
    bool causal
){
    extern __shared__ float smem[];
    float* Ks = smem;                 // [BLOCK_N * d]
    float* Vs = Ks + BLOCK_N * d;     // [BLOCK_N * d]

    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * BLOCK_M;
    const int row_idx = row_start + tid;

    // 允许“空闲线程”进入同步，避免 __syncthreads 死锁
    const bool active = (tid < BLOCK_M) && (row_idx < N_local);
    const int global_row = gpu_id * N_local + row_idx;

    // 寄存器缓存
    float Qi[MAX_D];
    float num[MAX_D];
    float mi = -FLT_MAX;
    float li = 0.f;

    // 初始化寄存器
    if (active) {
        for (int k = 0; k < d; ++k) {
            Qi[k]  = Q_local[row_idx * d + k];
            num[k] = 0.f;
        }
    }

    // 遍历本地 K/V 的列分块
    for (int col_start = 0; col_start < N_local; col_start += BLOCK_N) {
        const int bc = min(BLOCK_N, N_local - col_start);

        // load K/V tile to shared
        for (int i = threadIdx.x; i < bc * d; i += blockDim.x) {
            Ks[i] = K_local[col_start * d + i];
            Vs[i] = V_local[col_start * d + i];
        }
        __syncthreads();

        if (active) {
            // 计算 scores 与局部 online softmax 累加
            float smax = -FLT_MAX;
            // 先找 tile 内最大值 (用于数值稳定的两段式，但我们仍采用全局 mi online）
            // 这里直接与 mi 合并
            // 先算 S 并记录最大
            for (int j = 0; j < bc; ++j) {
                float dot = 0.f;
                // Ks 布局: [bc, d]
                const float* Kj = &Ks[j * d];
                #pragma unroll
                for (int k = 0; k < MAX_D; ++k) {
                    if (k < d) dot += Qi[k] * Kj[k];
                }
                const int global_col = gpu_id * N_local + (col_start + j);
                if (causal && global_col > global_row) {
                    // mask
                } else {
                    smax = fmaxf(smax, dot);
                }
            }

            float m_new = fmaxf(mi, smax);
            float scale = expf(mi - m_new);

            // 累加 e^{S - m_new}
            float esum_tile = 0.f;
            // 同时把分子累加
            for (int j = 0; j < bc; ++j) {
                float dot = 0.f;
                const float* Kj = &Ks[j * d];
                #pragma unroll
                for (int k = 0; k < MAX_D; ++k) {
                    if (k < d) dot += Qi[k] * Kj[k];
                }
                const int global_col = gpu_id * N_local + (col_start + j);
                float e = 0.f;
                if (!(causal && global_col > global_row)) {
                    e = expf(dot - m_new);
                }
                esum_tile += e;

                const float* Vj = &Vs[j * d];
                #pragma unroll
                for (int k = 0; k < MAX_D; ++k) {
                    if (k < d) num[k] = scale * num[k] + e * Vj[k];
                }
            }

            li = scale * li + esum_tile;
            mi = m_new;
        }
        __syncthreads();
    }

    if (active) {
        // 写回本地分子、m、l
        for (int k = 0; k < d; ++k) {
            out_num_local[row_idx * d + k] = num[k];
        }
        m_local[row_idx] = mi;
        l_local[row_idx] = li;
    }
}

// 按 m_global 重新缩放本地 (l_i, num_i) ： l_i *= exp(m_i - m_global), num_i *= exp(m_i - m_global)
__global__ void rescale_locals_kernel(
    float* __restrict__ out_num_local,
    float* __restrict__ l_local,
    const float* __restrict__ m_local,
    const float* __restrict__ m_global,
    int N_local, int d
){
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N_local) return;
    const float alpha = expf(m_local[row] - m_global[row]);

    // scale l
    l_local[row] *= alpha;

    // scale num row
    float* nr = &out_num_local[row * d];
    #pragma unroll
    for (int k = 0; k < MAX_D; ++k) {
        if (k < d) nr[k] *= alpha;
    }
}

// 最终归一：O = num_global / l_global
__global__ void finalize_output_kernel(
    const float* __restrict__ num_global,
    const float* __restrict__ l_global,
    float* __restrict__ O_local,
    int N_local, int d
){
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N_local) return;
    const float denom = fmaxf(l_global[row], 1e-20f); // 避免除零
    const float inv = 1.f / denom;
    const float* nr = &num_global[row * d];
    float* Or = &O_local[row * d];
    #pragma unroll
    for (int k = 0; k < MAX_D; ++k) {
        if (k < d) Or[k] = nr[k] * inv;
    }
}

// --------------------- CPU Reference (for small N) ---------------------
void cpu_attention_ref(const std::vector<float>& Q,
                       const std::vector<float>& K,
                       const std::vector<float>& V,
                       std::vector<float>& O,
                       int N, int d, bool causal)
{
    for (int i = 0; i < N; ++i) {
        float m = -FLT_MAX;
        std::vector<float> S(N);
        for (int j = 0; j < N; ++j) {
            float dot = 0.f;
            for (int k = 0; k < d; ++k) dot += Q[i*d+k]*K[j*d+k];
            if (causal && j > i) S[j] = -INFINITY;
            else { S[j] = dot; m = fmaxf(m, dot); }
        }
        double l = 0.0;
        for (int j = 0; j < N; ++j)
            if (S[j] != -INFINITY) l += std::exp(S[j] - m);

        for (int k = 0; k < d; ++k) {
            double num = 0.0;
            for (int j = 0; j < N; ++j)
                if (S[j] != -INFINITY) num += std::exp(S[j] - m) * V[j*d+k];
            O[i*d+k] = static_cast<float>(num / l);
        }
    }
}

// --------------------- Main ---------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <num_gpus 1..8>\n", argv[0]);
        return 1;
    }
    int num_gpus = std::atoi(argv[1]);
    if (num_gpus < 1 || num_gpus > 8) {
        printf("num_gpus must be in [1,8]\n");
        return 1;
    }

    // Problem size (可自行调大做性能评测)
    const int B = 1;           // batch=1
    const int H = 1;           // single head
    const int d = 64;          // head dim (<= MAX_D)
    const int N = 1024;        // sequence length (全局)
    assert(d <= MAX_D);
    assert(N % num_gpus == 0);
    const int N_local = N / num_gpus;

    // 准备 NCCL (单进程多设备)
    std::vector<int> devs(num_gpus);
    for (int i = 0; i < num_gpus; ++i) devs[i] = i;

    std::vector<ncclComm_t> comms(num_gpus);
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devs.data()));

    std::vector<cudaStream_t> streams(num_gpus);
    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
    }

    // Host 生成全局 Q/K/V（为了验证，简单地全量生成；真正训练时通常每卡只需本卡分片）
    std::vector<float> hQ(N*d), hK(N*d), hV(N*d);
    for (int i = 0; i < N*d; ++i) {
        hQ[i] = float((i % 11) - 5) / 11.0f;
        hK[i] = float(((i * 3) % 17) - 8) / 17.0f;
        hV[i] = float(((i * 7) % 19) - 9) / 19.0f;
    }

    // 为每卡分配设备内存（只存本地分片）
    struct Buffers {
        float *Q, *K, *V;
        float *num_local;   // [N_local, d]
        float *m_local;     // [N_local]
        float *l_local;     // [N_local]
        float *m_global;    // [N_local] (全局最大)
        float *O_local;     // [N_local, d]
    };
    std::vector<Buffers> bufs(num_gpus);

    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        size_t seg_mat = size_t(N_local) * d * sizeof(float);
        size_t seg_vec = size_t(N_local) * sizeof(float);

        CUDA_CHECK(cudaMalloc(&bufs[r].Q, seg_mat));
        CUDA_CHECK(cudaMalloc(&bufs[r].K, seg_mat));
        CUDA_CHECK(cudaMalloc(&bufs[r].V, seg_mat));
        CUDA_CHECK(cudaMalloc(&bufs[r].num_local, seg_mat));
        CUDA_CHECK(cudaMalloc(&bufs[r].m_local, seg_vec));
        CUDA_CHECK(cudaMalloc(&bufs[r].l_local, seg_vec));
        CUDA_CHECK(cudaMalloc(&bufs[r].m_global, seg_vec));
        CUDA_CHECK(cudaMalloc(&bufs[r].O_local, seg_mat));

        // 拷贝各自的分片 (连续切 N/num_gpus)
        const int row0 = r * N_local;
        CUDA_CHECK(cudaMemcpyAsync(bufs[r].Q, hQ.data() + row0 * d, seg_mat,
                                   cudaMemcpyHostToDevice, streams[r]));
        CUDA_CHECK(cudaMemcpyAsync(bufs[r].K, hK.data() + row0 * d, seg_mat,
                                   cudaMemcpyHostToDevice, streams[r]));
        CUDA_CHECK(cudaMemcpyAsync(bufs[r].V, hV.data() + row0 * d, seg_mat,
                                   cudaMemcpyHostToDevice, streams[r]));
    }

    // 同步确保数据就位
    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    // 启动局部 kernel：计算 (m_local, l_local, num_local)
    const dim3 grid_local(div_up(N_local, BLOCK_M));
    const dim3 block_local(BLOCK_M);                 // 每行一个线程
    const size_t smem_bytes = size_t(BLOCK_N) * d * 2 * sizeof(float);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        local_partials_kernel<<<grid_local, block_local, smem_bytes, streams[r]>>>(
            bufs[r].Q, bufs[r].K, bufs[r].V,
            bufs[r].num_local, bufs[r].m_local, bufs[r].l_local,
            N_local, N, d, /*gpu_id*/ r, /*num_gpus*/ num_gpus,
            /*causal*/ true
        );
    }

    // 同步局部计算
    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    // (1) allreduce-max: 得到 m_global（注意：按行独立，所以直接对向量做 Max）
    // 这里为了简便，直接把 m_local -> m_global 做 in-place：先拷贝到 m_global 再 AllReduce Max
    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaMemcpyAsync(bufs[r].m_global, bufs[r].m_local,
                                   size_t(N_local)*sizeof(float),
                                   cudaMemcpyDeviceToDevice, streams[r]));
    }
    // 分组 allreduce
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < num_gpus; ++r) {
        NCCL_CHECK(ncclAllReduce((const void*)bufs[r].m_global,
                                 (void*)bufs[r].m_global,
                                 N_local, ncclFloat, ncclMax,
                                 comms[r], streams[r]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // (2) 每卡按 m_global 重新缩放 l_local 与 num_local
    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        const dim3 grid_rs(div_up(N_local, 256));
        const dim3 block_rs(256);
        rescale_locals_kernel<<<grid_rs, block_rs, 0, streams[r]>>>(
            bufs[r].num_local, bufs[r].l_local,
            bufs[r].m_local, bufs[r].m_global,
            N_local, d
        );
    }

    // (3) allreduce-sum: 汇总 l_global 与 num_global（就地累加）
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < num_gpus; ++r) {
        NCCL_CHECK(ncclAllReduce((const void*)bufs[r].l_local,
                                 (void*)bufs[r].l_local,
                                 N_local, ncclFloat, ncclSum,
                                 comms[r], streams[r]));
        NCCL_CHECK(ncclAllReduce((const void*)bufs[r].num_local,
                                 (void*)bufs[r].num_local,
                                 N_local * d, ncclFloat, ncclSum,
                                 comms[r], streams[r]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // (4) 最终归一：O = num_global / l_global
    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        const dim3 grid_f(div_up(N_local, 256));
        const dim3 block_f(256);
        finalize_output_kernel<<<grid_f, block_f, 0, streams[r]>>>(
            bufs[r].num_local, bufs[r].l_local, bufs[r].O_local,
            N_local, d
        );
    }

    // 同步
    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("Distributed FlashAttention (forward, single-head) finished in %lld ms with %d GPU(s)\n",
           (long long)ms, num_gpus);

    // 验证正确性（可在 N 较小时启用）
    {
        std::vector<float> hO(N*d, 0.f);
        // gather O_local
        for (int r = 0; r < num_gpus; ++r) {
            CUDA_CHECK(cudaSetDevice(devs[r]));
            const int row0 = r * N_local;
            CUDA_CHECK(cudaMemcpy(hO.data() + row0 * d, bufs[r].O_local,
                                  size_t(N_local) * d * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }

        // CPU baseline（注意：N 很大时会很慢，建议先把 N 设置到 <= 1024）
        std::vector<float> O_cpu(N*d, 0.f);
        cpu_attention_ref(hQ, hK, hV, O_cpu, N, d, /*causal*/ true);

        double max_err = 0.0;
        for (int i = 0; i < N*d; ++i) {
            max_err = std::max(max_err, std::abs(double(hO[i]) - double(O_cpu[i])));
        }
        printf("Validation max abs error = %.3e\n", max_err);
    }

    // 资源回收
    for (int r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(devs[r]));
        CUDA_CHECK(cudaFree(bufs[r].Q));
        CUDA_CHECK(cudaFree(bufs[r].K));
        CUDA_CHECK(cudaFree(bufs[r].V));
        CUDA_CHECK(cudaFree(bufs[r].num_local));
        CUDA_CHECK(cudaFree(bufs[r].m_local));
        CUDA_CHECK(cudaFree(bufs[r].l_local));
        CUDA_CHECK(cudaFree(bufs[r].m_global));
        CUDA_CHECK(cudaFree(bufs[r].O_local));
        CUDA_CHECK(cudaStreamDestroy(streams[r]));
        ncclCommDestroy(comms[r]);
    }

    return 0;
}
