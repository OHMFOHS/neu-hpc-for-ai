// deepseek_moe.cu
//
// DeepseekV3MoE: Unfused, distributed, data-parallel, expert-parallel CUDA kernel
// Implementation of Mixture of Experts (MoE) for DeepSeek-V3
//
// Components:
// - DeepseekTopKRouter: Gate that routes tokens to top-k experts
// - DeepseekV3MLP: Shared expert MLP block
// - DeepseekV3NaiveMoe: Routed experts (list of DeepseekV3MLP blocks)
// - DeepseekV3MoE: Main module combining all components
//
// Build:
//   nvcc -O3 -std=c++17 -ccbin mpicc \
//        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
//        -lcudart -lnccl -lm -o deepseek_moe deepseek_moe.cu
//
// Run:
//   mpirun --allow-run-as-root -np <num_gpus> ./deepseek_moe <batch_size> <seq_len> <hidden_dim> <num_experts> <top_k>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

#ifndef CUDART_INF_F
#define CUDART_INF_F (FLT_MAX)
#endif

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

// ===================== Configuration =====================
static const int TILE_SIZE = 16;  // Tile size for GEMM operations
static const int BLOCK_SIZE = 256; // Default block size for kernels

// ===================== Helper Functions =====================

// Fill matrix with random values
__host__ void fill_matrix(float* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    unsigned int seed = 42u;
    for (size_t i = 0; i < size; ++i) {
        seed = 1664525u * seed + 1013904223u;
        float r = (seed & 0x00FFFFFF) / float(0x01000000);
        data[i] = min_val + (max_val - min_val) * r;
    }
}

// ===================== DeepseekTopKRouter =====================
// Routes tokens to top-k experts based on gate scores
// Input: gate_logits [batch_size * seq_len, num_experts]
// Output: expert_indices [batch_size * seq_len, top_k], expert_weights [batch_size * seq_len, top_k]

__global__ void topk_router_kernel(
    const float* __restrict__ gate_logits,  // [batch_seq, num_experts]
    int* __restrict__ expert_indices,        // [batch_seq, top_k]
    float* __restrict__ expert_weights,      // [batch_seq, top_k]
    int batch_seq, int num_experts, int top_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq) return;

    // Each thread handles one token
    const float* token_scores = gate_logits + idx * num_experts;
    
    // Simple top-k selection (using bubble-sort like approach for small k)
    // For production, use more efficient algorithms like radix select
    float top_scores[8];  // Assume top_k <= 8
    int top_indices[8];
    
    // Initialize with first k elements
    for (int i = 0; i < top_k && i < num_experts; ++i) {
        top_scores[i] = token_scores[i];
        top_indices[i] = i;
    }
    
    // Sort first k elements (simple insertion sort)
    for (int i = 1; i < top_k && i < num_experts; ++i) {
        float score = top_scores[i];
        int idx_val = top_indices[i];
        int j = i - 1;
        while (j >= 0 && top_scores[j] < score) {
            top_scores[j + 1] = top_scores[j];
            top_indices[j + 1] = top_indices[j];
            --j;
        }
        top_scores[j + 1] = score;
        top_indices[j + 1] = idx_val;
    }
    
    // Process remaining elements
    for (int i = top_k; i < num_experts; ++i) {
        float score = token_scores[i];
        if (score > top_scores[top_k - 1]) {
            // Insert into sorted array
            int pos = top_k - 1;
            while (pos > 0 && score > top_scores[pos - 1]) {
                top_scores[pos] = top_scores[pos - 1];
                top_indices[pos] = top_indices[pos - 1];
                --pos;
            }
            top_scores[pos] = score;
            top_indices[pos] = i;
        }
    }
    
    // Compute softmax over top-k
    float max_score = top_scores[0];
    for (int i = 1; i < top_k; ++i) {
        if (top_scores[i] > max_score) max_score = top_scores[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        top_scores[i] = __expf(top_scores[i] - max_score);
        sum_exp += top_scores[i];
    }
    
    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    for (int i = 0; i < top_k; ++i) {
        expert_indices[idx * top_k + i] = top_indices[i];
        expert_weights[idx * top_k + i] = top_scores[i] * inv_sum;
    }
}

// ===================== GEMM Kernels =====================
// Tiled GEMM kernel for MLP operations

template<int TILE>
__global__ void gemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f)
{
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE; ++p)
            acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row * N + col;
        float old = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * acc + beta * old;
    }
}

// ===================== Activation Functions =====================

__device__ inline float silu_activation(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = silu_activation(input[idx]);
}

// ===================== DeepseekV3MLP =====================
// MLP block: up_proj -> SiLU -> down_proj
// Input: [batch_seq, hidden_dim]
// Output: [batch_seq, hidden_dim]

__host__ void deepseek_v3_mlp_forward(
    const float* input,           // [batch_seq, hidden_dim]
    const float* gate_proj,       // [hidden_dim, intermediate_dim]
    const float* up_proj,         // [hidden_dim, intermediate_dim]
    const float* down_proj,       // [intermediate_dim, hidden_dim]
    float* output,                // [batch_seq, hidden_dim]
    float* intermediate,          // [batch_seq, intermediate_dim] (temp buffer)
    int batch_seq, int hidden_dim, int intermediate_dim,
    cudaStream_t stream)
{
    // Step 1: gate_proj: input @ gate_proj^T -> gate [batch_seq, intermediate_dim]
    dim3 block_gemm(TILE_SIZE, TILE_SIZE);
    dim3 grid_gemm((intermediate_dim + TILE_SIZE - 1) / TILE_SIZE,
                   (batch_seq + TILE_SIZE - 1) / TILE_SIZE);
    
    // Allocate temp buffer for gate
    float* gate = intermediate;  // Reuse intermediate buffer
    
    gemm_tiled<TILE_SIZE><<<grid_gemm, block_gemm, 0, stream>>>(
        input, gate_proj, gate, batch_seq, intermediate_dim, hidden_dim, 1.0f, 0.0f);
    
    // Step 2: up_proj: input @ up_proj^T -> up [batch_seq, intermediate_dim]
    float* up = intermediate + batch_seq * intermediate_dim;
    gemm_tiled<TILE_SIZE><<<grid_gemm, block_gemm, 0, stream>>>(
        input, up_proj, up, batch_seq, intermediate_dim, hidden_dim, 1.0f, 0.0f);
    
    // Step 3: SiLU(gate) * up
    dim3 block_act((batch_seq * intermediate_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_act(BLOCK_SIZE);
    
    // Apply SiLU to gate, then multiply with up
    // For simplicity, we'll do: gate = SiLU(gate), then gate = gate * up
    silu_kernel<<<block_act, grid_act, 0, stream>>>(gate, gate, batch_seq * intermediate_dim);
    
    // Element-wise multiply: gate = gate * up
    // Using a simple kernel for element-wise multiply
    // (In production, use cuBLAS or optimized kernel)
    // For now, we'll compute in the next step
    
    // Step 4: down_proj: (SiLU(gate) * up) @ down_proj^T -> output
    // Actually compute gate * up first
    // For simplicity, assume we have a temp buffer for the multiplied result
    // We'll use gate as the result: gate = gate * up (element-wise)
    // Then: output = gate @ down_proj^T
    
    // Element-wise multiply kernel
    // For now, we compute this in a separate kernel call would be needed
    // To keep it simple, let's restructure:
    // We need: output = down_proj^T @ (SiLU(gate) * up)
    // Which is: (SiLU(gate) * up) @ down_proj^T
    // So: temp = SiLU(gate) * up, then output = temp @ down_proj^T
    
    // For this implementation, we'll use a simplified approach
    // where we compute the operations step by step
}

// Simplified MLP kernel that does all operations in one pass
__global__ void mlp_forward_kernel(
    const float* __restrict__ input,        // [batch_seq, hidden_dim]
    const float* __restrict__ gate_proj,    // [hidden_dim, intermediate_dim]
    const float* __restrict__ up_proj,      // [hidden_dim, intermediate_dim]
    const float* __restrict__ down_proj,    // [intermediate_dim, hidden_dim]
    float* __restrict__ output,             // [batch_seq, hidden_dim]
    int batch_seq, int hidden_dim, int intermediate_dim)
{
    int token_idx = blockIdx.x;
    if (token_idx >= batch_seq) return;
    
    int tid = threadIdx.x;
    const float* token_input = input + token_idx * hidden_dim;
    float* token_output = output + token_idx * hidden_dim;
    
    // Each thread handles one output dimension
    if (tid < hidden_dim) {
        // Compute: output = down_proj^T @ (SiLU(gate_proj^T @ input) * (up_proj^T @ input))
        
        // First compute gate and up projections
        float gate_val = 0.0f;
        float up_val = 0.0f;
        for (int k = 0; k < hidden_dim; ++k) {
            gate_val += token_input[k] * gate_proj[k * intermediate_dim + (tid % intermediate_dim)];
            up_val += token_input[k] * up_proj[k * intermediate_dim + (tid % intermediate_dim)];
        }
        
        // Apply SiLU and multiply
        float activated = silu_activation(gate_val) * up_val;
        
        // Compute down projection
        float out_val = 0.0f;
        int inter_idx = tid % intermediate_dim;
        for (int k = 0; k < intermediate_dim; ++k) {
            out_val += (k == inter_idx ? activated : 0.0f) * down_proj[k * hidden_dim + tid];
        }
        
        token_output[tid] = out_val;
    }
}

// Better MLP implementation using tiled GEMM
__host__ void deepseek_v3_mlp_forward_optimized(
    const float* input,           // [batch_seq, hidden_dim]
    const float* gate_proj,       // [hidden_dim, intermediate_dim]
    const float* up_proj,         // [hidden_dim, intermediate_dim]
    const float* down_proj,       // [intermediate_dim, hidden_dim]
    float* output,                // [batch_seq, hidden_dim]
    float* temp_buffer,           // Temporary buffer [batch_seq, intermediate_dim * 2]
    int batch_seq, int hidden_dim, int intermediate_dim,
    cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    // gate = input @ gate_proj^T
    float* gate = temp_buffer;
    dim3 grid_gate((intermediate_dim + TILE_SIZE - 1) / TILE_SIZE,
                   (batch_seq + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tiled<TILE_SIZE><<<grid_gate, block, 0, stream>>>(
        input, gate_proj, gate, batch_seq, intermediate_dim, hidden_dim, 1.0f, 0.0f);
    
    // up = input @ up_proj^T
    float* up = temp_buffer + batch_seq * intermediate_dim;
    gemm_tiled<TILE_SIZE><<<grid_gate, block, 0, stream>>>(
        input, up_proj, up, batch_seq, intermediate_dim, hidden_dim, 1.0f, 0.0f);
    
    // gate = SiLU(gate)
    dim3 grid_silu((batch_seq * intermediate_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_silu(BLOCK_SIZE);
    silu_kernel<<<grid_silu, block_silu, 0, stream>>>(gate, gate, batch_seq * intermediate_dim);
    
    // gate = gate * up (element-wise multiply)
    // Simple element-wise multiply kernel
    // For now, we'll use a fused approach in the next GEMM
    // Actually, we need to do element-wise multiply first
    // Let's create a simple kernel for this
    
    // output = (gate * up) @ down_proj^T
    // Since we need gate * up, let's compute it in a kernel
    // For simplicity, we'll do: temp = gate (already SiLU'd), then multiply with up
    // Then: output = temp @ down_proj^T
}

// Element-wise multiply kernel
__global__ void elementwise_multiply_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = a[idx] * b[idx];
}

// Complete MLP forward pass
__host__ void deepseek_v3_mlp_forward_complete(
    const float* input,           // [batch_seq, hidden_dim]
    const float* gate_proj,       // [hidden_dim, intermediate_dim]
    const float* up_proj,         // [hidden_dim, intermediate_dim]
    const float* down_proj,       // [intermediate_dim, hidden_dim]
    float* output,                // [batch_seq, hidden_dim]
    float* temp_buffer,           // [batch_seq, intermediate_dim * 2]
    int batch_seq, int hidden_dim, int intermediate_dim,
    cudaStream_t stream)
{
    dim3 block_2d(TILE_SIZE, TILE_SIZE);
    
    // Allocate pointers in temp buffer
    float* gate = temp_buffer;
    float* up = temp_buffer + batch_seq * intermediate_dim;
    float* activated = gate;  // Reuse gate buffer after SiLU
    
    // Step 1: gate = input @ gate_proj^T [batch_seq, intermediate_dim]
    dim3 grid_1((intermediate_dim + TILE_SIZE - 1) / TILE_SIZE,
                (batch_seq + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tiled<TILE_SIZE><<<grid_1, block_2d, 0, stream>>>(
        input, gate_proj, gate, batch_seq, intermediate_dim, hidden_dim, 1.0f, 0.0f);
    
    // Step 2: up = input @ up_proj^T [batch_seq, intermediate_dim]
    gemm_tiled<TILE_SIZE><<<grid_1, block_2d, 0, stream>>>(
        input, up_proj, up, batch_seq, intermediate_dim, hidden_dim, 1.0f, 0.0f);
    
    // Step 3: gate = SiLU(gate)
    dim3 grid_act((batch_seq * intermediate_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_act(BLOCK_SIZE);
    silu_kernel<<<grid_act, block_act, 0, stream>>>(gate, activated, batch_seq * intermediate_dim);
    
    // Step 4: activated = activated * up (element-wise)
    elementwise_multiply_kernel<<<grid_act, block_act, 0, stream>>>(
        activated, up, activated, batch_seq * intermediate_dim);
    
    // Step 5: output = activated @ down_proj^T [batch_seq, hidden_dim]
    dim3 grid_2((hidden_dim + TILE_SIZE - 1) / TILE_SIZE,
                (batch_seq + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tiled<TILE_SIZE><<<grid_2, block_2d, 0, stream>>>(
        activated, down_proj, output, batch_seq, hidden_dim, intermediate_dim, 1.0f, 0.0f);
}

// ===================== DeepseekV3NaiveMoe =====================
// Routed experts: applies expert MLPs based on routing
// This distributes experts across GPUs (expert parallelism)

__host__ void deepseek_v3_naive_moe_forward(
    const float* input,                   // [batch_seq, hidden_dim]
    const float* expert_weights,          // [batch_seq, top_k]
    const int* expert_indices,            // [batch_seq, top_k]
    const float** expert_gate_projs,      // [num_experts][hidden_dim, intermediate_dim]
    const float** expert_up_projs,        // [num_experts][hidden_dim, intermediate_dim]
    const float** expert_down_projs,      // [num_experts][intermediate_dim, hidden_dim]
    float* output,                        // [batch_seq, hidden_dim]
    float* temp_buffer,                   // Temporary buffer
    int batch_seq, int hidden_dim, int intermediate_dim,
    int num_experts, int top_k,
    int expert_start, int expert_end,     // Expert range for this GPU
    cudaStream_t stream)
{
    // Initialize output to zero
    CHECK_CUDA(cudaMemsetAsync(output, 0, batch_seq * hidden_dim * sizeof(float), stream));
    
    // For each expert in this GPU's range
    for (int exp_idx = expert_start; exp_idx < expert_end; ++exp_idx) {
        // Find tokens routed to this expert
        // Create a mask or index list of tokens that use this expert
        // For simplicity, we'll process all tokens and mask by expert index
        
        // Allocate temporary buffer for expert output
        float* expert_output = temp_buffer;
        
        // Apply expert MLP to all input tokens
        // (In production, would only process tokens routed to this expert)
        deepseek_v3_mlp_forward_complete(
            input,
            expert_gate_projs[exp_idx],
            expert_up_projs[exp_idx],
            expert_down_projs[exp_idx],
            expert_output,
            temp_buffer + batch_seq * hidden_dim,  // Use rest of buffer for MLP temp
            batch_seq, hidden_dim, intermediate_dim,
            stream);
        
        // Scale by expert weights and accumulate
        // Only accumulate for tokens where expert_indices contains exp_idx
        // For now, we'll do a simple weighted accumulation
        // In production, would use sparse accumulation
        
        // Weighted accumulation kernel
        // output += expert_weights * expert_output (where expert is selected)
    }
}

// Kernel to accumulate expert outputs with weights
// More efficient version that processes all features for a token
__global__ void accumulate_expert_outputs_kernel(
    const float* __restrict__ expert_output,  // [batch_seq, hidden_dim]
    const float* __restrict__ expert_weights, // [batch_seq, top_k]
    const int* __restrict__ expert_indices,   // [batch_seq, top_k]
    float* __restrict__ output,               // [batch_seq, hidden_dim]
    int batch_seq, int hidden_dim, int top_k, int expert_id)
{
    int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int feat_idx = threadIdx.x;
    
    if (token_idx >= batch_seq || feat_idx >= hidden_dim) return;
    
    // Check if this token routes to this expert
    float weight = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        if (expert_indices[token_idx * top_k + k] == expert_id) {
            weight = expert_weights[token_idx * top_k + k];
            break;
        }
    }
    
    if (weight > 0.0f) {
        float val = expert_output[token_idx * hidden_dim + feat_idx];
        atomicAdd(&output[token_idx * hidden_dim + feat_idx], weight * val);
    }
}

// Alternative: accumulate without atomics by using separate buffers per expert
// This is more efficient for expert parallelism
__global__ void weighted_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int batch_seq, int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * hidden_dim;
    
    if (idx >= total) return;
    
    int token_idx = idx / hidden_dim;
    int feat_idx = idx % hidden_dim;
    
    float weight = weights[token_idx];
    output[idx] += weight * a[idx];
}

// ===================== DeepseekV3MoE =====================
// Main MoE module combining router, shared experts, and routed experts

__host__ void deepseek_v3_moe_forward(
    const float* hidden_states,           // [batch_size, seq_len, hidden_dim]
    const float* router_logits,           // [batch_size, seq_len, num_experts]
    const float* shared_gate_proj,        // [hidden_dim, intermediate_dim]
    const float* shared_up_proj,          // [hidden_dim, intermediate_dim]
    const float* shared_down_proj,        // [intermediate_dim, hidden_dim]
    const float** expert_gate_projs,      // [num_experts][hidden_dim, intermediate_dim]
    const float** expert_up_projs,        // [num_experts][hidden_dim, intermediate_dim]
    const float** expert_down_projs,      // [num_experts][intermediate_dim, hidden_dim]
    float* output,                        // [batch_size, seq_len, hidden_dim]
    float* temp_buffer,                   // Large temporary buffer
    int batch_size, int seq_len, int hidden_dim, int intermediate_dim,
    int num_experts, int top_k,
    int rank, int world_size,
    ncclComm_t nccl_comm,
    cudaStream_t stream)
{
    int batch_seq = batch_size * seq_len;
    
    // Allocate device buffers
    int* d_expert_indices = nullptr;
    float* d_expert_weights = nullptr;
    float* d_shared_output = nullptr;
    float* d_routed_output = nullptr;
    float* d_router_logits_flat = nullptr;
    
    CHECK_CUDA(cudaMalloc(&d_expert_indices, batch_seq * top_k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_weights, batch_seq * top_k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shared_output, batch_seq * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_routed_output, batch_seq * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_router_logits_flat, batch_seq * num_experts * sizeof(float)));
    
    // Flatten router logits if needed
    CHECK_CUDA(cudaMemcpyAsync(d_router_logits_flat, router_logits,
                               batch_seq * num_experts * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    
    // Step 1: Router - compute top-k experts
    dim3 grid_router((batch_seq + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_router(BLOCK_SIZE);
    topk_router_kernel<<<grid_router, block_router, 0, stream>>>(
        d_router_logits_flat, d_expert_indices, d_expert_weights,
        batch_seq, num_experts, top_k);
    CHECK_CUDA(cudaPeekAtLastError());
    
    // Step 2: Shared experts (always applied)
    deepseek_v3_mlp_forward_complete(
        hidden_states,
        shared_gate_proj, shared_up_proj, shared_down_proj,
        d_shared_output,
        temp_buffer,
        batch_seq, hidden_dim, intermediate_dim,
        stream);
    
    // Step 3: Routed experts (expert parallelism)
    // Distribute experts across GPUs
    int experts_per_gpu = (num_experts + world_size - 1) / world_size;
    int expert_start = rank * experts_per_gpu;
    int expert_end = std::min(expert_start + experts_per_gpu, num_experts);
    
    // Initialize routed output
    CHECK_CUDA(cudaMemsetAsync(d_routed_output, 0, batch_seq * hidden_dim * sizeof(float), stream));
    
    // Process experts assigned to this GPU
    for (int exp_idx = expert_start; exp_idx < expert_end; ++exp_idx) {
        // Apply expert MLP
        float* expert_temp = temp_buffer + batch_seq * hidden_dim * 2;  // Use MLP temp space
        deepseek_v3_mlp_forward_complete(
            hidden_states,
            expert_gate_projs[exp_idx],
            expert_up_projs[exp_idx],
            expert_down_projs[exp_idx],
            expert_temp,  // Expert output
            temp_buffer + batch_seq * hidden_dim * 3,  // MLP internal temp
            batch_seq, hidden_dim, intermediate_dim,
            stream);
        
        // Accumulate with weights
        dim3 grid_acc(batch_seq);
        dim3 block_acc(hidden_dim);
        accumulate_expert_outputs_kernel<<<grid_acc, block_acc, 0, stream>>>(
            expert_temp, d_expert_weights, d_expert_indices,
            d_routed_output, batch_seq, hidden_dim, top_k, exp_idx);
        CHECK_CUDA(cudaPeekAtLastError());
    }
    
    // Step 4: AllReduce routed outputs across GPUs (expert parallelism)
    CHECK_NCCL(ncclAllReduce(
        (const void*)d_routed_output, (void*)d_routed_output,
        batch_seq * hidden_dim, ncclFloat, ncclSum, nccl_comm, stream));
    
    // Step 5: Combine shared and routed outputs
    // output = shared_output + routed_output
    dim3 grid_combine((batch_seq * hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_combine(BLOCK_SIZE);
    // Simple add kernel
    // For now, we'll use a simple kernel
}

// Element-wise add kernel
__global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = a[idx] + b[idx];
}

// Complete MoE forward pass
__host__ void deepseek_v3_moe_forward_complete(
    const float* hidden_states,           // [batch_size, seq_len, hidden_dim]
    const float* router_logits,           // [batch_size, seq_len, num_experts]
    const float* shared_gate_proj,        // [hidden_dim, intermediate_dim]
    const float* shared_up_proj,          // [hidden_dim, intermediate_dim]
    const float* shared_down_proj,        // [intermediate_dim, hidden_dim]
    const float** expert_gate_projs,      // [num_experts][hidden_dim, intermediate_dim]
    const float** expert_up_projs,        // [num_experts][hidden_dim, intermediate_dim]
    const float** expert_down_projs,      // [num_experts][intermediate_dim, hidden_dim]
    float* output,                        // [batch_size, seq_len, hidden_dim]
    float* temp_buffer,                   // Large temporary buffer
    int batch_size, int seq_len, int hidden_dim, int intermediate_dim,
    int num_experts, int top_k,
    int rank, int world_size,
    ncclComm_t nccl_comm,
    cudaStream_t stream)
{
    int batch_seq = batch_size * seq_len;
    
    // Allocate device buffers
    int* d_expert_indices = nullptr;
    float* d_expert_weights = nullptr;
    float* d_shared_output = nullptr;
    float* d_routed_output = nullptr;
    
    CHECK_CUDA(cudaMalloc(&d_expert_indices, batch_seq * top_k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_weights, batch_seq * top_k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shared_output, batch_seq * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_routed_output, batch_seq * hidden_dim * sizeof(float)));
    
    // Step 1: Router
    dim3 grid_router((batch_seq + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_router(BLOCK_SIZE);
    topk_router_kernel<<<grid_router, block_router, 0, stream>>>(
        router_logits, d_expert_indices, d_expert_weights,
        batch_seq, num_experts, top_k);
    CHECK_CUDA(cudaPeekAtLastError());
    
    // Step 2: Shared experts
    // Use temp_buffer for MLP intermediates (gate, up projections)
    deepseek_v3_mlp_forward_complete(
        hidden_states,
        shared_gate_proj, shared_up_proj, shared_down_proj,
        d_shared_output,
        temp_buffer,  // MLP uses temp_buffer for gate and up projections
        batch_seq, hidden_dim, intermediate_dim,
        stream);
    
    // Step 3: Routed experts (expert parallelism)
    int experts_per_gpu = (num_experts + world_size - 1) / world_size;
    int expert_start = rank * experts_per_gpu;
    int expert_end = std::min(expert_start + experts_per_gpu, num_experts);
    
    CHECK_CUDA(cudaMemsetAsync(d_routed_output, 0, batch_seq * hidden_dim * sizeof(float), stream));
    
    // Process experts on this GPU
    // Temp buffer layout for expert MLP:
    // - temp_buffer: MLP intermediates (gate, up) - reused for each expert
    // - d_expert_output_temp: Expert output - allocated separately
    float* expert_mlp_temp = temp_buffer;  // MLP temp space
    
    // Allocate buffers for expert processing
    float* d_expert_mask_weights = nullptr;
    float* d_expert_output_temp = nullptr;  // Temporary buffer for expert output
    CHECK_CUDA(cudaMalloc(&d_expert_mask_weights, batch_seq * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_output_temp, batch_seq * hidden_dim * sizeof(float)));
    
    for (int exp_idx = expert_start; exp_idx < expert_end; ++exp_idx) {
        // Extract weights for this expert
        dim3 grid_mask((batch_seq + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block_mask(BLOCK_SIZE);
        extract_expert_weights_kernel<<<grid_mask, block_mask, 0, stream>>>(
            d_expert_indices, d_expert_weights, d_expert_mask_weights,
            batch_seq, top_k, exp_idx);
        CHECK_CUDA(cudaPeekAtLastError());
        
        // Apply expert MLP
        // Use d_expert_output_temp for expert output, temp_buffer for MLP intermediates
        deepseek_v3_mlp_forward_complete(
            hidden_states,
            expert_gate_projs[exp_idx],
            expert_up_projs[exp_idx],
            expert_down_projs[exp_idx],
            d_expert_output_temp,  // Expert output goes here
            expert_mlp_temp,  // MLP uses temp_buffer for gate and up
            batch_seq, hidden_dim, intermediate_dim,
            stream);
        
        // Accumulate with weights using weighted_add
        dim3 grid_add((batch_seq * hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block_add(BLOCK_SIZE);
        weighted_add_kernel<<<grid_add, block_add, 0, stream>>>(
            d_expert_output_temp, d_expert_mask_weights, d_routed_output,
            batch_seq, hidden_dim);
        CHECK_CUDA(cudaPeekAtLastError());
    }
    
    CHECK_CUDA(cudaFree(d_expert_mask_weights));
    CHECK_CUDA(cudaFree(d_expert_output_temp));
    
    // Step 4: AllReduce routed outputs
    CHECK_NCCL(ncclAllReduce(
        (const void*)d_routed_output, (void*)d_routed_output,
        batch_seq * hidden_dim, ncclFloat, ncclSum, nccl_comm, stream));
    
    // Step 5: Combine outputs
    dim3 grid_add((batch_seq * hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_add(BLOCK_SIZE);
    add_kernel<<<grid_add, block_add, 0, stream>>>(
        d_shared_output, d_routed_output, output, batch_seq * hidden_dim);
    CHECK_CUDA(cudaPeekAtLastError());
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_expert_indices));
    CHECK_CUDA(cudaFree(d_expert_weights));
    CHECK_CUDA(cudaFree(d_shared_output));
    CHECK_CUDA(cudaFree(d_routed_output));
}

// Kernel to extract expert weights
__global__ void extract_expert_weights_kernel(
    const int* __restrict__ expert_indices,   // [batch_seq, top_k]
    const float* __restrict__ expert_weights, // [batch_seq, top_k]
    float* __restrict__ output_weights,       // [batch_seq]
    int batch_seq, int top_k, int expert_id)
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_seq) return;
    
    float weight = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        if (expert_indices[token_idx * top_k + k] == expert_id) {
            weight = expert_weights[token_idx * top_k + k];
            break;
        }
    }
    output_weights[token_idx] = weight;
}

// ===================== Main Function =====================
int main(int argc, char** argv) {
    // Initialize MPI
    CHECK_MPI(MPI_Init(&argc, &argv));
    int rank = 0, world_size = 1;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    
    // Parse arguments
    if (argc < 6) {
        if (rank == 0) {
            printf("Usage: %s <batch_size> <seq_len> <hidden_dim> <num_experts> <top_k> [intermediate_dim]\n", argv[0]);
            printf("Example: %s 2 128 512 8 2 2048\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    int batch_size = atoi(argv[1]);
    int seq_len = atoi(argv[2]);
    int hidden_dim = atoi(argv[3]);
    int num_experts = atoi(argv[4]);
    int top_k = atoi(argv[5]);
    int intermediate_dim = (argc > 6) ? atoi(argv[6]) : hidden_dim * 4;
    
    if (rank == 0) {
        printf("DeepseekV3MoE Configuration:\n");
        printf("  Batch size: %d\n", batch_size);
        printf("  Sequence length: %d\n", seq_len);
        printf("  Hidden dimension: %d\n", hidden_dim);
        printf("  Intermediate dimension: %d\n", intermediate_dim);
        printf("  Number of experts: %d\n", num_experts);
        printf("  Top-k: %d\n", top_k);
        printf("  World size: %d\n", world_size);
    }
    
    // Set device
    CHECK_CUDA(cudaSetDevice(rank));
    
    // Initialize NCCL
    ncclUniqueId ncclId;
    if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&ncclId));
    CHECK_MPI(MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclComm_t nccl_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, world_size, ncclId, rank));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    int batch_seq = batch_size * seq_len;
    
    // Allocate and initialize host data
    std::vector<float> h_hidden_states(batch_seq * hidden_dim);
    std::vector<float> h_router_logits(batch_seq * num_experts);
    std::vector<float> h_shared_gate_proj(hidden_dim * intermediate_dim);
    std::vector<float> h_shared_up_proj(hidden_dim * intermediate_dim);
    std::vector<float> h_shared_down_proj(intermediate_dim * hidden_dim);
    
    fill_matrix(h_hidden_states.data(), batch_seq * hidden_dim);
    fill_matrix(h_router_logits.data(), batch_seq * num_experts);
    fill_matrix(h_shared_gate_proj.data(), hidden_dim * intermediate_dim);
    fill_matrix(h_shared_up_proj.data(), hidden_dim * intermediate_dim);
    fill_matrix(h_shared_down_proj.data(), intermediate_dim * hidden_dim);
    
    // Allocate expert weights (one set per expert)
    std::vector<std::vector<float>> h_expert_gate_projs(num_experts);
    std::vector<std::vector<float>> h_expert_up_projs(num_experts);
    std::vector<std::vector<float>> h_expert_down_projs(num_experts);
    
    for (int i = 0; i < num_experts; ++i) {
        h_expert_gate_projs[i].resize(hidden_dim * intermediate_dim);
        h_expert_up_projs[i].resize(hidden_dim * intermediate_dim);
        h_expert_down_projs[i].resize(intermediate_dim * hidden_dim);
        fill_matrix(h_expert_gate_projs[i].data(), hidden_dim * intermediate_dim);
        fill_matrix(h_expert_up_projs[i].data(), hidden_dim * intermediate_dim);
        fill_matrix(h_expert_down_projs[i].data(), intermediate_dim * hidden_dim);
    }
    
    // Allocate device memory
    float* d_hidden_states = nullptr;
    float* d_router_logits = nullptr;
    float* d_shared_gate_proj = nullptr;
    float* d_shared_up_proj = nullptr;
    float* d_shared_down_proj = nullptr;
    float* d_output = nullptr;
    
    CHECK_CUDA(cudaMalloc(&d_hidden_states, batch_seq * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_router_logits, batch_seq * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shared_gate_proj, hidden_dim * intermediate_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shared_up_proj, hidden_dim * intermediate_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shared_down_proj, intermediate_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, batch_seq * hidden_dim * sizeof(float)));
    
    // Allocate expert weights on device
    std::vector<float*> d_expert_gate_projs(num_experts);
    std::vector<float*> d_expert_up_projs(num_experts);
    std::vector<float*> d_expert_down_projs(num_experts);
    
    for (int i = 0; i < num_experts; ++i) {
        CHECK_CUDA(cudaMalloc(&d_expert_gate_projs[i], hidden_dim * intermediate_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_expert_up_projs[i], hidden_dim * intermediate_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_expert_down_projs[i], intermediate_dim * hidden_dim * sizeof(float)));
    }
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpyAsync(d_hidden_states, h_hidden_states.data(),
                               batch_seq * hidden_dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_router_logits, h_router_logits.data(),
                               batch_seq * num_experts * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_shared_gate_proj, h_shared_gate_proj.data(),
                               hidden_dim * intermediate_dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_shared_up_proj, h_shared_up_proj.data(),
                               hidden_dim * intermediate_dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_shared_down_proj, h_shared_down_proj.data(),
                               intermediate_dim * hidden_dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    
    for (int i = 0; i < num_experts; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d_expert_gate_projs[i], h_expert_gate_projs[i].data(),
                                   hidden_dim * intermediate_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_expert_up_projs[i], h_expert_up_projs[i].data(),
                                   hidden_dim * intermediate_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_expert_down_projs[i], h_expert_down_projs[i].data(),
                                   intermediate_dim * hidden_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
    
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Prepare pointer arrays for experts
    std::vector<const float*> expert_gate_proj_ptrs(num_experts);
    std::vector<const float*> expert_up_proj_ptrs(num_experts);
    std::vector<const float*> expert_down_proj_ptrs(num_experts);
    
    for (int i = 0; i < num_experts; ++i) {
        expert_gate_proj_ptrs[i] = d_expert_gate_projs[i];
        expert_up_proj_ptrs[i] = d_expert_up_projs[i];
        expert_down_proj_ptrs[i] = d_expert_down_projs[i];
    }
    
    // Allocate temporary buffer (large enough for MLP operations)
    // Need for MLP: batch_seq * intermediate_dim * 2 (gate, up projections)
    // This buffer is reused for each expert MLP computation
    size_t mlp_temp_size = (size_t)batch_seq * intermediate_dim * 2;
    float* d_temp_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_temp_buffer, mlp_temp_size * sizeof(float)));
    
    // Verify allocation
    if (d_temp_buffer == nullptr) {
        fprintf(stderr, "Rank %d: Failed to allocate temp buffer of size %zu bytes\n",
                rank, mlp_temp_size * sizeof(float));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == 0) {
        printf("Allocated MLP temp buffer: %.2f MB\n", 
               mlp_temp_size * sizeof(float) / (1024.0 * 1024.0));
    }
    
    // Run MoE forward pass
    if (rank == 0) {
        printf("Running DeepseekV3MoE forward pass...\n");
    }
    
    deepseek_v3_moe_forward_complete(
        d_hidden_states,
        d_router_logits,
        d_shared_gate_proj, d_shared_up_proj, d_shared_down_proj,
        expert_gate_proj_ptrs.data(),
        expert_up_proj_ptrs.data(),
        expert_down_proj_ptrs.data(),
        d_output,
        d_temp_buffer,
        batch_size, seq_len, hidden_dim, intermediate_dim,
        num_experts, top_k,
        rank, world_size,
        nccl_comm,
        stream);
    
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Copy output back
    std::vector<float> h_output(batch_seq * hidden_dim);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output,
                          batch_seq * hidden_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    if (rank == 0) {
        printf("MoE forward pass completed.\n");
        printf("Output sample: [0,0]=%.6f, [0,1]=%.6f, [last,last]=%.6f\n",
               h_output[0], h_output[1], h_output[batch_seq * hidden_dim - 1]);
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_hidden_states));
    CHECK_CUDA(cudaFree(d_router_logits));
    CHECK_CUDA(cudaFree(d_shared_gate_proj));
    CHECK_CUDA(cudaFree(d_shared_up_proj));
    CHECK_CUDA(cudaFree(d_shared_down_proj));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp_buffer));
    
    for (int i = 0; i < num_experts; ++i) {
        CHECK_CUDA(cudaFree(d_expert_gate_projs[i]));
        CHECK_CUDA(cudaFree(d_expert_up_projs[i]));
        CHECK_CUDA(cudaFree(d_expert_down_projs[i]));
    }
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NCCL(ncclCommDestroy(nccl_comm));
    CHECK_MPI(MPI_Finalize());
    
    return 0;
}

