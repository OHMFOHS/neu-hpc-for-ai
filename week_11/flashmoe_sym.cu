// ============================================================================
// FlashMoE (Minimal Symmetric Layout Demo)
// - Single-process multi-GPU (no MPI)
// - Symmetric tensor layout L with AllToAll (send/recv) and overlap
// - Uses existing tiny test weights (H=2, I=4, E=16, K=2, T=6)
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)
#define NCCL_CHECK(call) do { ncclResult_t _nccl_res = (call); if (_nccl_res != ncclSuccess) { \
  fprintf(stderr,"NCCL error %s at %s:%d\n", ncclGetErrorString(_nccl_res), __FILE__, __LINE__); exit(1);} } while(0)

// Tiny config (matches provided JSON fixtures)
#define HIDDEN_SIZE 2
#define INTERMEDIATE_SIZE 4
#define N_ROUTED_EXPERTS 16
#define K_TOP 2
#define MAX_TOKENS 6

// We reuse weights/test vectors from the reference implementation
static const float W_ROUTER[N_ROUTED_EXPERTS * HIDDEN_SIZE] = {
    -0.6661239862442017f, -0.13873639702796936f,
    -0.339631587266922f,  -0.18859760463237762f,
    -0.6246570944786072f,  0.28375449776649475f,
    -0.6338542103767395f, -0.04505790024995804f,
     0.2457990050315857f, -0.2383487969636917f,
     0.4012238085269928f,  0.08918479830026627f,
     0.3886972963809967f,  0.45368340611457825f,
    -0.31215009093284607f, 0.256978303194046f,
    -0.3059298098087311f,  0.22165030241012573f,
    -0.3694550096988678f,  0.3270857036113739f,
     0.14315040409564972f, -0.27672138810157776f,
    -0.346832811832428f,   0.18294529616832733f,
     0.6597462892532349f,  0.33933940529823303f,
    -0.06828120350837708f, -0.034321799874305725f,
     0.40191149711608887f, -0.49144458770751953f,
     0.23507679998874664f, -0.2343025952577591f
};
static const float W_GATE_PROJ[INTERMEDIATE_SIZE * HIDDEN_SIZE] = {
    -0.6661239862442017f, -0.13873639702796936f,
    -0.339631587266922f,  -0.18859760463237762f,
    -0.6246570944786072f,  0.28375449776649475f,
    -0.6338542103767395f, -0.04505790024995804f
};
static const float W_UP_PROJ[INTERMEDIATE_SIZE * HIDDEN_SIZE] = {
     0.2457990050315857f, -0.2383487969636917f,
     0.4012238085269928f,  0.08918479830026627f,
     0.3886972963809967f,  0.45368340611457825f,
    -0.31215009093284607f, 0.256978303194046f
};
static const float W_DOWN_PROJ[HIDDEN_SIZE * INTERMEDIATE_SIZE] = {
    -0.21632499992847443f,  0.15673039853572845f, -0.2612442076206207f,  0.231284499168396f,
     0.10122259706258774f, -0.19567160308361053f, -0.24524779617786407f,  0.12936189770698547f
};
static const float TEST_INPUT[MAX_TOKENS * HIDDEN_SIZE] = {
    -0.5338150262832642f, -0.9619768261909485f,
    -1.7629637718200684f,  0.4865371882915497f,
     2.1058542728424072f, -0.5918406844139099f,
    -1.242525577545166f,  -0.7119956016540527f,
    -1.9618711471557617f, -0.8310434818267822f,
     1.7153427600860596f,  1.2089829444885254f
};

// Device const weights
__constant__ float d_W_ROUTER[N_ROUTED_EXPERTS * HIDDEN_SIZE];
__constant__ float d_W_GATE_PROJ[INTERMEDIATE_SIZE * HIDDEN_SIZE];
__constant__ float d_W_UP_PROJ[INTERMEDIATE_SIZE * HIDDEN_SIZE];
__constant__ float d_W_DOWN_PROJ[HIDDEN_SIZE * INTERMEDIATE_SIZE];

__device__ inline float silu_dev(float x) { return x / (1.0f + expf(-x)); }

// Router logits
__global__ void router_kernel(const float* X, float* logits, int T) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* x = &X[t * HIDDEN_SIZE];
  float* out = &logits[t * N_ROUTED_EXPERTS];
  for (int e = 0; e < N_ROUTED_EXPERTS; ++e) {
    float v = 0.f;
    v += d_W_ROUTER[e * HIDDEN_SIZE + 0] * x[0];
    v += d_W_ROUTER[e * HIDDEN_SIZE + 1] * x[1];
    out[e] = v;
  }
}

// Softmax + topk (K=2) fused
__global__ void softmax_topk_kernel(const float* logits, int* top_idx, float* top_w, int T) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* l = &logits[t * N_ROUTED_EXPERTS];
  float maxv = l[0];
  for (int i = 1; i < N_ROUTED_EXPERTS; ++i) maxv = maxv > l[i] ? maxv : l[i];
  float sum = 0.f;
  float p[N_ROUTED_EXPERTS];
  for (int i = 0; i < N_ROUTED_EXPERTS; ++i) { p[i] = expf(l[i] - maxv); sum += p[i]; }
  for (int i = 0; i < N_ROUTED_EXPERTS; ++i) p[i] /= sum;
  int best0 = 0, best1 = 1;
  if (p[1] > p[0]) { best0 = 1; best1 = 0; }
  for (int i = 2; i < N_ROUTED_EXPERTS; ++i) {
    if (p[i] > p[best0]) { best1 = best0; best0 = i; }
    else if (p[i] > p[best1]) { best1 = i; }
  }
  float w0 = p[best0], w1 = p[best1], norm = w0 + w1 + 1e-9f;
  top_idx[t * K_TOP + 0] = best0; top_idx[t * K_TOP + 1] = best1;
  top_w[t * K_TOP + 0] = w0 / norm; top_w[t * K_TOP + 1] = w1 / norm;
}

// MLP
__device__ void mlp_forward_device(const float* x, float* y) {
  float inter[INTERMEDIATE_SIZE];
  #pragma unroll
  for (int i = 0; i < INTERMEDIATE_SIZE; ++i) {
    float g = d_W_GATE_PROJ[i * HIDDEN_SIZE + 0] * x[0] +
              d_W_GATE_PROJ[i * HIDDEN_SIZE + 1] * x[1];
    float u = d_W_UP_PROJ[i * HIDDEN_SIZE + 0] * x[0] +
              d_W_UP_PROJ[i * HIDDEN_SIZE + 1] * x[1];
    inter[i] = silu_dev(g) * u;
  }
  y[0] = d_W_DOWN_PROJ[0] * inter[0] + d_W_DOWN_PROJ[1] * inter[1] +
         d_W_DOWN_PROJ[2] * inter[2] + d_W_DOWN_PROJ[3] * inter[3];
  y[1] = d_W_DOWN_PROJ[4] * inter[0] + d_W_DOWN_PROJ[5] * inter[1] +
         d_W_DOWN_PROJ[6] * inter[2] + d_W_DOWN_PROJ[7] * inter[3];
}

// Unified symmetric tensor L: L[0] = dispatch phase, L[1] = combine phase
// Physical layout: L_data[phase][peer][slot][H], W_data[phase][peer][slot], T_data[phase][peer][slot], E_data[phase][peer][slot]
// Semantic: L[0] = send, L[1] = recv (after AllToAll)
__global__ void dispatch_kernel(const float* X, const int* top_idx, const float* top_w,
                                float* L_data, float* W_data, int* T_data, int* E_data,
                                int* token_to_slot_map, int* counters, int T, int world_size, int experts_per_gpu, int max_tokens_per_gpu) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* x = &X[t * HIDDEN_SIZE];
  int slot_base = t * K_TOP; // Each token has K_TOP slots in the map
  for (int k = 0; k < K_TOP; ++k) {
    int eid = top_idx[t * K_TOP + k];
    float w = top_w[t * K_TOP + k];
    int tgt = eid / experts_per_gpu;
    int slot = atomicAdd(&counters[tgt], 1);
    if (slot >= max_tokens_per_gpu) return; // overflow guard
    // Write to L[0] (dispatch phase)
    size_t base = ((size_t)tgt * max_tokens_per_gpu + slot) * HIDDEN_SIZE;
    L_data[base + 0] = x[0];
    L_data[base + 1] = x[1];
    W_data[tgt * max_tokens_per_gpu + slot] = w;
    T_data[tgt * max_tokens_per_gpu + slot] = t;
    E_data[tgt * max_tokens_per_gpu + slot] = eid;
    // Build token->slot mapping for O(T×K) combine: token_to_slot_map[token*K + k] = (peer << 16) | slot
    token_to_slot_map[slot_base + k] = (tgt << 16) | slot;
  }
}

// Expert compute on L[1] (received after AllToAll), write back in-place
// This runs in parallel with AllToAll communication (overlap)
__global__ void expert_apply_kernel(float* L_data, float* W_data, int* T_data, int* E_data,
                                    int* recv_counts, int max_tokens_per_gpu, int rank, int experts_per_gpu) {
  int peer = blockIdx.x; // one block per peer
  int idx = threadIdx.x;
  int count = recv_counts[peer];
  if (idx >= count) return;
  // Access L[1] (recv phase) - same physical buffer after AllToAll
  size_t base = ((size_t)peer * max_tokens_per_gpu + idx) * HIDDEN_SIZE;
  float tmp[HIDDEN_SIZE];
  mlp_forward_device(&L_data[base], tmp);
  int token = T_data[peer * max_tokens_per_gpu + idx];
  float w = W_data[peer * max_tokens_per_gpu + idx];
  // Write weighted output back to L[1] slot (for combine)
  L_data[base + 0] = tmp[0] * w;
  L_data[base + 1] = tmp[1] * w;
  // Keep token id for combine
  T_data[peer * max_tokens_per_gpu + idx] = token;
}

// Combine: O(T×K) - read from L[0] using token_to_slot_map
// After second AllToAll, data should be back at original locations (AllToAll is symmetric)
// So we can use token_to_slot_map directly for O(T×K) combine
__global__ void combine_kernel(const float* shared_out, const float* L_data, const int* token_to_slot_map,
                               float* Y, int T, int max_tokens_per_gpu) {
  int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= T) return;
  float acc0 = shared_out[token * HIDDEN_SIZE + 0];
  float acc1 = shared_out[token * HIDDEN_SIZE + 1];
  
  // Direct lookup using token_to_slot_map: each token has exactly K_TOP entries
  int map_base = token * K_TOP;
  for (int k = 0; k < K_TOP; ++k) {
    int map_val = token_to_slot_map[map_base + k];
    if (map_val < 0) continue; // invalid entry (shouldn't happen, but guard anyway)
    int peer = map_val >> 16;
    int slot = map_val & 0xFFFF;
    size_t base = ((size_t)peer * max_tokens_per_gpu + slot) * HIDDEN_SIZE;
    acc0 += L_data[base + 0];
    acc1 += L_data[base + 1];
  }
  Y[token * HIDDEN_SIZE + 0] = acc0;
  Y[token * HIDDEN_SIZE + 1] = acc1;
}

// Shared expert
__global__ void shared_expert_kernel(const float* X, float* Y, int T) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  float tmp[HIDDEN_SIZE];
  mlp_forward_device(&X[t * HIDDEN_SIZE], tmp);
  Y[t * HIDDEN_SIZE + 0] = tmp[0];
  Y[t * HIDDEN_SIZE + 1] = tmp[1];
}

// ============================================================================
// CPU Reference Implementation (for numerical verification)
// ============================================================================

static inline float silu_cpu(float x) {
  return x / (1.0f + std::exp(-x));
}

// CPU MLP forward
void mlp_forward_cpu(const float* x, float* y) {
  float inter[INTERMEDIATE_SIZE];
  for (int i = 0; i < INTERMEDIATE_SIZE; ++i) {
    float g = W_GATE_PROJ[i * HIDDEN_SIZE + 0] * x[0] +
              W_GATE_PROJ[i * HIDDEN_SIZE + 1] * x[1];
    float u = W_UP_PROJ[i * HIDDEN_SIZE + 0] * x[0] +
              W_UP_PROJ[i * HIDDEN_SIZE + 1] * x[1];
    inter[i] = silu_cpu(g) * u;
  }
  y[0] = W_DOWN_PROJ[0] * inter[0] + W_DOWN_PROJ[1] * inter[1] +
         W_DOWN_PROJ[2] * inter[2] + W_DOWN_PROJ[3] * inter[3];
  y[1] = W_DOWN_PROJ[4] * inter[0] + W_DOWN_PROJ[5] * inter[1] +
         W_DOWN_PROJ[6] * inter[2] + W_DOWN_PROJ[7] * inter[3];
}

// CPU Router forward
void router_forward_cpu(const float* X, float* router_logits, int T) {
  for (int t = 0; t < T; ++t) {
    const float* x = &X[t * HIDDEN_SIZE];
    float* logits = &router_logits[t * N_ROUTED_EXPERTS];
    for (int e = 0; e < N_ROUTED_EXPERTS; ++e) {
      logits[e] = W_ROUTER[e * HIDDEN_SIZE + 0] * x[0] +
                  W_ROUTER[e * HIDDEN_SIZE + 1] * x[1];
    }
  }
}

// CPU Softmax + Top-K
void softmax_topk_cpu(const float* logits, int* top_idx, float* top_w, int T) {
  for (int t = 0; t < T; ++t) {
    const float* l = &logits[t * N_ROUTED_EXPERTS];
    // Softmax
    float maxv = l[0];
    for (int i = 1; i < N_ROUTED_EXPERTS; ++i) {
      if (l[i] > maxv) maxv = l[i];
    }
    float sum = 0.0f;
    float p[N_ROUTED_EXPERTS];
    for (int i = 0; i < N_ROUTED_EXPERTS; ++i) {
      p[i] = std::exp(l[i] - maxv);
      sum += p[i];
    }
    for (int i = 0; i < N_ROUTED_EXPERTS; ++i) {
      p[i] /= sum;
    }
    // Top-K (K=2)
    int best0 = 0, best1 = 1;
    if (p[1] > p[0]) { best0 = 1; best1 = 0; }
    for (int i = 2; i < N_ROUTED_EXPERTS; ++i) {
      if (p[i] > p[best0]) {
        best1 = best0;
        best0 = i;
      } else if (p[i] > p[best1]) {
        best1 = i;
      }
    }
    // Renormalize top-K
    float w0 = p[best0], w1 = p[best1], norm = w0 + w1 + 1e-9f;
    top_idx[t * K_TOP + 0] = best0;
    top_idx[t * K_TOP + 1] = best1;
    top_w[t * K_TOP + 0] = w0 / norm;
    top_w[t * K_TOP + 1] = w1 / norm;
  }
}

// CPU MoE forward (reference implementation)
void moe_forward_cpu(const float* X, float* Y, int T) {
  // Step 1: Router
  float* router_logits = (float*)malloc(T * N_ROUTED_EXPERTS * sizeof(float));
  router_forward_cpu(X, router_logits, T);
  
  // Step 2: Softmax + Top-K
  int* top_idx = (int*)malloc(T * K_TOP * sizeof(int));
  float* top_w = (float*)malloc(T * K_TOP * sizeof(float));
  softmax_topk_cpu(router_logits, top_idx, top_w, T);
  
  // Step 3: Routed experts (all experts use same weights in this simplified version)
  for (int t = 0; t < T; ++t) {
    const float* x = &X[t * HIDDEN_SIZE];
    float routed_out[HIDDEN_SIZE] = {0.0f};
    for (int k = 0; k < K_TOP; ++k) {
      int expert_id = top_idx[t * K_TOP + k];
      float weight = top_w[t * K_TOP + k];
      float expert_out[HIDDEN_SIZE];
      mlp_forward_cpu(x, expert_out);
      routed_out[0] += weight * expert_out[0];
      routed_out[1] += weight * expert_out[1];
    }
    Y[t * HIDDEN_SIZE + 0] = routed_out[0];
    Y[t * HIDDEN_SIZE + 1] = routed_out[1];
  }
  
  // Step 4: Shared expert
  for (int t = 0; t < T; ++t) {
    const float* x = &X[t * HIDDEN_SIZE];
    float shared_out[HIDDEN_SIZE];
    mlp_forward_cpu(x, shared_out);
    Y[t * HIDDEN_SIZE + 0] += shared_out[0];
    Y[t * HIDDEN_SIZE + 1] += shared_out[1];
  }
  
  free(router_logits);
  free(top_idx);
  free(top_w);
}

int main() {
  // Check CUDA driver version compatibility
  int driver_version = 0;
  cudaError_t driver_check = cudaDriverGetVersion(&driver_version);
  if (driver_check != cudaSuccess) {
    fprintf(stderr, "Warning: Could not get CUDA driver version: %s\n", 
            cudaGetErrorString(driver_check));
  } else {
    int runtime_version = 0;
    cudaError_t runtime_check = cudaRuntimeGetVersion(&runtime_version);
    if (runtime_check == cudaSuccess) {
      printf("CUDA Driver Version: %d, Runtime Version: %d\n", 
             driver_version / 1000, runtime_version / 1000);
      if (driver_version < runtime_version) {
        fprintf(stderr, "Warning: CUDA driver version (%d) may be insufficient for runtime (%d)\n",
                driver_version / 1000, runtime_version / 1000);
      }
    }
  }
  
  int world_size;
  CUDA_CHECK(cudaGetDeviceCount(&world_size));
  if (world_size < 1) { 
    printf("No GPUs found\n"); 
    return 0; 
  }
  printf("Found %d GPU(s)\n", world_size);
  
  // Check each GPU and establish CUDA context on each device
  // This is important for NCCL initialization
  printf("Establishing CUDA contexts on %d GPU(s)...\n", world_size);
  fflush(stdout);
  for (int i = 0; i < world_size; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    printf("GPU %d: %s (Compute %d.%d)\n", i, prop.name, prop.major, prop.minor);
    fflush(stdout);
    
    // Establish CUDA context by allocating a small buffer
    void* dummy;
    CUDA_CHECK(cudaMalloc(&dummy, 1024));
    CUDA_CHECK(cudaFree(dummy));
    printf("  -> CUDA context established on GPU %d\n", i);
    fflush(stdout);
  }
  
  const int T = MAX_TOKENS;
  const int experts_per_gpu = (N_ROUTED_EXPERTS + world_size - 1) / world_size;
  const int max_tokens_per_gpu = (T * K_TOP + world_size - 1) / world_size; // upper bound

  // NCCL setup - initialize after CUDA contexts are established
  printf("\nInitializing NCCL for %d GPU(s)...\n", world_size);
  fflush(stdout);
  std::vector<ncclComm_t> comms(world_size);
  std::vector<int> devs(world_size);
  for (int i = 0; i < world_size; ++i) devs[i] = i;
  
  // Set device to 0 before NCCL init (some NCCL implementations require this)
  CUDA_CHECK(cudaSetDevice(0));
  printf("Setting device to 0 before NCCL init...\n");
  fflush(stdout);
  
  printf("Calling ncclCommInitAll...\n");
  fflush(stdout);
  ncclResult_t nccl_init_res = ncclCommInitAll(comms.data(), world_size, devs.data());
  printf("ncclCommInitAll returned\n");
  fflush(stdout);
  
  if (nccl_init_res != ncclSuccess) {
    fprintf(stderr, "NCCL initialization failed: %s\n", ncclGetErrorString(nccl_init_res));
    fprintf(stderr, "This may be due to CUDA driver/runtime version mismatch.\n");
    fprintf(stderr, "Try setting NCCL_DEBUG=INFO for more details.\n");
    return 1;
  }
  printf("✅ NCCL initialized successfully\n");
  fflush(stdout);

  // Streams: comp for computation, comm for communication (overlap)
  std::vector<cudaStream_t> s_comp(world_size), s_comm(world_size);
  std::vector<cudaEvent_t> dispatch_done(world_size), comm_done(world_size);
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaStreamCreate(&s_comp[r]));
    CUDA_CHECK(cudaStreamCreate(&s_comm[r]));
    CUDA_CHECK(cudaEventCreate(&dispatch_done[r]));
    CUDA_CHECK(cudaEventCreate(&comm_done[r]));
  }

  // Unified symmetric tensor L: L[0] = dispatch, L[1] = combine (after AllToAll)
  // Physical: separate arrays for clarity, but semantic is unified L
  std::vector<float*> d_X(world_size), d_logits(world_size), d_shared_out(world_size), d_output(world_size);
  std::vector<int*> d_top_idx(world_size), d_dispatch_counts(world_size), d_recv_counts(world_size);
  std::vector<float*> d_top_w(world_size);
  // L[0] phase buffers (dispatch)
  std::vector<float*> L0_data(world_size), W0_data(world_size);
  std::vector<int*> T0_data(world_size), E0_data(world_size);
  // L[1] phase buffers (combine, after AllToAll)
  std::vector<float*> L1_data(world_size), W1_data(world_size);
  std::vector<int*> T1_data(world_size), E1_data(world_size);
  // Token->slot mapping for O(T×K) combine
  std::vector<int*> d_token_to_slot_map(world_size);

  size_t logits_bytes = T * N_ROUTED_EXPERTS * sizeof(float);
  size_t top_bytes_i = T * K_TOP * sizeof(int);
  size_t top_bytes_f = T * K_TOP * sizeof(float);
  size_t L_bytes = (size_t)world_size * max_tokens_per_gpu * HIDDEN_SIZE * sizeof(float);
  size_t slot_bytes_f = (size_t)world_size * max_tokens_per_gpu * sizeof(float);
  size_t slot_bytes_i = (size_t)world_size * max_tokens_per_gpu * sizeof(int);
  size_t map_bytes = T * K_TOP * sizeof(int);

  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMalloc(&d_X[r], T * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits[r], logits_bytes));
    CUDA_CHECK(cudaMalloc(&d_top_idx[r], top_bytes_i));
    CUDA_CHECK(cudaMalloc(&d_top_w[r], top_bytes_f));
    CUDA_CHECK(cudaMalloc(&d_shared_out[r], T * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output[r], T * HIDDEN_SIZE * sizeof(float)));
    // L[0] phase (dispatch)
    CUDA_CHECK(cudaMalloc(&L0_data[r], L_bytes));
    CUDA_CHECK(cudaMalloc(&W0_data[r], slot_bytes_f));
    CUDA_CHECK(cudaMalloc(&T0_data[r], slot_bytes_i));
    CUDA_CHECK(cudaMalloc(&E0_data[r], slot_bytes_i));
    // L[1] phase (combine, after AllToAll)
    CUDA_CHECK(cudaMalloc(&L1_data[r], L_bytes));
    CUDA_CHECK(cudaMalloc(&W1_data[r], slot_bytes_f));
    CUDA_CHECK(cudaMalloc(&T1_data[r], slot_bytes_i));
    CUDA_CHECK(cudaMalloc(&E1_data[r], slot_bytes_i));
    // Token mapping
    CUDA_CHECK(cudaMalloc(&d_token_to_slot_map[r], map_bytes));
    CUDA_CHECK(cudaMalloc(&d_dispatch_counts[r], world_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recv_counts[r], world_size * sizeof(int)));
  }

  // Copy weights to all GPUs
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMemcpyToSymbol(d_W_ROUTER, W_ROUTER, sizeof(W_ROUTER)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_W_GATE_PROJ, W_GATE_PROJ, sizeof(W_GATE_PROJ)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_W_UP_PROJ, W_UP_PROJ, sizeof(W_UP_PROJ)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_W_DOWN_PROJ, W_DOWN_PROJ, sizeof(W_DOWN_PROJ)));
    CUDA_CHECK(cudaMemcpy(d_X[r], TEST_INPUT, T * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Launch router + topk + dispatch (L[0] phase)
  printf("\n[Phase 1] Router + TopK + Dispatch (L[0] phase)...\n");
  fflush(stdout);
  int block = 128;
  int grid_tokens = (T + block - 1) / block;

  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMemsetAsync(L0_data[r], 0, L_bytes, s_comp[r]));
    CUDA_CHECK(cudaMemsetAsync(W0_data[r], 0, slot_bytes_f, s_comp[r]));
    CUDA_CHECK(cudaMemsetAsync(T0_data[r], -1, slot_bytes_i, s_comp[r]));
    CUDA_CHECK(cudaMemsetAsync(E0_data[r], 0, slot_bytes_i, s_comp[r]));
    CUDA_CHECK(cudaMemsetAsync(d_token_to_slot_map[r], -1, map_bytes, s_comp[r]));
    CUDA_CHECK(cudaMemsetAsync(d_dispatch_counts[r], 0, world_size * sizeof(int), s_comp[r]));
    router_kernel<<<grid_tokens, block, 0, s_comp[r]>>>(d_X[r], d_logits[r], T);
    softmax_topk_kernel<<<grid_tokens, block, 0, s_comp[r]>>>(d_logits[r], d_top_idx[r], d_top_w[r], T);
    dispatch_kernel<<<grid_tokens, block, 0, s_comp[r]>>>(
        d_X[r], d_top_idx[r], d_top_w[r],
        L0_data[r], W0_data[r], T0_data[r], E0_data[r],
        d_token_to_slot_map[r], d_dispatch_counts[r], T, world_size, experts_per_gpu, max_tokens_per_gpu);
    // Shared expert can run in parallel (overlaps with dispatch)
    shared_expert_kernel<<<grid_tokens, block, 0, s_comp[r]>>>(d_X[r], d_shared_out[r], T);
    // Record event when dispatch completes (for AllToAll dependency)
    CUDA_CHECK(cudaEventRecord(dispatch_done[r], s_comp[r]));
  }
  
  // Sync to get dispatch counts for debugging
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaStreamSynchronize(s_comp[r]));
  }
  
  // Debug: Print dispatch information from GPU 0
  if (world_size > 0) {
    CUDA_CHECK(cudaSetDevice(0));
    std::vector<int> h_dispatch_counts(world_size);
    std::vector<int> h_top_idx(T * K_TOP);
    std::vector<float> h_top_w(T * K_TOP);
    CUDA_CHECK(cudaMemcpy(h_dispatch_counts.data(), d_dispatch_counts[0], world_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_top_idx.data(), d_top_idx[0], T * K_TOP * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_top_w.data(), d_top_w[0], T * K_TOP * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("✅ Router + Dispatch kernels launched\n");
    printf("\n[DEBUG] Dispatch counts from GPU 0:\n");
    for (int p = 0; p < world_size; ++p) {
      printf("  -> GPU %d: %d tokens\n", p, h_dispatch_counts[p]);
    }
    printf("[DEBUG] Token routing from GPU 0 (first 3 tokens):\n");
    for (int t = 0; t < 3 && t < T; ++t) {
      printf("  Token %d: ", t);
      for (int k = 0; k < K_TOP; ++k) {
        int eid = h_top_idx[t * K_TOP + k];
        int tgt = eid / experts_per_gpu;
        float w = h_top_w[t * K_TOP + k];
        printf("(expert=%d->GPU%d, w=%.4f) ", eid, tgt, w);
      }
      printf("\n");
    }
    
    // Debug: Check what was actually written to L0_data (sample first few entries)
    std::vector<float> h_L0_check(world_size * max_tokens_per_gpu * HIDDEN_SIZE);
    std::vector<int> h_T0_check(world_size * max_tokens_per_gpu);
    std::vector<int> h_E0_check(world_size * max_tokens_per_gpu);
    CUDA_CHECK(cudaMemcpy(h_L0_check.data(), L0_data[0], world_size * max_tokens_per_gpu * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_T0_check.data(), T0_data[0], world_size * max_tokens_per_gpu * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_E0_check.data(), E0_data[0], world_size * max_tokens_per_gpu * sizeof(int), cudaMemcpyDeviceToHost));
    printf("[DEBUG] Sample L0_data on GPU 0 after dispatch (first 3 slots for GPU 0):\n");
    for (int slot = 0; slot < 3 && slot < h_dispatch_counts[0]; ++slot) {
      int idx = 0 * max_tokens_per_gpu + slot;
      printf("  Slot %d: token=%d, expert=%d, data=[%.6f, %.6f]\n",
             slot, h_T0_check[idx], h_E0_check[idx],
             h_L0_check[idx * HIDDEN_SIZE], h_L0_check[idx * HIDDEN_SIZE + 1]);
    }
    fflush(stdout);
  }

  // REAL OVERLAP: AllToAll starts on s_comm (async, doesn't wait for dispatch)
  // But we need counts first, so wait for dispatch to finish counting
  printf("\n[Phase 2] Waiting for dispatch to complete, then starting AllToAll...\n");
  fflush(stdout);
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaStreamWaitEvent(s_comm[r], dispatch_done[r], 0));
  }

  // counts AllGather (on comm stream)
  printf("Starting AllGather for dispatch counts...\n");
  fflush(stdout);
  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < world_size; ++r) {
    NCCL_CHECK(ncclAllGather((const void*)d_dispatch_counts[r], (void*)d_recv_counts[r],
                             world_size, ncclInt, comms[r], s_comm[r]));
  }
  NCCL_CHECK(ncclGroupEnd());
  printf("✅ AllGather complete\n");
  fflush(stdout);

  // AllToAll: L[0] -> L[1] (on comm stream, async)
  // CRITICAL: All GPUs must call ncclGroupStart/End together (collective operation)
  printf("\n[Phase 3] Starting AllToAll: L[0] -> L[1] (this may take a few seconds)...\n");
  fflush(stdout);
  
  // Set devices for all GPUs first
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
  }
  
  printf("Calling ncclGroupStart (all GPUs together)...\n");
  fflush(stdout);
  // All GPUs call GroupStart together (collective)
  NCCL_CHECK(ncclGroupStart());
  printf("ncclGroupStart returned, setting up Send/Recv operations...\n");
  fflush(stdout);
  
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    for (int peer = 0; peer < world_size; ++peer) {
      size_t sz_f = max_tokens_per_gpu * HIDDEN_SIZE;
      size_t sz_s = max_tokens_per_gpu;
      // L[0] -> L[1] via AllToAll
      NCCL_CHECK(ncclSend(L0_data[r] + peer * max_tokens_per_gpu * HIDDEN_SIZE, sz_f, ncclFloat, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclRecv(L1_data[r] + peer * max_tokens_per_gpu * HIDDEN_SIZE, sz_f, ncclFloat, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclSend(W0_data[r] + peer * max_tokens_per_gpu, sz_s, ncclFloat, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclRecv(W1_data[r] + peer * max_tokens_per_gpu, sz_s, ncclFloat, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclSend(T0_data[r] + peer * max_tokens_per_gpu, sz_s, ncclInt, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclRecv(T1_data[r] + peer * max_tokens_per_gpu, sz_s, ncclInt, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclSend(E0_data[r] + peer * max_tokens_per_gpu, sz_s, ncclInt, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclRecv(E1_data[r] + peer * max_tokens_per_gpu, sz_s, ncclInt, peer, comms[r], s_comm[r]));
    }
  }
  
  printf("All Send/Recv operations queued, calling ncclGroupEnd...\n");
  fflush(stdout);
  // All GPUs call GroupEnd together (collective)
  NCCL_CHECK(ncclGroupEnd());
  printf("ncclGroupEnd returned\n");
  fflush(stdout);
  
  // Record events when AllToAll completes
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaEventRecord(comm_done[r], s_comm[r]));
  }
  printf("✅ AllToAll operations launched (async, running in background)\n");
  fflush(stdout);

  // REAL OVERLAP: expert_apply runs on s_comp in parallel with AllToAll
  // But it needs L[1] data, so wait for comm_done event
  printf("\n[Phase 4] Waiting for AllToAll to complete, then applying experts...\n");
  fflush(stdout);
  
  // Debug: Check received data after first AllToAll (from GPU 0)
  if (world_size > 0) {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamSynchronize(s_comm[0]));
    std::vector<int> h_recv_counts(world_size);
    CUDA_CHECK(cudaMemcpy(h_recv_counts.data(), d_recv_counts[0], world_size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("[DEBUG] Receive counts on GPU 0 after first AllToAll:\n");
    for (int p = 0; p < world_size; ++p) {
      printf("  From GPU %d: %d tokens\n", p, h_recv_counts[p]);
    }
    
    // Sample a few entries from L1_data
    std::vector<float> h_L1_sample(world_size * max_tokens_per_gpu * HIDDEN_SIZE);
    std::vector<int> h_T1_sample(world_size * max_tokens_per_gpu);
    CUDA_CHECK(cudaMemcpy(h_L1_sample.data(), L1_data[0], world_size * max_tokens_per_gpu * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_T1_sample.data(), T1_data[0], world_size * max_tokens_per_gpu * sizeof(int), cudaMemcpyDeviceToHost));
    printf("[DEBUG] Sample data received on GPU 0:\n");
    for (int p = 0; p < world_size && p < 2; ++p) {
      int count = h_recv_counts[p];
      if (count > 0) {
        printf("  From GPU %d (first entry): token=%d, data=[%.6f, %.6f]\n", 
               p, h_T1_sample[p * max_tokens_per_gpu], 
               h_L1_sample[p * max_tokens_per_gpu * HIDDEN_SIZE],
               h_L1_sample[p * max_tokens_per_gpu * HIDDEN_SIZE + 1]);
      }
    }
    fflush(stdout);
  }
  
  int threads_slot = 128;
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    // Wait for AllToAll to complete before processing L[1]
    CUDA_CHECK(cudaStreamWaitEvent(s_comp[r], comm_done[r], 0));
    expert_apply_kernel<<<world_size, threads_slot, 0, s_comp[r]>>>(
        L1_data[r], W1_data[r], T1_data[r], E1_data[r], d_recv_counts[r],
        max_tokens_per_gpu, r, experts_per_gpu);
  }
  printf("✅ Expert computation kernels launched\n");
  fflush(stdout);

  // Wait for expert computation to complete before second AllToAll
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaStreamSynchronize(s_comp[r]));
  }

  // Second AllToAll: Send expert outputs back to original GPUs (L[1] -> L[0])
  // After expert apply, results are in L1_data[r] organized by source peer
  // We need to send them back so each GPU can combine its own tokens
  printf("\n[Phase 5] Second AllToAll: Sending expert outputs back to original GPUs...\n");
  fflush(stdout);
  
  // Clear L0_data to prepare for receiving expert outputs
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMemsetAsync(L0_data[r], 0, L_bytes, s_comm[r]));
  }
  
  // Record event when expert computation completes
  cudaEvent_t expert_done[4];
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaEventCreate(&expert_done[r]));
    CUDA_CHECK(cudaEventRecord(expert_done[r], s_comp[r]));
    CUDA_CHECK(cudaStreamWaitEvent(s_comm[r], expert_done[r], 0));
  }

  // Second AllToAll: L[1] -> L[0] (reverse direction)
  // Key insight: After first AllToAll, L1_data[r][peer] contains data that GPU r received from peer
  // After expert apply, L1_data[r][peer] contains processed data from peer
  // In second AllToAll, GPU r sends L1_data[r][peer] back to peer
  // GPU r receives from peer into L0_data[r][peer] (this is the processed result of data GPU r sent to peer)
  printf("Calling ncclGroupStart for second AllToAll...\n");
  fflush(stdout);
  NCCL_CHECK(ncclGroupStart());
  printf("ncclGroupStart returned, setting up Send/Recv operations...\n");
  fflush(stdout);
  
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    for (int peer = 0; peer < world_size; ++peer) {
      size_t sz_f = max_tokens_per_gpu * HIDDEN_SIZE;
      size_t sz_s = max_tokens_per_gpu;
      // GPU r sends L1_data[r] (data received from peer, now processed) back to peer
      // GPU r receives from peer into L0_data[r] (processed result of data GPU r sent to peer)
      NCCL_CHECK(ncclSend(L1_data[r] + peer * max_tokens_per_gpu * HIDDEN_SIZE, sz_f, ncclFloat, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclRecv(L0_data[r] + peer * max_tokens_per_gpu * HIDDEN_SIZE, sz_f, ncclFloat, peer, comms[r], s_comm[r]));
      // Also transfer T_data so combine_kernel can find token IDs
      NCCL_CHECK(ncclSend(T1_data[r] + peer * max_tokens_per_gpu, sz_s, ncclInt, peer, comms[r], s_comm[r]));
      NCCL_CHECK(ncclRecv(T0_data[r] + peer * max_tokens_per_gpu, sz_s, ncclInt, peer, comms[r], s_comm[r]));
    }
  }
  
  printf("All Send/Recv operations queued, calling ncclGroupEnd...\n");
  fflush(stdout);
  NCCL_CHECK(ncclGroupEnd());
  printf("ncclGroupEnd returned\n");
  fflush(stdout);
  
  // Record event when second AllToAll completes
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaEventRecord(comm_done[r], s_comm[r]));
  }
  printf("✅ Second AllToAll operations launched (async, running in background)\n");
  fflush(stdout);

  // Wait for second AllToAll to complete before combine
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaStreamWaitEvent(s_comp[r], comm_done[r], 0));
  }

  // Debug: Check data after second AllToAll (from GPU 0)
  if (world_size > 0) {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamSynchronize(s_comm[0]));
    std::vector<int> h_dispatch_counts_check(world_size);
    CUDA_CHECK(cudaMemcpy(h_dispatch_counts_check.data(), d_dispatch_counts[0], world_size * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<float> h_L0_sample(world_size * max_tokens_per_gpu * HIDDEN_SIZE);
    std::vector<int> h_T0_sample(world_size * max_tokens_per_gpu);
    CUDA_CHECK(cudaMemcpy(h_L0_sample.data(), L0_data[0], world_size * max_tokens_per_gpu * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_T0_sample.data(), T0_data[0], world_size * max_tokens_per_gpu * sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("\n[DEBUG] Data on GPU 0 after second AllToAll (before combine):\n");
    for (int p = 0; p < world_size; ++p) {
      int count = h_dispatch_counts_check[p];
      printf("  To GPU %d: %d tokens\n", p, count);
      if (count > 0 && count <= 3) {
        for (int s = 0; s < count; ++s) {
          int idx = p * max_tokens_per_gpu + s;
          printf("    Slot %d: token=%d, data=[%.6f, %.6f]\n", 
                 s, h_T0_sample[idx],
                 h_L0_sample[idx * HIDDEN_SIZE],
                 h_L0_sample[idx * HIDDEN_SIZE + 1]);
        }
      }
    }
    fflush(stdout);
  }

  // Combine: O(T×K) using T_data to find token locations
  // Now L0_data contains the expert outputs sent back from other GPUs
  printf("\n[Phase 6] Combining outputs (O(T×K) combine)...\n");
  
  // Debug: Manual verification for token 0 on GPU 0 using token_to_slot_map
  if (world_size > 0 && T > 0) {
    CUDA_CHECK(cudaSetDevice(0));
    std::vector<int> h_token_map(T * K_TOP);
    std::vector<float> h_L0_debug(world_size * max_tokens_per_gpu * HIDDEN_SIZE);
    std::vector<float> h_shared_debug(T * HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(h_token_map.data(), d_token_to_slot_map[0], T * K_TOP * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L0_debug.data(), L0_data[0], world_size * max_tokens_per_gpu * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_shared_debug.data(), d_shared_out[0], T * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("[DEBUG] Manual combine verification for token 0 using token_to_slot_map:\n");
    float manual_acc0 = h_shared_debug[0];
    float manual_acc1 = h_shared_debug[1];
    printf("  Shared expert: [%.6f, %.6f]\n", manual_acc0, manual_acc1);
    int map_base = 0 * K_TOP;
    for (int k = 0; k < K_TOP; ++k) {
      int map_val = h_token_map[map_base + k];
      if (map_val < 0) {
        printf("  Entry %d: invalid (map_val=%d)\n", k, map_val);
        continue;
      }
      int peer = map_val >> 16;
      int slot = map_val & 0xFFFF;
      size_t base = ((size_t)peer * max_tokens_per_gpu + slot) * HIDDEN_SIZE;
      printf("  Entry %d: peer=%d, slot=%d, data=[%.6f, %.6f]\n",
             k, peer, slot, h_L0_debug[base], h_L0_debug[base + 1]);
      manual_acc0 += h_L0_debug[base];
      manual_acc1 += h_L0_debug[base + 1];
    }
    printf("  Manual combine result: [%.6f, %.6f]\n", manual_acc0, manual_acc1);
    fflush(stdout);
  }
  
  fflush(stdout);
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    combine_kernel<<<grid_tokens, block, 0, s_comp[r]>>>(
        d_shared_out[r], L0_data[r], d_token_to_slot_map[r],
        d_output[r], T, max_tokens_per_gpu);
  }
  printf("✅ Combine kernel launched\n");
  fflush(stdout);

  printf("\n[Phase 6] Synchronizing all streams...\n");
  fflush(stdout);
  for (int r = 0; r < world_size; ++r) CUDA_CHECK(cudaStreamSynchronize(s_comp[r]));
  printf("✅ All computation complete\n");
  fflush(stdout);

  // Debug: Check shared expert output and final output
  if (world_size > 0) {
    CUDA_CHECK(cudaSetDevice(0));
    std::vector<float> h_shared(T * HIDDEN_SIZE);
    std::vector<float> h_out(T * HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(h_shared.data(), d_shared_out[0], T * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_output[0], T * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\n[DEBUG] Shared expert outputs (first 3 tokens):\n");
    for (int t = 0; t < 3 && t < T; ++t) {
      printf("  Token %d: [%.6f, %.6f]\n", t, 
             h_shared[t * HIDDEN_SIZE], h_shared[t * HIDDEN_SIZE + 1]);
    }
    printf("[DEBUG] Final outputs (first 3 tokens):\n");
    for (int t = 0; t < 3 && t < T; ++t) {
      printf("  Token %d: [%.6f, %.6f]\n", t, 
             h_out[t * HIDDEN_SIZE], h_out[t * HIDDEN_SIZE + 1]);
    }
    fflush(stdout);
  }

  // Pull result from GPU0
  std::vector<float> h_out(T * HIDDEN_SIZE);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_output[0], T * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  printf("FlashMoE symmetric layout demo done. First token output: [%.6f, %.6f]\n", h_out[0], h_out[1]);
  
  // ============================================================================
  // Numerical Verification: Compare GPU output with CPU reference
  // ============================================================================
  printf("\n============================================================\n");
  printf("Numerical Verification\n");
  printf("============================================================\n");
  
  // Run CPU reference
  std::vector<float> h_ref(T * HIDDEN_SIZE);
  moe_forward_cpu(TEST_INPUT, h_ref.data(), T);
  
  // Compute errors
  float max_err = 0.0f;
  float sum_err = 0.0f;
  int error_count = 0;
  for (int i = 0; i < T * HIDDEN_SIZE; ++i) {
    float err = std::fabs(h_out[i] - h_ref[i]);
    if (err > max_err) max_err = err;
    sum_err += err;
    if (err > 1e-5f) error_count++;
  }
  float avg_err = sum_err / (T * HIDDEN_SIZE);
  
  printf("Max absolute error:  %.9e\n", max_err);
  printf("Average error:      %.9e\n", avg_err);
  printf("Elements with error > 1e-5: %d / %d\n", error_count, T * HIDDEN_SIZE);
  printf("\n");
  
  // Print first few tokens comparison
  printf("First 3 tokens comparison:\n");
  for (int t = 0; t < 3 && t < T; ++t) {
    printf("  Token %d:\n", t);
    printf("    GPU:  [%12.8f, %12.8f]\n",
           h_out[t * HIDDEN_SIZE + 0], h_out[t * HIDDEN_SIZE + 1]);
    printf("    CPU:  [%12.8f, %12.8f]\n",
           h_ref[t * HIDDEN_SIZE + 0], h_ref[t * HIDDEN_SIZE + 1]);
    float err0 = std::fabs(h_out[t * HIDDEN_SIZE + 0] - h_ref[t * HIDDEN_SIZE + 0]);
    float err1 = std::fabs(h_out[t * HIDDEN_SIZE + 1] - h_ref[t * HIDDEN_SIZE + 1]);
    printf("    Diff: [%12.6e, %12.6e]\n", err0, err1);
  }
  printf("\n");
  
  // Final verdict
  if (max_err < 1e-5f) {
    printf("✅✅✅ VERIFICATION PASSED! ✅✅✅\n");
    printf("GPU output matches CPU reference within tolerance (1e-5)\n");
    printf("FlashMoE forward is numerically correct.\n");
  } else if (max_err < 1e-3f) {
    printf("⚠️  ACCEPTABLE: Error within loose tolerance (1e-3)\n");
    printf("May indicate minor numerical differences in dispatch/combine.\n");
  } else {
    printf("❌ VERIFICATION FAILED: Error too large\n");
    printf("Check dispatch, AllToAll, or combine logic.\n");
  }
  printf("============================================================\n\n");

  // Cleanup
  for (int r = 0; r < world_size; ++r) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaFree(d_X[r]));
    CUDA_CHECK(cudaFree(d_logits[r]));
    CUDA_CHECK(cudaFree(d_top_idx[r]));
    CUDA_CHECK(cudaFree(d_top_w[r]));
    CUDA_CHECK(cudaFree(d_shared_out[r]));
    CUDA_CHECK(cudaFree(d_output[r]));
    CUDA_CHECK(cudaFree(L0_data[r]));
    CUDA_CHECK(cudaFree(L1_data[r]));
    CUDA_CHECK(cudaFree(W0_data[r]));
    CUDA_CHECK(cudaFree(W1_data[r]));
    CUDA_CHECK(cudaFree(T0_data[r]));
    CUDA_CHECK(cudaFree(T1_data[r]));
    CUDA_CHECK(cudaFree(E0_data[r]));
    CUDA_CHECK(cudaFree(E1_data[r]));
    CUDA_CHECK(cudaFree(d_token_to_slot_map[r]));
    CUDA_CHECK(cudaFree(d_dispatch_counts[r]));
    CUDA_CHECK(cudaFree(d_recv_counts[r]));
    CUDA_CHECK(cudaEventDestroy(dispatch_done[r]));
    CUDA_CHECK(cudaEventDestroy(comm_done[r]));
    CUDA_CHECK(cudaStreamDestroy(s_comp[r]));
    CUDA_CHECK(cudaStreamDestroy(s_comm[r]));
    NCCL_CHECK(ncclCommDestroy(comms[r]));
  }
  return 0;
}
