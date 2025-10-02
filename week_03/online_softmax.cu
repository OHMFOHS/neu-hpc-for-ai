// online_softmax.cu  (fixed with validity-aware reduction)
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

// Combine two "valid" segments (m,d). Caller must ensure both are valid
__device__ __forceinline__
void md_combine_both_valid(float ma, float da, float mb, float db, float &m, float &d) {
    float m_new = fmaxf(ma, mb);
    // Note: both sides are valid here, so -inf - -inf case won't occur
    float d_new = da * __expf(ma - m_new) + db * __expf(mb - m_new);
    m = m_new; d = d_new;
}

// One block per row; column-wise parallel + segmented; shared memory reduction with validity bits
template<int BLOCK_SIZE>
__global__ void softmax_online_rows_cuda(const float* __restrict__ X,
                                         float* __restrict__ Y,
                                         int rows, int cols)
{
    int r = blockIdx.x; // row r
    if (r >= rows) return;

    const float* x = X + (size_t)r * cols;
    float*       y = Y + (size_t)r * cols;

    // Local online scan per thread (across stride columns)
    float m_loc = -INFINITY, d_loc = 0.0f;
    int   v_loc = 0; // validity bit: whether this thread has processed at least one element
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) {
        v_loc = 1;
        float m_old = m_loc;
        float xv    = x[c];
        m_loc = fmaxf(m_loc, xv);
        d_loc = d_loc * __expf(m_old - m_loc) + __expf(xv - m_loc);
    }

    // Shared memory: m, d, valid
    __shared__ float s_m[BLOCK_SIZE];
    __shared__ float s_d[BLOCK_SIZE];
    __shared__ int   s_v[BLOCK_SIZE];
    s_m[threadIdx.x] = m_loc;
    s_d[threadIdx.x] = d_loc;
    s_v[threadIdx.x] = v_loc;
    __syncthreads();

    // Reduce to row-wise (m,d) with validity-bit safe merging
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            int va = s_v[threadIdx.x];
            int vb = s_v[threadIdx.x + offset];

            if (va == 0 && vb == 0) {
                // Both invalid -> still invalid, keep (-inf, 0)
                // s_m/d/v already in this state, no change needed
            } else if (va == 0 && vb == 1) {
                // Only right side valid -> directly copy right side
                s_m[threadIdx.x] = s_m[threadIdx.x + offset];
                s_d[threadIdx.x] = s_d[threadIdx.x + offset];
                s_v[threadIdx.x] = 1;
            } else if (va == 1 && vb == 0) {
                // Only left side valid -> keep left side, unchanged
                // (do nothing)
            } else { // va==1 && vb==1
                float ma = s_m[threadIdx.x];
                float da = s_d[threadIdx.x];
                float mb = s_m[threadIdx.x + offset];
                float db = s_d[threadIdx.x + offset];
                float m_new, d_new;
                md_combine_both_valid(ma, da, mb, db, m_new, d_new);
                s_m[threadIdx.x] = m_new;
                s_d[threadIdx.x] = d_new;
                s_v[threadIdx.x] = 1;
            }
        }
        __syncthreads();
    }

    // Get row-wise (m,d), requires at least one element in this row (cols>0)
    float m = s_m[0];
    float d = s_d[0];
    int   v = s_v[0];

    // Write back: if this row is invalid (extreme cols==0), return directly; won't happen in normal matrices
    if (v == 0) return;

    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) {
        y[c] = __expf(x[c] - m) / d;
    }
}

// Launch wrapper
void softmax_online_rows_cuda_launch(const float* dX, float* dY,
                                     int rows, int cols, int block_size=256) {
    dim3 grid(rows);
    switch (block_size) {
        case 128: softmax_online_rows_cuda<128><<<grid, 128>>>(dX, dY, rows, cols); break;
        case 256: softmax_online_rows_cuda<256><<<grid, 256>>>(dX, dY, rows, cols); break;
        case 512: softmax_online_rows_cuda<512><<<grid, 512>>>(dX, dY, rows, cols); break;
        default : softmax_online_rows_cuda<256><<<grid, 256>>>(dX, dY, rows, cols); break;
    }
    CHECK_CUDA(cudaPeekAtLastError());
}

// ---------------- Test main ----------------
int main() {
    const int rows = 2, cols = 4;
    float hX[rows * cols] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 1.0f, 0.0f, -1.0f
    };
    float hY[rows * cols];

    float *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, sizeof(hX)));
    CHECK_CUDA(cudaMalloc(&dY, sizeof(hY)));
    CHECK_CUDA(cudaMemcpy(dX, hX, sizeof(hX), cudaMemcpyHostToDevice));

    // Can also use smaller blocks, e.g., 128; key is when >= col count there will be many invalid threads, this version handles correctly
    softmax_online_rows_cuda_launch(dX, dY, rows, cols, 256);

    CHECK_CUDA(cudaMemcpy(hY, dY, sizeof(hY), cudaMemcpyDeviceToHost));

    printf("Input:\n");
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) printf(" %6.3f", hX[r * cols + c]);
        printf("\n");
    }
    printf("Softmax:\n");
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) printf(" %6.3f", hY[r * cols + c]);
        printf("\n");
    }

    cudaFree(dX);
    cudaFree(dY);
    return 0;
}
