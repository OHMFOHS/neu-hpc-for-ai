# FlashMoE Implementation

This directory contains the implementation of FlashMoE, a fused Mixture of Experts (MoE) kernel with overlapped computation and communication.

## Features

- **Symmetric Tensor Layout**: Efficient all-to-all communication with balanced token distribution across GPUs
- **Non-blocking Communication**: Overlaps expert computation with all-to-all communication operations
- **Fused Kernel Approach**: Single kernel that combines expert computation and communication

## Assignment Requirements

### Required
- ✅ Symmetric tensor layout (Section 3.2, excluding 3.2.1)

### Optional
- Task abstraction and task queue (Section 3.0, 3.1, Appendix F) - *Not implemented in this version*

## Architecture

The FlashMoE implementation consists of:

1. **Router**: Top-k expert selection based on gate logits
2. **Shared Experts**: Always-applied MLP blocks
3. **Routed Experts**: Expert-parallel MLP blocks with symmetric tensor layout
4. **All-to-All Communication**: Non-blocking communication for token dispatch and combine

### Symmetric Tensor Layout

The symmetric tensor layout organizes tokens such that:
- Each GPU sends/receives balanced chunks of tokens
- Tokens are organized by expert assignment
- All-to-all communication is efficient with balanced load

### Overlapped Communication

- Non-blocking NCCL operations (`ncclSend`/`ncclRecv`) allow computation to proceed while communication is in progress
- Expert computation on local experts overlaps with all-to-all communication for remote experts

## Building

```bash
./build.sh
```

Or manually:
```bash
nvcc -O3 -std=c++17 -ccbin mpicc \
     -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
     -lcudart -lnccl -lm \
     -o flash_moe flash_moe.cu
```

## Running

```bash
./run.sh <num_gpus> <batch_size> <seq_len> <hidden_dim> <num_experts> <top_k> [intermediate_dim]
```

Or manually:
```bash
mpirun --allow-run-as-root -np <num_gpus> ./flash_moe <batch_size> <seq_len> <hidden_dim> <num_experts> <top_k> [intermediate_dim]
```

### Example

```bash
./run.sh 2 2 128 512 8 2 2048
```

This runs FlashMoE with:
- 2 GPUs
- Batch size: 2
- Sequence length: 128
- Hidden dimension: 512
- Number of experts: 8
- Top-k: 2
- Intermediate dimension: 2048

## Implementation Details

### Key Components

1. **Router Kernel** (`topk_router_kernel`): Selects top-k experts for each token
2. **MLP Forward** (`deepseek_v3_mlp_forward_complete`): Complete MLP forward pass with gate/up/down projections
3. **FlashMoE Forward** (`flash_moe_forward`): Main function that orchestrates routing, shared experts, routed experts, and all-to-all communication
4. **Symmetric Layout**: Token organization for efficient all-to-all communication

### Communication Pattern

1. **Dispatch Phase**: Tokens are organized by expert assignment and sent to the GPU that owns each expert
2. **Expert Computation**: Each GPU processes its assigned experts
3. **Combine Phase**: Expert outputs are gathered back using all-to-all communication

### Performance Optimizations

- Tiled GEMM operations for efficient matrix multiplication
- Non-blocking NCCL operations for communication overlap
- Symmetric tensor layout for balanced all-to-all communication
- Reused temporary buffers to minimize memory allocation

## Differences from Baseline MoE

The baseline MoE implementation (in `week_08/deepseek_moe.cu`) uses:
- Synchronous all-to-all operations
- Sequential expert processing
- No overlap between computation and communication

FlashMoE improves upon this by:
- Non-blocking all-to-all operations
- Overlapped computation and communication
- Symmetric tensor layout for efficient communication

## Notes

- This implementation provides a foundation for FlashMoE with symmetric tensor layout
- Full production implementation would include more sophisticated token reorganization and better overlap mechanisms
- The optional task abstraction and task queue features are not implemented in this version

