#!/bin/bash

# Build script for FlashMoE
# Usage: ./build.sh

set -e

echo "Building FlashMoE..."

nvcc -O3 -std=c++17 -ccbin mpicc \
     -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
     -lcudart -lnccl -lm \
     -o flash_moe flash_moe.cu

echo "Build complete! Binary: ./flash_moe"
echo "Run with: mpirun --allow-run-as-root -np <num_gpus> ./flash_moe <batch_size> <seq_len> <hidden_dim> <num_experts> <top_k> [intermediate_dim]"

