#!/bin/bash

# Run script for FlashMoE
# Usage: ./run.sh <num_gpus> <batch_size> <seq_len> <hidden_dim> <num_experts> <top_k> [intermediate_dim]

set -e

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <num_gpus> <batch_size> <seq_len> <hidden_dim> <num_experts> <top_k> [intermediate_dim]"
    echo "Example: $0 2 2 128 512 8 2 2048"
    exit 1
fi

NUM_GPUS=$1
BATCH_SIZE=$2
SEQ_LEN=$3
HIDDEN_DIM=$4
NUM_EXPERTS=$5
TOP_K=$6
INTERMEDIATE_DIM=${7:-$((HIDDEN_DIM * 4))}

echo "Running FlashMoE with:"
echo "  GPUs: $NUM_GPUS"
echo "  Batch size: $BATCH_SIZE"
echo "  Sequence length: $SEQ_LEN"
echo "  Hidden dimension: $HIDDEN_DIM"
echo "  Intermediate dimension: $INTERMEDIATE_DIM"
echo "  Number of experts: $NUM_EXPERTS"
echo "  Top-k: $TOP_K"
echo ""

mpirun --allow-run-as-root -np $NUM_GPUS ./flash_moe $BATCH_SIZE $SEQ_LEN $HIDDEN_DIM $NUM_EXPERTS $TOP_K $INTERMEDIATE_DIM

