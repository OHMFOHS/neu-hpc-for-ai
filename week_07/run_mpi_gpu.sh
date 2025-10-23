#!/bin/bash

set -Eeuoa pipefail

# Check that there are exactly 2 arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <file_path> <gpu_count>"
  exit 1
fi

# Check that $1 is a path to an existing file
if [ ! -f "$1" ]; then
  echo "Error: '$1' is not a valid file."
  exit 2
fi

# Check that $2 is a number (integer or decimal)
if ! [[ "$2" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: '$2' is not a valid number."
  exit 3
fi

cd ..
modal run scripts/modal_mpi_gpu.py::compile_and_run_cuda_$2 --code-path week_07/$1 