#!/bin/bash
mkdir -p bin
rm -f bin/main.cu
cp gemm.cu bin/main.cu
cd ..
modal run scripts/modal_nvcc.py --code-path week_02/bin/main.cu