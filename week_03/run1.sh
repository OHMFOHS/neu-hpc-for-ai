#!/bin/bash
mkdir -p bin
rm -f bin/main.cu
cp gemm.cu bin/gemm.cu
cd ..
modal run scripts/modal_nvcc.py --code-path week_03/bin/gemm.cu