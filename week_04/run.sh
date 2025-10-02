#!/bin/bash
mkdir -p bin
rm -f bin/main.cu
cp flash_attention.cu bin/main.cu
cd ..
modal run scripts/modal_nvcc.py --code-path week_04/bin/main.cu