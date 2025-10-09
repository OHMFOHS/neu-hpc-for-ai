#!/bin/bash
mkdir -p bin
rm -f bin/main.cu
cp flash_attention2.cu bin/main.cu
cd ..
modal run scripts/modal_nvcc.py --code-path week_05/bin/main.cu