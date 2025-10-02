#!/bin/bash
mkdir -p bin
rm -f bin/main.cu
cp online_softmax.cu bin/main.cu
cd ..
modal run scripts/modal_nvcc.py --code-path week_03/bin/main.cu