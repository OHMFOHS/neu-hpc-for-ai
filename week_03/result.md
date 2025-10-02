# Extend your GEMM CUDA kernel
(modal-env) programmaster@MacBookPro week_03 % ./run.sh
✓ Initialized. View run at https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-Fs1a7he4oBkFyPiO3F3O5G
✓ Created objects.
├── 🔨 Created mount /Users/programmaster/INFO7375/neu-hpc-for-ai/scripts/modal_nvcc.py
├── 🔨 Created mount /Users/programmaster/INFO7375/neu-hpc-for-ai/week_03
├── 🔨 Created mount /Users/programmaster/INFO7375/neu-hpc-for-ai/week_02
├── 🔨 Created mount /Users/programmaster/INFO7375/neu-hpc-for-ai/week_01
└── 🔨 Created function compile_and_run_cuda.

==========
== CUDA ==
==========

CUDA Version 12.8.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

GEMM inplace: C <- alpha*op(A)op(B) + beta*C
M=512 N=512 K=512 alpha=1.0000 beta=1.0000 kernel=tiled tile=16 reps=50 transA=N transB=N
Avg time: 1.936568 ms | Throughput: 138.88 GFLOP/s | Max |GPU-CPU|: 1746.885742

# online_softmax:
    Input:
    1.000  2.000  3.000  4.000
    2.000  1.000  0.000 -1.000
    Softmax:
    0.032  0.087  0.237  0.644
    0.644  0.237  0.087  0.032