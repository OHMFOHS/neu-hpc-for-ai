GEMM kernel:
(modal-env) programmaster@MacBookPro week_02 % ./run.sh
✓ Initialized. View run at https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-MhQmZfdawIrQHJFTBU4gxx
✓ Created objects.
├── 🔨 Created mount /Users/programmaster/INFO7375/neu-hpc-for-ai/scripts/modal_nvcc.py
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

GEMM: D = alpha*A*B + beta*C
M=512 N=512 K=512 alpha=1.0000 beta=1.0000 kernel=tiled tile=16 reps=50
Avg time: 1.925898 ms | Throughput: 139.65 GFLOP/s | Max |GPU-CPU|: 0.000031

llama2.java:
javac llama2.java
java week_02.llama2 week_02/stories15M.bin