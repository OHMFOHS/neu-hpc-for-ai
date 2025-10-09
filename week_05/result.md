Implement FlashAttention-2 as described in Section 3 of the paper
run: ./run.sh
result:(modal-env) programmaster@maojunchaodeMacBook-Pro week_05 % ./run.sh
✓ Initialized. View run at https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-kQsB8xUIdSYSUwmGvQQsM2
✓ Created objects.
├── 🔨 Created mount /Users/programmaster/INFO7375/neu-hpc-for-ai/scripts/modal_nvcc.py
├── 🔨 Created mount /Users/programmaster/INFO7375/neu-hpc-for-ai/week_05
├── 🔨 Created mount /Users/programmaster/INFO7375/neu-hpc-for-ai/week_04
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


=== Validation Result ===
Row 0  GPU=0.000000  CPU=0.000000  Δ=0.000000e+00
Row 1  GPU=0.061879  CPU=0.061879  Δ=3.725290e-09
Row 2  GPU=0.210442  CPU=0.210442  Δ=0.000000e+00
Row 3  GPU=0.341078  CPU=0.341078  Δ=0.000000e+00
Max abs error: 2.384186e-07
✅ SUCCESS: GPU matches CPU baseline.