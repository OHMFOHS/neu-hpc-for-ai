# scripts/modal_moe_nccl.py ✅ 最终正确版
import modal
import subprocess
import os

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .env({
        "NCCL_DEBUG": "INFO",  # Enable NCCL debug output
        "NCCL_DEBUG_SUBSYS": "ALL",  # Show all NCCL subsystems
        "NCCL_P2P_DISABLE": "1",  # Disable P2P to avoid deadlock issues
        "NCCL_IB_DISABLE": "1",  # Disable InfiniBand (not available in Modal)
    })
    .add_local_dir("week_08", remote_path="/root/week_08")
)

app = modal.App("moe-nccl-final")

@app.function(image=image, gpu="A100-40GB:4", timeout=600)
def run_moe_nccl():
    print("Working dir:", os.getcwd())

    cu_path = "/root/week_08/CUDA/deepseek_v3_moe_complete.cu"
    print("Compiling:", cu_path)

    subprocess.run(
        ["nvcc", "-O2", cu_path, "-o", "moe_nccl.bin", "-lnccl"],
        check=True,
        text=True,
    )

    print("Running NCCL MoE...")
    subprocess.run(["./moe_nccl.bin"], check=True, text=True)

if __name__ == "__main__":
    run_moe_nccl.remote()
