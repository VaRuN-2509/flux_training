import modal
from pathlib import Path

# ------------------------------------------------------------
# 1) Dependencies (installed ONCE at image-build time)
# ------------------------------------------------------------
all_deps = [
    # Core
    "accelerate",
    "peft",
    "torchmetrics",
    "lpips",
    "git+https://github.com/huggingface/diffusers.git",
    "einops",
    "fire>=0.6.0",
    "huggingface-hub",
    "safetensors",
    "sentencepiece",
    "transformers>=4.45.2",
    "tokenizers",
    "protobuf",
    "requests",
    "invisible-watermark",
    "ruff==0.6.8",
    "bitsandbytes",

    # Torch
    "torch==2.6.0",
    "torchvision",

    # Torch + transformers require numpy<2
    "numpy<2",
]

# ------------------------------------------------------------
# 2) BUILD THE IMAGE (cached forever)
# ------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install([
        "git",
        "gcc",
        "g++",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6"
    ])
    # Install deps ONCE and cache
    .pip_install(all_deps)
    # Copy your *entire repo* into the image so training never reloads
    .add_local_dir(".", "/root/ft")      # <-- KEY MODIFICATION
)

# ------------------------------------------------------------
# 3) VOLUME for datasets, checkpoints, outputs
# ------------------------------------------------------------
app = modal.App(name="flux_inference",image=image)
volume = modal.Volume.from_name("flux-project", create_if_missing=True)

CACHE_DIR = Path("/cache") 
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True) 
volumes = {CACHE_DIR: cache_volume}

vol = modal.Volume.from_name("dataset",create_if_missing=True)
# 4) TRAIN FUNCTION (no installs, only computation)
# ------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    min_containers=1,        # Keep container warm, no re-init
    volumes={
        "/root/data": vol,
    },
    env = {
        "HF_HOME": "/cache",
        "HF_HUB_CACHE": "/cache",
    },
    timeout=60*60*24,
)

def run_training(script_name="inference.py"):
    import subprocess
    import sys
    import os

    # your baked-in repo lives here
    code_dir = "/root/ft"

    # set PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{code_dir}:{code_dir}/src"

    # run training script
    subprocess.run(
        [sys.executable, script_name],
        cwd=code_dir,
        env=env,
        check=True,
    )



# ------------------------------------------------------------
# 5) Setup local volume
# ------------------------------------------------------------
@app.local_entrypoint()
def create_volume():
    print("Volume 'flux-project' is ready.")


