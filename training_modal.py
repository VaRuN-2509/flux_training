import modal
import sys
import subprocess
import os

GPU_TYPE = "A100"
PROJECT_FOLDER = "."  # folder with your training script
SCRIPT_NAME = "training.py"  # the script to execute

# ---- Step 1: Build the base image (only rebuilt when dependencies change) ----
# Give this image a unique tag, e.g., "flux-training-env:v1"
# If you change dependencies, bump to "v2" or similar.
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1", "libglib2.0-0", "libsm6", "libxrender1", "libxext6"])
    .pip_install(
        "torch",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate",
        "safetensors",
        "modal",
        "opencv-python",
        "numpy",
        "Pillow",
    )
    # Optional: install your package (without adding project code, for speed)
    .run_commands("pip install -e /root/flux_training[all] || true")
)

# ---- Step 2: Define Modal App ----
app = modal.App(name="flux-train")

@app.function(
    image=base_image,  # reuse the cached image
    gpu=GPU_TYPE,
    timeout=6 * 3600,  # 6 hours
    mounts=[
        modal.Mount.from_local_dir(
            PROJECT_FOLDER,  # your local code directory
            remote_path="/root/flux_training",  # mounted path in container
        )
    ],
)
def run_training():
    print(f"--- Running '{SCRIPT_NAME}' on {GPU_TYPE} ---")

    cwd = "/root/flux_training"
    env = os.environ.copy()
    env["PYTHONPATH"] = cwd

    command = [sys.executable, SCRIPT_NAME]

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()

    process.wait()
    print(f"--- Training finished with exit code {process.returncode} ---")
    if process.returncode != 0:
        raise SystemExit(f"Training script exited with code {process.returncode}")

@app.local_entrypoint()
def main():
    print(f"Deploying '{SCRIPT_NAME}' with cached dependencies...")
    run_training.remote()
