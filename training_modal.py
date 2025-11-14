import modal
import sys
import subprocess
import os

GPU_TYPE = "A100-80GB"
PROJECT_FOLDER = "."  # include all files in this folder
SCRIPT_NAME = "training.py"  # the script to execute

all_deps = [
    # --- Core dependencies ---
    "accelerate",
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
    "accelerate",

    # --- Optional: torch ---
    "torch==2.6.0",
    "torchvision",

    # --- Optional: streamlit ---
    "streamlit",
    "streamlit-drawable-canvas",
    "streamlit-keyup",

    # --- Optional: gradio ---
    "gradio",

    # --- Optional: tensorrt ---
    "tensorrt-cu12==10.12.0.36",
    "colored",
    "opencv-python-headless==4.8.0.74",
    "onnx>=1.18.0",
    "onnxruntime~=1.22.0",
    "onnxruntime-gpu~=1.22.0",
    "onnx-graphsurgeon",
    "polygraphy>=0.49.22",

    # --- Meta optional group `all` ---
    "flux[gradio]",
    "flux[streamlit]",
    "flux[torch]",
]

# all_deps = ['accelerate', 'torchvision', 'einops', 'fire >= 0.6.0', 'huggingface-hub', 'safetensors', 'sentencepiece', 'transformers >= 4.45.2', 'tokenizers', 'protobuf', 'requests', 'invisible-watermark', 'ruff == 0.6.8', 'accelerate', 'flux[gradio]', 'flux[streamlit]', 'flux[torch]']
# ---- Build the Modal Image ----
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install([
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6"
    ])
    .add_local_dir(PROJECT_FOLDER, remote_path="/root/flux_training", copy=True)
    .run_commands("cd /root/flux_training")
    .uv_pip_install(all_deps + ["numpy<2"])
)
# ---- Define Modal App ----
app = modal.App(image=image, name="run-flux-training")

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,  # seconds (30 min); increase for longer runs
)
def run_training():
    print(f"--- Running '{SCRIPT_NAME}' in {PROJECT_FOLDER} on {GPU_TYPE} GPU ---")

    cwd = "/root/flux_training"
    env = os.environ.copy()
    # env["PYTHONPATH"] = cwd  # ensures imports like `from flux import ...` work
    env["PYTHONPATH"] = f"{cwd}:{cwd}/src"
    # Command to run the script
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

    # Stream output live to Modal logs
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()

    process.wait()
    print(f"--- Training finished with exit code {process.returncode} ---")

    if process.returncode != 0:
        raise SystemExit(f"Training script exited with code {process.returncode}")

@app.local_entrypoint()
def main():
    print(f"Deploying and running '{SCRIPT_NAME}' from '{PROJECT_FOLDER}' on Modal...")
    run_training.remote()
