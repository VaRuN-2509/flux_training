import modal
import sys
import subprocess
import os

GPU_TYPE = "A100"
PROJECT_FOLDER = "."  # include all files in this folder
SCRIPT_NAME = "training.py"  # the script to execute

# ---- Build the Modal Image ----
image = (
    modal.Image.debian_slim(python_version="3.10")
    # System dependencies (OpenCV, Torch, etc.)
    .apt_install(["libgl1", "libglib2.0-0", "libsm6", "libxrender1", "libxext6"])
    
    # Upload your entire project directory to /root/flux_training
    .add_local_dir(PROJECT_FOLDER, remote_path="/root/flux_training", copy=True)
    
    # Install Python dependencies in editable mode
    .run_commands(
        "cd /root/flux_training && pip install -e .[all]"
    )
)

# ---- Define Modal App ----
app = modal.App(name="run-flux-training")

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,  # seconds (30 min); increase for longer runs
)
def run_training():
    print(f"--- Running '{SCRIPT_NAME}' in {PROJECT_FOLDER} on {GPU_TYPE} GPU ---")

    cwd = "/root/flux_training"
    env = os.environ.copy()
    env["PYTHONPATH"] = cwd  # ensures imports like `from flux import ...` work

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
