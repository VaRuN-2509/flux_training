import modal
import sys
import subprocess
import os

GPU_TYPE = "A100"
PROJECT_FOLDER = "."  # current folder

# Build the Modal image
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install system packages OpenCV needs
    .apt_install(["libgl1", "libglib2.0-0", "libsm6", "libxrender1", "libxext6"])
    
    # Upload your repo
    .add_local_dir(PROJECT_FOLDER, remote_path="/root/flux_training", copy=True)
    
    # Install Python dependencies (editable install with extras)
    .run_commands(
        "cd /root/flux_training && pip install -e .[all]"
    )
)


app = modal.App(name="run-flux-module")

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
)
def run_module():
    print(f"--- Running 'python -m flux ...' in {PROJECT_FOLDER} on {GPU_TYPE} GPU ---")
    
    cwd = "/root/flux_training"
    env = os.environ.copy()
    env["PYTHONPATH"] = cwd  # Ensure Python finds your flux package

    command = [
        sys.executable,
        "-m",
        "flux",
        "kontext",
        "--track_usage",
        "--prompt",
        "replace the logo with the te"
    ]

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
    print(f"--- Module finished with exit code {process.returncode} ---")
    if process.returncode != 0:
        raise SystemExit(f"Module exited with code {process.returncode}")

@app.local_entrypoint()
def main():
    print(f"Deploying and running flux module from '{PROJECT_FOLDER}' on Modal...")
    run_module.remote()
