import modal
import os
import subprocess

# Create a persistent volume for model weights AND code
volume = modal.Volume.from_name("multitalk", create_if_missing=True)

VOLUME_PATH = "/data"
REPO_DIR = "MultiTalk"
REPO_PATH = f"{VOLUME_PATH}/{REPO_DIR}"
REPO_URL = "https://github.com/puterhimself/MultiTalk"

# Define the image with all dependencies
# Using PyTorch 2.5.1 which has torch.distributed.tensor.experimental required by xfuser
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "wget", "ninja-build")
    # Install PyTorch 2.5.1 with CUDA 12.4 support
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1", 
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Install xformers compatible with PyTorch 2.5.1
    .pip_install(
        "xformers==0.0.28.post3",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "packaging",
        "ninja",
        "wheel",
        "setuptools",
        "psutil",
    )
    # Install flash-attn (compatible with PyTorch 2.5.x)
    .run_commands(
        "export MAX_JOBS=4 && pip install flash-attn==2.7.4.post1 --no-build-isolation"
    )
    # Install xDiT/xfuser - pinning to a version that works
    .pip_install("xfuser>=0.4.1")
    # Audio/TTS dependencies
    .pip_install(
        "soundfile",
        "misaki[en]",
        "num2words",
        "spacy",
        "phonemizer-fork",
        "espeakng_loader",
        "librosa",
    )
    # Core ML dependencies
    .pip_install(
        "huggingface_hub",
        "gradio>=5.0.0",
        "opencv-python",
        "numpy<2",
        "scipy",
        "moviepy",
    )
    # Requirements from the repo
    .pip_install(
        "diffusers>=0.31.0",
        "transformers>=4.49.0",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "tqdm",
        "imageio",
        "easydict",
        "ftfy",
        "dashscope",
        "imageio-ffmpeg",
        "scikit-image",
        "loguru",
        "pyloudnorm",
        "optimum-quanto==0.2.6",
        "einops",
        "safetensors",
    )
)

app = modal.App("meigen-multitalk", image=image)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=7200,  # 2 hour timeout for downloads
)
def setup_volume():
    """Clone/Update repo and download weights to the persistent volume."""
    os.makedirs(VOLUME_PATH, exist_ok=True)

    # --- 1. Repo Management ---
    if os.path.exists(f"{REPO_PATH}/.git"):
        print(f"📂 Repo exists at {REPO_PATH}, pulling latest...")
        subprocess.run(["git", "-C", REPO_PATH, "fetch", "--all"], check=True)
        subprocess.run(["git", "-C", REPO_PATH, "reset", "--hard", "origin/main"], check=True)
        print("✅ Repo updated!")
    else:
        print(f"📥 Cloning repo to {REPO_PATH}...")
        subprocess.run(["git", "clone", REPO_URL, REPO_PATH], check=True)
        print("✅ Repo cloned!")

    # --- 2. Weights Management ---
    weights_dir = f"{REPO_PATH}/weights"
    os.makedirs(weights_dir, exist_ok=True)

    # Define models: (local_folder_name, huggingface_repo_id)
    models_to_download = [
        ("Wan2.1-I2V-14B-480P", "Wan-AI/Wan2.1-I2V-14B-480P"),
        ("chinese-wav2vec2-base", "TencentGameMate/chinese-wav2vec2-base"),
        ("Kokoro-82M", "hexgrad/Kokoro-82M"),
        ("MeiGen-MultiTalk", "MeiGen-AI/MeiGen-MultiTalk"),
    ]

    for folder_name, repo_id in models_to_download:
        target_path = f"{weights_dir}/{folder_name}"
        if not os.path.exists(target_path) or not os.listdir(target_path):
            print(f"📥 Downloading {folder_name}...")
            cmd = ["huggingface-cli", "download", repo_id, "--local-dir", target_path]
            subprocess.run(cmd, check=True)
            print(f"✅ {folder_name} downloaded!")
        else:
            print(f"✅ {folder_name} already exists, skipping...")

    # Special case: Extra file for chinese-wav2vec2-base from PR 1
    wav2vec_target = f"{weights_dir}/chinese-wav2vec2-base"
    print("📥 Checking/Downloading chinese-wav2vec2-base model.safetensors (PR 1)...")
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            "TencentGameMate/chinese-wav2vec2-base",
            "model.safetensors",
            "--revision",
            "refs/pr/1",
            "--local-dir",
            wav2vec_target,
        ],
        check=True,
    )

    # --- 3. Link MultiTalk weights into Wan directory ---
    wan_dir = f"{weights_dir}/Wan2.1-I2V-14B-480P"
    multitalk_dir = f"{weights_dir}/MeiGen-MultiTalk"
    
    # Backup original index file if exists and not already backed up
    original_index = f"{wan_dir}/diffusion_pytorch_model.safetensors.index.json"
    backup_index = f"{wan_dir}/diffusion_pytorch_model.safetensors.index.json_old"
    if os.path.exists(original_index) and not os.path.exists(backup_index):
        print("📦 Backing up original index file...")
        os.rename(original_index, backup_index)
    
    # Copy MultiTalk files (more reliable than symlinks in container volumes)
    multitalk_index = f"{multitalk_dir}/diffusion_pytorch_model.safetensors.index.json"
    multitalk_weights = f"{multitalk_dir}/multitalk.safetensors"
    
    if os.path.exists(multitalk_index):
        print("📦 Copying MultiTalk index file...")
        subprocess.run(["cp", "-f", multitalk_index, wan_dir], check=True)
    
    if os.path.exists(multitalk_weights):
        print("📦 Copying MultiTalk weights...")
        subprocess.run(["cp", "-f", multitalk_weights, wan_dir], check=True)

    # Commit the volume to persist changes
    volume.commit()
    print("🎉 Setup complete! Code and weights synced to volume.")
    
    print(f"📂 {weights_dir} contents:")
    for item in os.listdir(weights_dir):
        print(f"  - {item}")


@app.function(
    gpu="L40S",
    timeout=3600,
    max_containers=1,
    cpu=4.0,
    memory=65536,  # 64GB RAM for model loading
    volumes={VOLUME_PATH: volume},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8418)
def run_gradio():
    """Run the Gradio server."""
    # Verify repo exists
    if not os.path.exists(f"{REPO_PATH}/app.py"):
        raise RuntimeError(
            f"Repo not found at {REPO_PATH}! Run 'modal run modal_serve.py::setup_volume' first."
        )

    os.chdir(REPO_PATH)
    print(f"🚀 Starting MultiTalk Gradio Server from {REPO_PATH}...")

    # Print debug info
    import torch
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA device: {torch.cuda.get_device_name(0)}")

    # Verify weights exist
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        print(f"⚠️ Warning: 'weights' directory not found in {os.getcwd()}")
    else:
        print("📂 Weights directory contents:")
        for item in os.listdir(weights_dir):
            print(f"  - {item}")

    # Check for MultiTalk weights in Wan directory
    wan_weights = f"{weights_dir}/Wan2.1-I2V-14B-480P"
    if os.path.exists(wan_weights):
        print("📂 Wan weights directory contents:")
        for item in os.listdir(wan_weights):
            print(f"  - {item}")

    # Run app with LoRA for faster inference (8 steps)
    # Using FusionX LoRA if available, otherwise run without
    lora_path = f"{weights_dir}/MeiGen-MultiTalk/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
    
    cmd = [
        "python", "app.py",
        "--ckpt_dir", f"{weights_dir}/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", f"{weights_dir}/chinese-wav2vec2-base",
        "--num_persistent_param_in_dit", "0",  # Low VRAM mode
    ]
    
    # Add LoRA if available
    if os.path.exists(lora_path):
        print(f"✅ Found FusionX LoRA at {lora_path}")
        cmd.extend([
            "--lora_dir", lora_path,
            "--lora_scale", "1.0",
            "--sample_shift", "2",
        ])
    else:
        print("ℹ️ FusionX LoRA not found, running without acceleration")
    
    print(f"📂 Running: {' '.join(cmd)}")
    subprocess.Popen(cmd)

    import time
    while True:
        time.sleep(1)


@app.local_entrypoint()
def main(setup: bool = False):
    """
    Usage:
      modal run modal_serve.py --setup    # Initial setup (clone repo + download weights)
      modal deploy modal_serve.py         # Deploy the app
      modal serve modal_serve.py          # Run development server
    """
    if setup:
        print("🚀 Running setup (clone + download)...")
        setup_volume.remote()
        print("✅ Setup complete!")
    else:
        print("ℹ️  MultiTalk Modal Deployment")
        print("  Run 'modal run modal_serve.py --setup' to initialize volume.")
        print("  Then 'modal deploy modal_serve.py' to deploy.")
