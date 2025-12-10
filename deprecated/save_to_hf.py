"""
Save merged model to HuggingFace Hub in proper format for vLLM

This script:
1. Loads the already-merged model from disk
2. Pushes it to HuggingFace Hub with proper format
3. Downloads it back to local cache in correct structure
"""

import os
from dotenv import load_dotenv

# Load HF token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN not found in .env file")
    print("   Please create a .env file with: HF_TOKEN=your_token_here")
    exit(1)

# Configuration
MERGED_MODEL_DIR = "/models/fine-tuned/ministral_3_sudoku_vllm"
HF_REPO_NAME = "applied-ai-subscr/ministral_3_sudoku_vllm"

print("=" * 70)
print("Push Model to HuggingFace Hub")
print("=" * 70)

# Check if merged model exists
if not os.path.exists(MERGED_MODEL_DIR):
    print(f"❌ ERROR: Merged model not found at {MERGED_MODEL_DIR}")
    print("   Please run merge_for_vllm_v2.py first")
    exit(1)

print(f"\n📁 Loading model from: {MERGED_MODEL_DIR}")
print(f"🔄 Pushing to: {HF_REPO_NAME}")

from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MERGED_MODEL_DIR,
    max_seq_length=4096,
    load_in_4bit=False,
)

print("✓ Model loaded")

# Push to hub
print(f"\n🚀 Pushing to HuggingFace Hub: {HF_REPO_NAME}")
print("⏳ This may take several minutes...")

model.push_to_hub(
    HF_REPO_NAME,
    token=HF_TOKEN,
    private=False,  # Set to True if you want a private repo
)

tokenizer.push_to_hub(
    HF_REPO_NAME,
    token=HF_TOKEN,
    private=False,
)

print("✅ Model pushed successfully to HuggingFace Hub!")

# Now download to local cache in proper format
print("\n📥 Downloading to local HuggingFace cache...")
print("   This ensures proper cache structure for vLLM")

from huggingface_hub import snapshot_download

cache_dir = snapshot_download(
    repo_id=HF_REPO_NAME,
    token=HF_TOKEN,
    cache_dir="/models",  # Your cache directory
)

print(f"✓ Model cached at: {cache_dir}")

# Deployment instructions
print("\n" + "=" * 70)
print("🚀 Deploy with vLLM")
print("=" * 70)

print(f"""
Now you can run:

docker run -d \\
  --name vllm_ministral_sudoku \\
  --gpus all \\
  --ipc=host \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  -p 8003:8000 \\
  -v /mnt/models:/root/.cache/huggingface \\
  nvcr.io/nvidia/vllm:25.09-py3 \\
  vllm serve {HF_REPO_NAME} \\
    --tokenizer_mode mistral \\
    --config_format mistral \\
    --load_format mistral \\
    --gpu-memory-utilization 0.9

Or with offline mode:

docker run -d \\
  --name vllm_ministral_sudoku \\
  --gpus all \\
  --ipc=host \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  -p 8003:8000 \\
  -e HF_HUB_OFFLINE=1 \\
  -v /mnt/models:/root/.cache/huggingface \\
  nvcr.io/nvidia/vllm:25.09-py3 \\
  vllm serve {HF_REPO_NAME} \\
    --tokenizer_mode mistral \\
    --config_format mistral \\
    --load_format mistral \\
    --gpu-memory-utilization 0.9
""")

print("=" * 70)
print("✅ Done!")
print("=" * 70)