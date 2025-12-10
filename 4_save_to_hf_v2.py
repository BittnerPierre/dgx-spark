"""
Push merged model (already in HF format) to HuggingFace Hub,
then download it in a proper cache structure for vLLM.
"""
# !pip install huggingface_hub
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder, snapshot_download

# 1. Load token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN not found in .env file")
    print("   Please create a .env file with: HF_TOKEN=your_token_here")
    raise SystemExit(1)

# 2. Config
MERGED_MODEL_DIR = "/models/fine-tuned/ministral_3_sudoku_vllm"
HF_REPO_NAME = "applied-ai-subscr/ministral_3_sudoku_vllm"

print("=" * 70)
print("Push Model to HuggingFace Hub (no GPU needed)")
print("=" * 70)

if not os.path.exists(MERGED_MODEL_DIR):
    print(f"❌ ERROR: Merged model not found at {MERGED_MODEL_DIR}")
    print("   Please run your merge script first.")
    raise SystemExit(1)

print(f"\n📁 Local model dir : {MERGED_MODEL_DIR}")
print(f"🔄 Remote repo     : {HF_REPO_NAME}")

# 3. Create repo (if not exists)
api = HfApi()
create_repo(
    repo_id=HF_REPO_NAME,
    token=HF_TOKEN,
    private=False,   # True si tu veux un repo privé
    exist_ok=True,
)

# 4. Upload folder as-is
print("\n🚀 Uploading folder to HuggingFace Hub...")
upload_folder(
    repo_id=HF_REPO_NAME,
    folder_path=MERGED_MODEL_DIR,
    token=HF_TOKEN,
)

print("✅ Model folder uploaded to HuggingFace Hub!")

# 5. Download to local cache with proper HF structure
print("\n📥 Downloading to local HF cache (/mnt/models)...")

cache_dir = snapshot_download(
    repo_id=HF_REPO_NAME,
    token=HF_TOKEN,
    cache_dir="/models",
)

print(f"✓ Model cached at: {cache_dir}")

print("\n" + "=" * 70)
print("🚀 Deploy with NVIDIA vLLM (DGX Spark)")
print("=" * 70)

print(f"""
docker run -d \\
  --name vllm_ministral_sudoku \\
  --gpus all \\
  --ipc=host \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  -p 8003:8000 \\
  -e HF_TOKEN={HF_TOKEN} \\
  -v /mnt/models:/root/.cache/huggingface \\
  nvcr.io/nvidia/vllm:25.11-py3 \\
  vllm serve {HF_REPO_NAME} \\
    --tokenizer_mode mistral \\
    --config_format mistral \\
    --load_format mistral \\
    --gpu-memory-utilization 0.9
""")

print("=" * 70)
print("✅ Done!")
print("=" * 70)
