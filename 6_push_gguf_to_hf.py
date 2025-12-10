#!/usr/bin/env python3
"""
Upload existing GGUF files to HuggingFace Hub
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# Load environment
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise SystemExit("❌ HF_TOKEN not found in .env file")

# Configuration
GGUF_DIR = "/workspace/model_gguf"
HF_REPO = "applied-ai-subscr/ministral_3_3B_sudoku_gguf"  # Change if needed
REPO_TYPE = "model"

print("=" * 70)
print("📤 Upload GGUF files to HuggingFace Hub")
print("=" * 70)
print(f"📁 Local directory: {GGUF_DIR}")
print(f"🌐 Target repo: {HF_REPO}")
print()

# Check GGUF files exist
if not os.path.exists(GGUF_DIR):
    raise SystemExit(f"❌ Directory not found: {GGUF_DIR}")

gguf_files = list(Path(GGUF_DIR).glob("*.gguf"))
if not gguf_files:
    raise SystemExit(f"❌ No GGUF files found in {GGUF_DIR}")

print(f"✓ Found {len(gguf_files)} GGUF file(s):")
for f in gguf_files:
    size_gb = f.stat().st_size / (1024**3)
    print(f"  • {f.name} ({size_gb:.2f} GB)")
print()

# Initialize HF API
api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
print(f"🔧 Creating/checking repo: {HF_REPO}")
try:
    create_repo(
        repo_id=HF_REPO,
        token=HF_TOKEN,
        repo_type=REPO_TYPE,
        exist_ok=True,
        private=False,  # Set to True for private repo
    )
    print("✓ Repository ready")
except Exception as e:
    print(f"⚠️  Note: {e}")
print()

# Upload each GGUF file
print("📤 Uploading files...")
print()

for gguf_file in gguf_files:
    print(f"Uploading {gguf_file.name}...")
    try:
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=HF_REPO,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
        )
        print(f"  ✅ {gguf_file.name} uploaded")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    print()

# Create a simple README
readme_content = f"""---
license: apache-2.0
library_name: transformers
tags:
  - unsloth
  - gguf
  - llama.cpp
  - ministral
  - sudoku
---

# Ministral 3 3B Sudoku - GGUF

Quantized GGUF versions of the fine-tuned Ministral-3-3B model for Sudoku tasks.

## Available Quantizations

- **F16** (6.4 GB): 16-bit float, original quality
- **Q8_0** (3.5 GB): 8-bit quantization, very good quality

## Usage with llama.cpp

```bash
# Download a model
huggingface-cli download {HF_REPO} ministral-3-3b-sudoku-q8_0.gguf --local-dir ./models

# Run with llama.cpp
./llama-cli \\
  -m ./models/ministral-3-3b-sudoku-q8_0.gguf \\
  -c 4096 \\
  -ngl 99 \\
  -p "Solve this Sudoku..."

# Or start a server
./llama-server \\
  -m ./models/ministral-3-3b-sudoku-q8_0.gguf \\
  -c 4096 \\
  -ngl 99 \\
  --port 8080
```

## Model Details

- Base model: unsloth/Ministral-3-3B-Instruct-2512
- Fine-tuned with Unsloth
- Converted to GGUF using llama.cpp converter
"""

readme_path = Path(GGUF_DIR) / "README.md"
readme_path.write_text(readme_content)
print("📝 Created README.md")

try:
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type=REPO_TYPE,
        token=HF_TOKEN,
    )
    print("  ✅ README.md uploaded")
except Exception as e:
    print(f"  ⚠️  README upload failed: {e}")

print()
print("=" * 70)
print("✅ Upload complete!")
print("=" * 70)
print(f"🌐 View your model at: https://huggingface.co/{HF_REPO}")
print()
