"""
Simple merge script for vLLM inference (FP16)

This script:
1. Loads Ministral-3-3B-Instruct base model
2. Loads fine-tuned LoRA adapters from GRPO training
3. Merges them to FP16 format
4. Ready for vLLM deployment

For a 3B model, FP16 is perfect (~6GB VRAM on 119GB DGX Spark).
No need for quantization!
"""

import os
from unsloth import FastVisionModel
import torch

print("=" * 70)
print("Ministral-3-3B Sudoku - Simple Merge for vLLM")
print("=" * 70)

# Configuration
BASE_MODEL = "unsloth/Ministral-3-3B-Instruct-2512"
LORA_ADAPTERS_PATH = "grpo_saved_lora"
OUTPUT_DIR = "ministral_3_sudoku_vllm"

print(f"\n📥 Loading base model: {BASE_MODEL}")
print(f"📥 Loading LoRA adapters: {LORA_ADAPTERS_PATH}")

# Load base model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = 4096,
    load_in_4bit = False,
    fast_inference = True,
)
print("✓ Base model loaded")

# Load LoRA adapters
print(f"\n📦 Loading LoRA adapters...")
from peft import PeftModel
model.load_adapter(LORA_ADAPTERS_PATH)
print("✓ LoRA adapters loaded")

# Merge to 16bit
print(f"\n💾 Merging to FP16 for vLLM: {OUTPUT_DIR}")
print("⏳ This will take a few minutes...")

model.save_pretrained_merged(
    OUTPUT_DIR,
    tokenizer,
    save_method = "merged_16bit",
)

print(f"\n✅ Model merged successfully!")
print(f"📁 Output: /workspace/{OUTPUT_DIR}/")
print(f"💾 Size: ~6GB")

# Deployment instructions
print("\n" + "=" * 70)
print("🚀 Deploy with vLLM")
print("=" * 70)

print(f"""
Run this command to start vLLM:

docker run -d \\
  --name vllm_ministral_sudoku \\
  --gpus all \\
  --ipc=host \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  -p 8003:8000 \\
  -v /workspace/{OUTPUT_DIR}:/model \\
  nvcr.io/nvidia/vllm:25.09-py3 \\
  vllm serve /model \\
    --tokenizer_mode mistral \\
    --config_format mistral \\
    --load_format mistral \\
    --gpu-memory-utilization 0.9

Test with completions endpoint:
curl http://localhost:8003/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "/model",
    "prompt": "def strategy(board, initial):\\n    # Solve sudoku",
    "max_tokens": 512,
    "temperature": 0.7
  }}'

Check container logs:
docker logs -f vllm_ministral_sudoku

Stop container:
docker stop vllm_ministral_sudoku
""")

print("=" * 70)
print("✅ Done! Follow commands above to deploy.")
print("=" * 70)
