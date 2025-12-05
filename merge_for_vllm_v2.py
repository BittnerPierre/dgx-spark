"""
Merge LoRA adapters for vLLM - Following Unsloth pattern

This script follows the exact pattern from ministral_3_rl_sudoku.py:
1. Load base model with FastVisionModel
2. Apply PEFT structure with get_peft_model (same config as training)
3. Load trained adapter weights
4. Save merged model
"""

import os
from unsloth import FastVisionModel
import torch

print("=" * 70)
print("Ministral-3-3B Sudoku - Merge for vLLM (Unsloth method)")
print("=" * 70)

# Configuration - MUST match training script
BASE_MODEL = "unsloth/Ministral-3-3B-Instruct-2512"
LORA_ADAPTERS_PATH = "grpo_saved_lora"
OUTPUT_DIR = "ministral_3_sudoku_vllm"

max_seq_length = 4096  # Same as training
lora_rank = 32         # Same as training

print(f"\n📥 Loading base model: {BASE_MODEL}")

# Step 1: Load base model (same as training)
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = max_seq_length,
    load_in_4bit = False,  # False for LoRA 16bit
)
print("✓ Base model loaded")

# Step 2: Apply PEFT structure (same config as training!)
print(f"\n📦 Applying PEFT structure (rank={lora_rank})...")
model = FastVisionModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank * 2,  # 64
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
print("✓ PEFT structure created")

# Step 3: Load trained adapter weights
print(f"\n📦 Loading trained adapter weights from: {LORA_ADAPTERS_PATH}")

# Load adapter weights from saved checkpoint
from safetensors.torch import load_file
import os.path as osp

adapter_file = osp.join(LORA_ADAPTERS_PATH, "adapter_model.safetensors")
if osp.exists(adapter_file):
    adapter_weights = load_file(adapter_file)
    print(f"✓ Loaded {len(adapter_weights)} adapter tensors")

    # Load into model
    model.load_state_dict(adapter_weights, strict=False)
    print("✓ Adapter weights loaded into model")
else:
    print(f"❌ ERROR: {adapter_file} not found!")
    print(f"   Available files in {LORA_ADAPTERS_PATH}:")
    for f in os.listdir(LORA_ADAPTERS_PATH):
        print(f"   - {f}")
    exit(1)

# Step 4: Save merged model (Unsloth method)
print(f"\n💾 Merging and saving to: {OUTPUT_DIR}")
print("⏳ This may take a few minutes...")

model.save_pretrained_merged(
    OUTPUT_DIR,
    tokenizer,
    save_method = "merged_16bit",
)

print(f"\n✅ Model merged successfully!")
print(f"📁 Output: /workspace/{OUTPUT_DIR}/")

# Deployment instructions
print("\n" + "=" * 70)
print("🚀 Deploy with vLLM")
print("=" * 70)

print(f"""
docker run -d \\
  --name vllm_ministral_sudoku \\
  --gpus all \\
  --ipc=host \\
  -p 8003:8000 \\
  -v /workspace/{OUTPUT_DIR}:/model \\
  nvcr.io/nvidia/vllm:25.09-py3 \\
  vllm serve /model \\
    --tokenizer_mode mistral \\
    --config_format mistral \\
    --load_format mistral \\
    --gpu-memory-utilization 0.9

Test:
curl http://localhost:8003/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "/model", "prompt": "Solve sudoku", "max_tokens": 512}}'
""")

print("=" * 70)
print("✅ Done!")
print("=" * 70)
