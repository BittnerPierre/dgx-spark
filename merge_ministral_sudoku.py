"""
Merge LoRA adapters with Ministral-3-3B base model and export for inference.

This script:
1. Loads Ministral-3-3B-Instruct base model
2. Loads the fine-tuned LoRA adapters from GRPO training
3. Merges them together
4. Exports in multiple formats:
   - merged_16bit: For vLLM inference (recommended)
   - q4_k_m GGUF: For llama.cpp inference (smaller, faster)

Requirements:
- Same packages as ministral_3_rl_sudoku.py
- Enough disk space (~7GB for 16bit + ~2GB for GGUF)
"""

import os
from unsloth import FastVisionModel
import torch

print("=" * 60)
print("Ministral-3-3B Sudoku Model - Merge Script")
print("=" * 60)

# Configuration
BASE_MODEL = "unsloth/Ministral-3-3B-Instruct-2512"
LORA_ADAPTERS_PATH = "grpo_saved_lora"  # Path to LoRA adapters
OUTPUT_DIR_16BIT = "ministral_3_sudoku_16bit"  # For vLLM
OUTPUT_DIR_GGUF = "ministral_3_sudoku_gguf"    # For llama.cpp

print(f"\n📥 Loading base model: {BASE_MODEL}")
print(f"📥 Loading LoRA adapters from: {LORA_ADAPTERS_PATH}")

# Load base model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = 4096,  # Same as training
    load_in_4bit = False,   # Must be False for merging
    fast_inference = True,  # Enable for faster inference
)

print("✓ Base model loaded")

# Load LoRA adapters
print(f"\n📦 Loading LoRA adapters...")
from peft import PeftModel

# The model from FastVisionModel is already prepared, just load the adapter
model.load_adapter(LORA_ADAPTERS_PATH)
print("✓ LoRA adapters loaded")

# Verify adapters are loaded
print("\n🔍 Verifying LoRA adapters...")
adapter_weights = model.get_adapter_state_dict(LORA_ADAPTERS_PATH)
print(f"✓ Found {len(adapter_weights)} adapter weights")

# === Option 1: Merge to 16bit for vLLM ===
print("\n" + "=" * 60)
print("Option 1: Merging to 16bit for vLLM (unquantized)")
print("=" * 60)

print(f"💾 Saving merged model to: {OUTPUT_DIR_16BIT}")
print("⏳ This may take a few minutes...")

model.save_pretrained_merged(
    OUTPUT_DIR_16BIT,
    tokenizer,
    save_method = "merged_16bit",
)

print(f"✅ 16bit model saved to: {OUTPUT_DIR_16BIT}")
print(f"   Use with vLLM: vllm serve {OUTPUT_DIR_16BIT}")

# === Option 1b: Merge to 4bit for vLLM (using merged_4bit) ===
print("\n" + "=" * 60)
print("Option 1b: Merging to 4bit for vLLM (quantized)")
print("=" * 60)

OUTPUT_DIR_4BIT = "ministral_3_sudoku_4bit"
print(f"💾 Saving 4bit model to: {OUTPUT_DIR_4BIT}")
print("⏳ This may take a few minutes...")

model.save_pretrained_merged(
    OUTPUT_DIR_4BIT,
    tokenizer,
    save_method = "merged_4bit",
)

print(f"✅ 4bit model saved to: {OUTPUT_DIR_4BIT}")
print(f"   Use with vLLM: vllm serve {OUTPUT_DIR_4BIT} --quantization bitsandbytes")
print(f"   ⚠️  Note: This uses bitsandbytes, which may be slower than AWQ")

# === Option 2: Quantize to q4_k_m GGUF for llama.cpp ===
print("\n" + "=" * 60)
print("Option 2: Quantizing to q4_k_m GGUF for llama.cpp")
print("=" * 60)

print(f"💾 Saving GGUF model to: {OUTPUT_DIR_GGUF}")
print("⏳ Quantization will take several minutes...")

model.save_pretrained_gguf(
    OUTPUT_DIR_GGUF,
    tokenizer,
    quantization_method = "q4_k_m",
)

print(f"✅ GGUF model saved to: {OUTPUT_DIR_GGUF}")
print(f"   Use with llama.cpp: llama-server -m {OUTPUT_DIR_GGUF}/model.gguf")

# === Summary ===
print("\n" + "=" * 60)
print("✅ MERGE COMPLETE!")
print("=" * 60)
print(f"\n📁 Output directories:")
print(f"   1. {OUTPUT_DIR_16BIT}/ (for vLLM 16bit)")
print(f"   2. {OUTPUT_DIR_4BIT}/ (for vLLM 4bit)")
print(f"   3. {OUTPUT_DIR_GGUF}/ (for llama.cpp)")

print(f"\n🚀 Next steps:")
print(f"   For vLLM 16bit (best quality, ~6-7GB VRAM):")
print(f"      docker run -d \\")
print(f"        --gpus all \\")
print(f"        -p 8003:8000 \\")
print(f"        -v /workspace/{OUTPUT_DIR_16BIT}:/model \\")
print(f"        nvcr.io/nvidia/vllm:25.09-py3 \\")
print(f"        vllm serve /model")
print(f"\n   For vLLM 4bit (good quality, ~2-3GB VRAM) ⭐ RECOMMENDED:")
print(f"      docker run -d \\")
print(f"        --gpus all \\")
print(f"        -p 8003:8000 \\")
print(f"        -v /workspace/{OUTPUT_DIR_4BIT}:/model \\")
print(f"        nvcr.io/nvidia/vllm:25.09-py3 \\")
print(f"        vllm serve /model --quantization bitsandbytes")
print(f"\n   For llama.cpp (lowest memory, ~2GB VRAM):")
print(f"      llama-server -m /workspace/{OUTPUT_DIR_GGUF}/*.gguf -ngl 99")

print("\n💡 Comparison:")
print("   - vLLM 16bit: Best quality, 6-7GB VRAM, fastest inference")
print("   - vLLM 4bit: Good quality, 2-3GB VRAM, fast inference ⭐")
print("   - llama.cpp q4_k_m: Good quality, 2GB VRAM, slower inference")
print("\n   Recommendation: Try vLLM 4bit first!")
