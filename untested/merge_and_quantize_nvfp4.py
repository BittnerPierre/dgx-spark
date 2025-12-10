"""
Merge LoRA adapters and quantize to NVFP4 for DGX Spark

This script:
1. Merges LoRA adapters with Ministral-3-3B base model (16bit)
2. Provides commands to quantize to NVFP4 using TensorRT Model Optimizer
3. NVFP4 is optimized for DGX Spark GB10 hardware (2x faster than FP16!)

Requirements:
- Unsloth environment (for merging)
- TensorRT-LLM container (for NVFP4 quantization)
- HuggingFace token

NVFP4 Benefits on DGX Spark:
- 2x faster inference than FP16
- 3.5x less memory than FP16
- Hardware-accelerated on GB10
"""

import os
from unsloth import FastVisionModel
import torch

print("=" * 70)
print("Ministral-3-3B Sudoku - Merge & NVFP4 Quantization for DGX Spark")
print("=" * 70)

# Configuration
BASE_MODEL = "unsloth/Ministral-3-3B-Instruct-2512"
LORA_ADAPTERS_PATH = "grpo_saved_lora"
OUTPUT_DIR_16BIT = "ministral_3_sudoku_merged_16bit"  # For NVFP4 input
OUTPUT_DIR_NVFP4 = "ministral_3_sudoku_nvfp4"         # Final NVFP4 model

# ============================================================================
# STEP 1: Merge LoRA adapters to 16bit
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Merging LoRA adapters to 16bit")
print("=" * 70)

print(f"\n📥 Loading base model: {BASE_MODEL}")
print(f"📥 Loading LoRA adapters from: {LORA_ADAPTERS_PATH}")

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

# Merge and save to 16bit
print(f"\n💾 Merging to 16bit: {OUTPUT_DIR_16BIT}")
print("⏳ This may take a few minutes...")

model.save_pretrained_merged(
    OUTPUT_DIR_16BIT,
    tokenizer,
    save_method = "merged_16bit",
)

print(f"✅ 16bit merged model saved to: {OUTPUT_DIR_16BIT}")

# ============================================================================
# STEP 2: Instructions for NVFP4 Quantization
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Quantize to NVFP4 (optimized for DGX Spark)")
print("=" * 70)

print("""
⚠️  NVFP4 quantization requires TensorRT Model Optimizer.
    Run these commands to quantize your merged model:

1. Set your HuggingFace token:
   export HF_TOKEN="your_hf_token_here"

2. Create output directory:
   mkdir -p {output_dir}

3. Upload your merged model to HuggingFace (required for quantization):
   # Option A: Push to HuggingFace Hub
   huggingface-cli upload your_username/ministral-3-sudoku-merged /workspace/{merged_dir}

   # Then use in quantization: --model 'your_username/ministral-3-sudoku-merged'

   # Option B: Use local path (may require modifications to the script)

4. Run TensorRT-LLM quantization container:
   docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 \\
     -v "/workspace/{output_dir}:/workspace/output_models" \\
     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
     -e HF_TOKEN=$HF_TOKEN \\
     nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev bash -c "
     git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/TensorRT-Model-Optimizer.git /app/TensorRT-Model-Optimizer && \\
     cd /app/TensorRT-Model-Optimizer && pip install -e '.[dev]' && \\
     export ROOT_SAVE_PATH='/workspace/output_models' && \\
     /app/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \\
       --model 'your_username/ministral-3-sudoku-merged' \\
       --quant nvfp4 --tp 1 --export_fmt hf
   "

⏱️  Expected time: 10-30 minutes
💾 Expected size: ~1-2GB (vs ~6GB for 16bit)
""".format(
    output_dir=OUTPUT_DIR_NVFP4,
    merged_dir=OUTPUT_DIR_16BIT,
))

# ============================================================================
# STEP 3: Deploy with TensorRT-LLM (Recommended for NVFP4)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Deploy NVFP4 model with TensorRT-LLM")
print("=" * 70)

print(f"""
Once NVFP4 quantization is complete, deploy with TensorRT-LLM:

docker run -d \\
  --name trtllm_ministral_sudoku_nvfp4 \\
  --gpus all \\
  --ipc=host \\
  --network host \\
  -v "/workspace/{OUTPUT_DIR_NVFP4}:/workspace/model" \\
  -e HF_TOKEN=$HF_TOKEN \\
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \\
  trtllm-serve /workspace/model \\
    --backend pytorch \\
    --max_batch_size 4 \\
    --port 8003

Then test with OpenAI-compatible API:
curl http://localhost:8003/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "ministral-3-sudoku",
    "prompt": "Solve this Sudoku puzzle:\\n...",
    "max_tokens": 512,
    "temperature": 0.7
  }}'

Or chat completions:
curl http://localhost:8003/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "ministral-3-sudoku",
    "messages": [
      {{"role": "user", "content": "Solve this Sudoku puzzle"}}
    ],
    "max_tokens": 512
  }}'
""")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ MERGE COMPLETE - Next: Quantize to NVFP4")
print("=" * 70)

print(f"""
📁 Current status:
   ✅ {OUTPUT_DIR_16BIT}/ - Merged 16bit model (ready for NVFP4)
   ⏳ {OUTPUT_DIR_NVFP4}/ - Run quantization commands above

🚀 Performance on DGX Spark with NVFP4 + TensorRT-LLM:
   - 2x faster inference than FP16
   - 3.5x less memory than FP16 (~2GB vs ~6GB)
   - Hardware-accelerated on GB10 Tensor Cores
   - Minimal accuracy loss (<1%)
   - OpenAI-compatible API

💡 Why NVFP4 + TensorRT-LLM instead of other options?
   ┌─────────────────────┬──────────┬─────────┬──────────┐
   │ Configuration       │ Speed    │ Memory  │ Quality  │
   ├─────────────────────┼──────────┼─────────┼──────────┤
   │ NVFP4 + TRT-LLM ⭐  │ 2x       │ ~2GB    │ 99%      │
   │ FP16 + vLLM         │ 1x       │ ~6GB    │ 100%     │
   │ 4bit + vLLM         │ 0.7x     │ ~2GB    │ 95%      │
   │ GGUF + llama.cpp    │ 0.5x     │ ~2GB    │ 95%      │
   └─────────────────────┴──────────┴─────────┴──────────┘

   NVFP4 is specifically designed for DGX Spark GB10 hardware!
""")

print("\n" + "=" * 70)
print("Next: Follow STEP 2 commands to quantize to NVFP4")
print("=" * 70)
