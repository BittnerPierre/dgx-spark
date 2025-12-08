"""
Export merged Ministral 3 model to GGUF format for llama.cpp
Works with the already-merged model in /workspace/model/

This script:
1. Loads the merged 16bit model from disk (NO model loading into GPU needed)
2. Uses Unsloth's GGUF export to create quantized versions
3. Optionally pushes to HuggingFace Hub
"""
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Configuration
MERGED_MODEL_DIR = "/workspace/model"
HF_REPO_NAME = "applied-ai-subscr/ministral_3_3B_sudoku_vllm"
GGUF_OUTPUT_DIR = "/workspace/model_gguf"

# GGUF quantization methods to generate
# See: https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp
QUANT_METHODS = [
    "q4_k_m",  # 4-bit, recommended, good quality
    "q8_0",    # 8-bit, very good quality
    "q5_k_m",  # 5-bit, balance between q4 and q8
    "f16",     # 16-bit float, original quality
]

print("=" * 70)
print("🎯 Export Ministral 3 3B Sudoku to GGUF format")
print("=" * 70)
print(f"📁 Input model: {MERGED_MODEL_DIR}")
print(f"📦 Output dir: {GGUF_OUTPUT_DIR}")
print(f"🔧 Quantizations: {', '.join(QUANT_METHODS)}")
print()

# Check if merged model exists
if not os.path.exists(MERGED_MODEL_DIR):
    raise SystemExit(f"❌ Merged model not found at {MERGED_MODEL_DIR}")

# Verify required files exist
required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
for fname in required_files:
    fpath = os.path.join(MERGED_MODEL_DIR, fname)
    if not os.path.exists(fpath):
        print(f"⚠️  Warning: {fname} not found")
    else:
        print(f"✓ Found: {fname}")

print()

# Load model using Unsloth (lightweight, just for export)
print("🔄 Loading merged model for GGUF export...")
print("   (This uses minimal GPU memory)")
print()

from unsloth import FastVisionModel
import torch

try:
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MERGED_MODEL_DIR,
        max_seq_length=4096,
        load_in_4bit=False,  # We have the 16bit merged model
        dtype=None,  # Auto-detect
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print()
    print("Troubleshooting:")
    print("  1. Make sure the merged model is complete in /workspace/model/")
    print("  2. Check that safetensors files are not corrupted")
    print("  3. Verify you have enough disk space")
    raise

# Create output directory
os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

# Export to GGUF
print()
print("=" * 70)
print("🔧 Converting to GGUF formats...")
print("=" * 70)
print()
print("⚠️  Note: This will take several minutes per quantization")
print("         Multi-quantization is faster than running separately")
print()

try:
    # Save locally first
    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        tokenizer,
        quantization_method=QUANT_METHODS,
    )

    print()
    print("=" * 70)
    print("✅ GGUF export complete!")
    print("=" * 70)
    print(f"📁 GGUF files saved to: {GGUF_OUTPUT_DIR}")
    print()

    # List generated files
    if os.path.exists(GGUF_OUTPUT_DIR):
        gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith('.gguf')]
        if gguf_files:
            print("Generated GGUF files:")
            for fname in sorted(gguf_files):
                fpath = os.path.join(GGUF_OUTPUT_DIR, fname)
                size_gb = os.path.getsize(fpath) / (1024**3)
                print(f"  • {fname} ({size_gb:.2f} GB)")
        else:
            print("⚠️  No .gguf files found in output directory")
    print()

except Exception as e:
    print()
    print("=" * 70)
    print("❌ GGUF export failed!")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    print("Common issues:")
    print("  1. Disk space: GGUF files are large, ensure enough space")
    print("  2. Memory: Some quantizations need significant RAM")
    print("  3. Model format: Ensure the input model is properly merged")
    raise

# Optionally push to HuggingFace Hub
push_to_hub = input("\n📤 Push GGUF files to HuggingFace Hub? (y/N): ").lower().strip() == 'y'

if push_to_hub:
    if not HF_TOKEN:
        print("❌ HF_TOKEN not found. Skipping push to hub.")
    else:
        print()
        print("🚀 Pushing GGUF files to HuggingFace Hub...")
        print(f"   Target: {HF_REPO_NAME}")
        print()

        try:
            model.push_to_hub_gguf(
                HF_REPO_NAME,
                tokenizer,
                quantization_method=QUANT_METHODS,
                token=HF_TOKEN,
            )
            print()
            print("✅ GGUF files pushed to HuggingFace Hub")
            print(f"🌐 View at: https://huggingface.co/{HF_REPO_NAME}")
        except Exception as e:
            print(f"❌ Push to hub failed: {e}")
            print("   But local GGUF files are still available!")

print()
print("=" * 70)
print("📖 Usage with llama.cpp")
print("=" * 70)
print()
print("To use with llama.cpp:")
print()
print(f"  ./llama-cli \\")
print(f"    -m {GGUF_OUTPUT_DIR}/unsloth.Q4_K_M.gguf \\")
print(f"    -c 4096 \\")
print(f"    -ngl 99 \\")
print(f"    --chat-template mistral \\")
print(f"    -p 'Create a Sudoku solving strategy...'")
print()
print("Or start a server:")
print()
print(f"  ./llama-server \\")
print(f"    -m {GGUF_OUTPUT_DIR}/unsloth.Q4_K_M.gguf \\")
print(f"    -c 4096 \\")
print(f"    -ngl 99 \\")
print(f"    --port 8080")
print()
print("=" * 70)
