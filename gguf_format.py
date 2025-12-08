"""
Generate GGUF quantized versions of the merged Ministral Sudoku model
and push them to HuggingFace Hub.

- Input  : merged bf16 model on disk (Unsloth format)
- Output : multiple GGUF quantizations (q4_k_m, q8_0, q5_k_m)
           saved locally AND pushed to HF.
"""

import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# 1. Load HF token
# ─────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN not found in .env file")
    print("   Please create a .env file with: HF_TOKEN=your_token_here")
    exit(1)

# ─────────────────────────────────────────────────────────────
# 2. Paths & config
# ─────────────────────────────────────────────────────────────
# Dossier où tu as sauvegardé le modèle mergé (bf16) après Unsloth
MERGED_MODEL_DIR = "/models/fine-tuned/ministral_3_sudoku_vllm"

# Repo HF où tu veux stocker les GGUF
# 👉 tu peux garder le même ou créer un repo dédié type "..._gguf"
HF_REPO_GGUF = "applied-ai-subscr/ministral_3_sudoku_vllm-gguf"

# Dossier local où on mettra les GGUF
LOCAL_GGUF_DIR = "/models/gguf/ministral_3_sudoku"

QUANT_METHODS = ["q4_k_m", "q8_0", "q5_k_m"]

print("=" * 70)
print("🎯 Generate GGUF quantized models from merged Ministral Sudoku")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 3. Check merged model
# ─────────────────────────────────────────────────────────────
if not os.path.exists(MERGED_MODEL_DIR):
    print(f"❌ ERROR: Merged model not found at {MERGED_MODEL_DIR}")
    print("   Make sure your Unsloth merged model is saved there.")
    exit(1)

print(f"📁 Loading merged model from: {MERGED_MODEL_DIR}")

from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MERGED_MODEL_DIR,
    max_seq_length=4096,
    load_in_4bit=False,   # merged bf16 model on disk
)

print("✅ Merged model loaded successfully")

# ─────────────────────────────────────────────────────────────
# 4. Save GGUF quantizations locally
# ─────────────────────────────────────────────────────────────
os.makedirs(LOCAL_GGUF_DIR, exist_ok=True)

print(f"\n💾 Saving GGUF quantizations locally to: {LOCAL_GGUF_DIR}")
print(f"   Quantizations: {', '.join(QUANT_METHODS)}")

model.save_pretrained_gguf(
    LOCAL_GGUF_DIR,
    tokenizer,
    quantization_method=QUANT_METHODS,
)

print("✅ Local GGUF files saved")

# ─────────────────────────────────────────────────────────────
# 5. Push GGUF to HuggingFace Hub
# ─────────────────────────────────────────────────────────────
print(f"\n🚀 Pushing GGUF quantizations to HF repo: {HF_REPO_GGUF}")

model.push_to_hub_gguf(
    HF_REPO_GGUF,
    tokenizer,
    quantization_method=QUANT_METHODS,
    token=HF_TOKEN,
)

print("✅ GGUF models pushed to HuggingFace Hub")

# ─────────────────────────────────────────────────────────────
# 6. Next steps (llama.cpp)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Done! Next steps with llama.cpp")
print("=" * 70)

print(f"""
You now have GGUF files in:
  {LOCAL_GGUF_DIR}

Example with llama.cpp:

  ./main \\
    -m {LOCAL_GGUF_DIR}/ministral_3_sudoku-q4_k_m.gguf \\
    -c 4096 \\
    -ngl 999 \\
    -n 256

Or run the server:

  ./server \\
    -m {LOCAL_GGUF_DIR}/ministral_3_sudoku-q4_k_m.gguf \\
    -c 4096 \\
    -ngl 999 \\
    --port 8080
""")
print("=" * 70)
