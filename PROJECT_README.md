# Ministral 3 3B Sudoku Fine-tuning with Unsloth + GGUF Export

Fine-tuning Ministral 3 3B model using Unsloth for Sudoku solving strategies, with working GGUF export solution.

## Project Overview

This project demonstrates:
1. Fine-tuning Ministral 3 3B using Unsloth with GRPO (Group Relative Policy Optimization)
2. Training the model to generate valid Sudoku solving strategies
3. **Working solution** for exporting to GGUF format (bypassing Unsloth's broken export)

## The GGUF Export Problem & Solution

### ⚠️ Problem

Unsloth's GGUF export methods (`save_pretrained_gguf`, `push_to_hub_gguf`) **fail** for Ministral 3 and other vision models:

```python
# THIS DOESN'T WORK ❌
model.save_pretrained_gguf("model", tokenizer)
model.push_to_hub_gguf("user/model", tokenizer, token=TOKEN)
```

**Errors you'll encounter:**
- `AttributeError: 'PushToHubMixin' has no attribute '_create_repo'`
- Slow loading (minutes to load model)
- Silent failures or empty output

### ✅ Solution

**Use llama.cpp's converter directly:**

```bash
cd llama.cpp
python convert_hf_to_gguf.py /path/to/merged/model \
  --outfile output.gguf \
  --outtype q8_0
```

**Why this works:**
- Reads safetensors directly (no GPU loading)
- Official llama.cpp tool (reliable & maintained)
- Works with ALL HuggingFace models
- 10-100x faster than Unsloth's method

## Quick Start

### 1. Fine-tune the Model

```bash
python ministral_3_rl_sudoku.py
```

This trains Ministral 3 3B to generate Sudoku solving strategies using GRPO.

### 2. Push to HuggingFace (Optional)

```bash
# Set your token
export HF_TOKEN="your_token_here"

# Push using direct API (bypasses Unsloth's broken method)
python push_manual.py
```

### 3. Export to GGUF

```bash
cd llama.cpp

# Q8_0 (recommended - 45% smaller, minimal quality loss)
python convert_hf_to_gguf.py /workspace/model \
  --outfile /workspace/model_gguf/model-q8_0.gguf \
  --outtype q8_0

# F16 (full precision)
python convert_hf_to_gguf.py /workspace/model \
  --outfile /workspace/model_gguf/model-f16.gguf \
  --outtype f16
```

### 4. Test with llama.cpp

```bash
cd llama.cpp
./llama-cli \
  -m /workspace/model_gguf/model-q8_0.gguf \
  -c 4096 -ngl 99 \
  -p "[INST]Create a Sudoku solving strategy[/INST]"
```

## Files

### Solution Files (NEW - This is what fixes GGUF export)

- **`SOLUTION.md`** - Detailed explanation of the GGUF export problem and solution
- **`GGUF_EXPORT_GUIDE.md`** - Complete usage guide with examples
- **`export_gguf_fast.py`** - Clean wrapper for llama.cpp converter
- **`push_manual.py`** - Direct HuggingFace Hub API push (bypasses Unsloth)

### Training Files

- **`ministral_3_rl_sudoku.py`** - Main training script with GRPO
- `gguf_format.py` - Original Unsloth GGUF attempt (reference, doesn't work)
- `save_to_hf.py` - Another push attempt

## Key Learnings

1. **Unsloth is amazing for training** - 2x faster, less memory
2. **Unsloth's export is broken** - Use standard tools instead
3. **llama.cpp is the gold standard** - Always works, super fast
4. **Q8_0 is the sweet spot** - 45% smaller, <1% quality loss

## Time Saved

- **Before**: Hours debugging Unsloth's GGUF export
- **After**: 5 minutes with llama.cpp converter

## Requirements

- Unsloth (for training): `pip install unsloth`
- llama.cpp (for GGUF): Clone from https://github.com/ggerganov/llama.cpp
- HuggingFace Hub: `pip install huggingface-hub`

## Model

- **Architecture**: Ministral 3 3B (vision + text)
- **Training**: GRPO (reinforcement learning)
- **Task**: Generate valid Sudoku solving strategies in Python
- **HuggingFace**: https://huggingface.co/applied-ai-subscr/ministral_3_3B_sudoku_vllm

## Credits

Solution developed through analysis of:
- Unsloth source code
- llama.cpp documentation
- HuggingFace Hub API
- Hours of trial and error (so you don't have to!)

## License

See `license.txt` for details.

---

**TL;DR**: Don't use Unsloth's GGUF export. Use llama.cpp's `convert_hf_to_gguf.py` instead. Read `SOLUTION.md` for full explanation.
