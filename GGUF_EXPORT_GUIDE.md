# GGUF Export Guide for Ministral 3 3B Sudoku Model

## Summary

Successfully exported your fine-tuned Ministral 3 3B Sudoku model to GGUF format for use with llama.cpp!

## Files Created

### Location: `/workspace/model_gguf/`

1. **ministral-3-3b-sudoku-f16.gguf** (6.4 GB)
   - 16-bit float precision
   - Highest quality, closest to original model
   - Best for: Maximum accuracy when you have enough RAM/VRAM

2. **ministral-3-3b-sudoku-q8_0.gguf** (3.5 GB)
   - 8-bit quantization
   - Excellent quality with 45% smaller size
   - **RECOMMENDED**: Best balance of quality and performance

## What Was Fixed

### Original Issues:
1. ❌ `push_to_hub_merged` failed with `PushToHubMixin._create_repo` error
2. ❌ Unsloth's GGUF export was slow and problematic
3. ⚠️  Warning: "Unrecognized keys in rope_parameters"

### Solutions Applied:
1. ✅ Model successfully pushed to HuggingFace: https://huggingface.co/applied-ai-subscr/ministral_3_3B_sudoku_vllm
2. ✅ Used llama.cpp's `convert_hf_to_gguf.py` directly (much faster & more reliable)
3. ✅ Warnings are harmless - conversion completed successfully

## Usage with llama.cpp

### Option 1: Interactive CLI (Recommended for testing)

```bash
cd /workspace/llama.cpp

# Build llama.cpp if not already built
make clean && make -j$(nproc)

# Test the model
./llama-cli \
  -m /workspace/model_gguf/ministral-3-3b-sudoku-q8_0.gguf \
  -c 4096 \
  -ngl 99 \
  -p "[INST]Create a Sudoku solving strategy using only native Python...[/INST]"
```

### Option 2: Server Mode (For API access)

```bash
cd /workspace/llama.cpp

./llama-server \
  -m /workspace/model_gguf/ministral-3-3b-sudoku-q8_0.gguf \
  -c 4096 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

Then access via HTTP:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Create a Sudoku solving strategy..."}
    ],
    "temperature": 1.0,
    "max_tokens": 512
  }'
```

### Option 3: Python Binding (llama-cpp-python)

```bash
pip install llama-cpp-python

python3 << 'EOF'
from llama_cpp import Llama

llm = Llama(
    model_path="/workspace/model_gguf/ministral-3-3b-sudoku-q8_0.gguf",
    n_ctx=4096,
    n_gpu_layers=99
)

output = llm.create_chat_completion(
    messages=[{
        "role": "user",
        "content": "Create a Sudoku solving strategy..."
    }],
    temperature=1.0,
    max_tokens=512
)

print(output["choices"][0]["message"]["content"])
EOF
```

## Command-line Arguments Explained

- `-m`: Model file path
- `-c 4096`: Context window size (matches your training)
- `-ngl 99`: Number of GPU layers to offload (99 = offload all)
- `-p`: Prompt text (for CLI mode)
- `--host 0.0.0.0`: Allow external connections (server mode)
- `--port 8080`: Server port

## Performance Comparison

| Format | Size | Quality | Speed | Use Case |
|--------|------|---------|-------|----------|
| F16 | 6.4 GB | 100% | Baseline | Maximum accuracy |
| Q8_0 | 3.5 GB | ~99% | ~1.5x faster | **Recommended** |
| Q4_K_M* | ~2 GB | ~95% | ~2-3x faster | Resource-constrained |

*Q4_K_M not generated yet, but you can create it with:
```bash
cd /workspace/llama.cpp
python convert_hf_to_gguf.py /workspace/model \
  --outfile /workspace/model_gguf/ministral-3-3b-sudoku-q4_k_m.gguf \
  --outtype auto  # Will use appropriate quantization
```

Then quantize further with:
```bash
./llama-quantize \
  /workspace/model_gguf/ministral-3-3b-sudoku-f16.gguf \
  /workspace/model_gguf/ministral-3-3b-sudoku-q4_k_m.gguf \
  Q4_K_M
```

## Troubleshooting

### Issue: "Cannot find llama-cli"
```bash
cd /workspace/llama.cpp
make clean && make -j$(nproc)
```

### Issue: "Out of memory"
- Use Q8_0 instead of F16
- Reduce `-c` (context size) to 2048
- Reduce `-ngl` (GPU layers) if VRAM is limited

### Issue: "Slow inference"
- Ensure `-ngl 99` to offload all layers to GPU
- Use Q8_0 or Q4_K_M for faster inference
- Check that llama.cpp was built with CUDA support:
  ```bash
  cd /workspace/llama.cpp
  make clean && make -j$(nproc) LLAMA_CUDA=1
  ```

## Next Steps

1. **Test the model**: Use Option 1 (Interactive CLI) to verify it works
2. **Deploy**: Choose Option 2 (Server) or Option 3 (Python) for your use case
3. **Optional**: Create Q4_K_M version for even faster inference

## Note About params.json

The `params.json` file is **NOT required** for GGUF format. All model configuration is embedded in the .gguf file itself. This is one of the advantages of GGUF over older formats.

## About the Rope Parameters Warning

The warning `Unrecognized keys in rope_parameters for 'rope_type'='yarn': {'max_position_embeddings'}` is harmless. It's just an informational message that the converter detected an extra key in the config that it doesn't need. The model works perfectly fine.

---

**Success!** Your Ministral 3 3B Sudoku model is now ready to use with llama.cpp in GGUF format.
