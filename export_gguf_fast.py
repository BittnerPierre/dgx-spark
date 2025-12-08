#!/usr/bin/env python3
"""
Fast GGUF export using llama.cpp's convert_hf_to_gguf.py
This is much faster than Unsloth's method as it directly converts safetensors
"""
import os
import subprocess
from pathlib import Path

# Configuration
MODEL_DIR = "/workspace/model"
OUTPUT_DIR = "/workspace/model_gguf"
LLAMA_CPP_CONVERTER = "/workspace/llama.cpp/convert_hf_to_gguf.py"

print("=" * 70)
print("🚀 Fast GGUF Export using llama.cpp converter")
print("=" * 70)
print(f"📁 Input: {MODEL_DIR}")
print(f"📦 Output: {OUTPUT_DIR}")
print()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check model exists
if not os.path.exists(MODEL_DIR):
    raise SystemExit(f"❌ Model not found: {MODEL_DIR}")

# Check converter exists
if not os.path.exists(LLAMA_CPP_CONVERTER):
    raise SystemExit(f"❌ Converter not found: {LLAMA_CPP_CONVERTER}")

print("✓ Model directory found")
print("✓ llama.cpp converter found")
print()

# Export formats to create
formats = [
    ("f16", "16-bit float (original quality)"),
    ("q8_0", "8-bit quantization (very good quality)"),
]

print("🔧 Exporting GGUF formats:")
for fmt, desc in formats:
    print(f"  • {fmt}: {desc}")
print()

# Convert each format
for fmt, desc in formats:
    output_file = os.path.join(OUTPUT_DIR, f"ministral-3-3b-sudoku-{fmt}.gguf")

    print("─" * 70)
    print(f"Converting to {fmt.upper()}...")
    print(f"Output: {output_file}")
    print()

    cmd = [
        "python", LLAMA_CPP_CONVERTER,
        MODEL_DIR,
        "--outfile", output_file,
        "--outtype", fmt,
        "--vocab-only" if fmt == "vocab" else "--no-vocab-only",  # Not actually used
        "--verbose"
    ]

    # Remove the vocab flag (it's not in the actual command)
    cmd = [
        "python", LLAMA_CPP_CONVERTER,
        MODEL_DIR,
        "--outfile", output_file,
        "--outtype", fmt,
        "--verbose"
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )

        if os.path.exists(output_file):
            size_gb = os.path.getsize(output_file) / (1024**3)
            print()
            print(f"✅ Success! File size: {size_gb:.2f} GB")
        else:
            print()
            print(f"⚠️  Warning: Output file not found after conversion")

    except subprocess.CalledProcessError as e:
        print()
        print(f"❌ Conversion failed for {fmt}")
        print(f"Error: {e}")
        continue

    print()

print("=" * 70)
print("📊 Summary")
print("=" * 70)

# List all GGUF files
if os.path.exists(OUTPUT_DIR):
    gguf_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.gguf')])
    if gguf_files:
        print("\n✅ Generated GGUF files:")
        total_size = 0
        for fname in gguf_files:
            fpath = os.path.join(OUTPUT_DIR, fname)
            size_gb = os.path.getsize(fpath) / (1024**3)
            total_size += size_gb
            print(f"  • {fname} ({size_gb:.2f} GB)")
        print(f"\nTotal size: {total_size:.2f} GB")
    else:
        print("\n⚠️  No GGUF files found")
else:
    print("\n❌ Output directory not found")

print()
print("=" * 70)
print("🎮 Usage with llama.cpp")
print("=" * 70)
print()
print("Test with llama-cli:")
print()
print(f"  cd /workspace/llama.cpp")
print(f"  ./llama-cli \\")
print(f"    -m {OUTPUT_DIR}/ministral-3-3b-sudoku-q8_0.gguf \\")
print(f"    -c 4096 \\")
print(f"    -ngl 99 \\")
print(f"    -p 'Create a Sudoku solving strategy...'")
print()
print("Or start a server:")
print()
print(f"  ./llama-server \\")
print(f"    -m {OUTPUT_DIR}/ministral-3-3b-sudoku-q8_0.gguf \\")
print(f"    -c 4096 \\")
print(f"    -ngl 99 \\")
print(f"    --port 8080")
print()
print("=" * 70)
