from safetensors import safe_open

tensors_ok = True
with safe_open("grpo_saved_lora/adapter_model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        if n_zeros.item() == 1.0:  # All zeros
            print(f"❌ {key} is all zeros!")
            tensors_ok = False
        else:
            zero_pct = n_zeros.item() * 100
            print(f"✓ {key}: {zero_pct:.1f}% zeros")

print("\n" + ("✅ All tensors OK!" if tensors_ok else "❌ Some tensors are broken"))