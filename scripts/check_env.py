"""Check local environment for DriveLM inference pipeline."""
import sys

def check_pkg(module_name, display_name=None):
    name = display_name or module_name
    try:
        m = __import__(module_name)
        ver = getattr(m, '__version__', 'installed')
        print(f"  [OK] {name}: {ver}")
        return True
    except ImportError:
        print(f"  [X]  {name}: NOT INSTALLED")
        return False

print("=" * 50)
print("DriveLM Local Environment Check")
print("=" * 50)

print(f"\nPython: {sys.version}")

# Core
print("\n--- Core ---")
check_pkg("torch", "PyTorch")
check_pkg("transformers")
check_pkg("peft")
check_pkg("accelerate")

# Quantization
print("\n--- Quantization ---")
check_pkg("bitsandbytes")
check_pkg("auto_gptq", "auto-gptq")
check_pkg("awq", "autoawq")

# VLM specific
print("\n--- VLM ---")
check_pkg("qwen_vl_utils", "qwen-vl-utils")
check_pkg("PIL", "Pillow")

# Serving
print("\n--- Serving ---")
check_pkg("vllm")
check_pkg("sglang")

# Profiling
print("\n--- Profiling ---")
check_pkg("flash_attn", "flash-attention")

# Config
print("\n--- Config ---")
check_pkg("yaml", "pyyaml")

print("\n" + "=" * 50)
