"""Quantize a merged HuggingFace model for efficient inference deployment.

Supports three quantization methods:
  - AWQ (W4A16): autoawq calibration + quantization → for vLLM serving
  - GPTQ (W4A16): auto-gptq calibration + quantization → for vLLM serving
  - GGUF (Q4_K_M): llama.cpp convert + quantize → for llama.cpp / Ollama

Calibration data is extracted from DriveLM train_mini.json (text-only, no images).

Dependencies:
  AWQ:  pip install autoawq
  GPTQ: pip install auto-gptq optimum
  GGUF: git clone https://github.com/ggerganov/llama.cpp && pip install gguf

Usage:
    python scripts/quantize_model.py awq  --input models/qwen25vl-3b-drivelm-baseline-merged
    python scripts/quantize_model.py gptq --input models/qwen25vl-3b-drivelm-baseline-merged
    python scripts/quantize_model.py gguf --input models/qwen25vl-3b-drivelm-baseline-merged \
                                          --llama-cpp /path/to/llama.cpp

Output:
    models/qwen25vl-3b-drivelm-baseline-{awq,gptq,gguf}/
"""

import argparse
import json
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_calib_texts(n_samples=128, max_length=512):
    """Load calibration texts from DriveLM data (text-only, no images)."""
    data_path = os.path.join(BASE_DIR, "data_processed", "train_mini.json")
    if not os.path.exists(data_path):
        # Fallback to val.json
        data_path = os.path.join(BASE_DIR, "data_processed", "val.json")
    if not os.path.exists(data_path):
        print("WARNING: No calibration data found, using default c4 dataset")
        return None

    print(f"Loading calibration data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for item in data[:n_samples]:
        parts = []
        for msg in item.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for p in content:
                    if p.get("type") == "text":
                        parts.append(p["text"])
        text = "\n".join(parts)
        if text.strip():
            texts.append(text[:max_length * 4])  # rough char limit

    print(f"  {len(texts)} calibration samples loaded")
    return texts


def resolve_output(input_path, output_path, method):
    """Auto-generate output path if not specified."""
    if output_path:
        return os.path.join(BASE_DIR, output_path) if not os.path.isabs(output_path) else output_path
    base = os.path.basename(input_path.rstrip("/\\"))
    name = base.replace("-merged", f"-{method}")
    return os.path.join(os.path.dirname(input_path), name)


# ──────────────────────────────────────────────────────────────
# AWQ
# ──────────────────────────────────────────────────────────────
def quantize_awq(input_path, output_path, bits=4, group_size=128):
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("ERROR: autoawq not installed.")
        print("  Install: pip install autoawq")
        sys.exit(1)
    from transformers import AutoTokenizer

    print(f"\n[AWQ W{bits}A16] {input_path}")
    print(f"  -> {output_path}")

    t0 = time.time()

    # Load model
    print("  Loading model...")
    model = AutoAWQForCausalLM.from_pretrained(input_path, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)

    # Calibration data
    calib_texts = load_calib_texts()
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }

    # Quantize
    print(f"  Quantizing (bits={bits}, group_size={group_size})...")
    if calib_texts:
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
    else:
        model.quantize(tokenizer, quant_config=quant_config)

    # Save
    print(f"  Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    elapsed = time.time() - t0
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".bin"))
    )
    print(f"  Done! {total_size / 1024**3:.2f} GB, {elapsed:.0f}s")
    return {"method": "awq", "bits": bits, "size_gb": round(total_size / 1024**3, 2), "time_s": round(elapsed)}


# ──────────────────────────────────────────────────────────────
# GPTQ
# ──────────────────────────────────────────────────────────────
def quantize_gptq(input_path, output_path, bits=4, group_size=128):
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("ERROR: auto-gptq not installed.")
        print("  Install: pip install auto-gptq")
        sys.exit(1)
    from transformers import AutoTokenizer

    print(f"\n[GPTQ W{bits}A16] {input_path}")
    print(f"  -> {output_path}")

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)

    # Build calibration examples
    calib_texts = load_calib_texts()
    examples = []
    if calib_texts:
        for text in calib_texts[:128]:
            enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            examples.append({"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0]})

    # Quantize config
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )

    # Load and quantize
    print("  Loading model...")
    model = AutoGPTQForCausalLM.from_pretrained(input_path, quantize_config, trust_remote_code=True)
    print(f"  Quantizing (bits={bits}, group_size={group_size})...")
    model.quantize(examples if examples else None)

    # Save
    print(f"  Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    elapsed = time.time() - t0
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".bin"))
    )
    print(f"  Done! {total_size / 1024**3:.2f} GB, {elapsed:.0f}s")
    return {"method": "gptq", "bits": bits, "size_gb": round(total_size / 1024**3, 2), "time_s": round(elapsed)}


# ──────────────────────────────────────────────────────────────
# GGUF
# ──────────────────────────────────────────────────────────────
def convert_gguf(input_path, output_path, quant_type="Q4_K_M", llama_cpp_path=None):
    """Convert HF model → GGUF via llama.cpp's convert script."""
    # Find llama.cpp
    if llama_cpp_path is None:
        for candidate in [
            os.path.join(BASE_DIR, "llama.cpp"),
            os.path.expanduser("~/llama.cpp"),
            os.path.expanduser("~\\llama.cpp"),
        ]:
            if os.path.isdir(candidate):
                llama_cpp_path = candidate
                break

    if not llama_cpp_path or not os.path.isdir(llama_cpp_path):
        print("ERROR: llama.cpp not found.")
        print("  Clone: git clone https://github.com/ggerganov/llama.cpp")
        print("  Then pass: --llama-cpp /path/to/llama.cpp")
        sys.exit(1)

    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        print(f"ERROR: convert_hf_to_gguf.py not found in {llama_cpp_path}")
        sys.exit(1)

    os.makedirs(output_path, exist_ok=True)
    model_name = os.path.basename(input_path.rstrip("/\\")).replace("-merged", "")

    print(f"\n[GGUF {quant_type}] {input_path}")
    print(f"  -> {output_path}")

    t0 = time.time()

    # Step 1: Convert to f16 GGUF
    f16_gguf = os.path.join(output_path, f"{model_name}-f16.gguf")
    print(f"  [1/2] Converting HF -> GGUF f16...")
    cmd_convert = [
        sys.executable, convert_script,
        input_path,
        "--outfile", f16_gguf,
        "--outtype", "f16",
    ]
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR in convert: {result.stderr[-500:]}")
        return None
    print(f"         {os.path.getsize(f16_gguf) / 1024**3:.2f} GB")

    # Step 2: Quantize to target type
    quant_gguf = os.path.join(output_path, f"{model_name}-{quant_type.lower()}.gguf")
    print(f"  [2/2] Quantizing -> {quant_type}...")

    # Find llama-quantize binary
    quantize_bin = None
    for name in ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]:
        for subdir in ["build/bin", "build", "bin", ""]:
            candidate = os.path.join(llama_cpp_path, subdir, name)
            if os.path.exists(candidate):
                quantize_bin = candidate
                break
        if quantize_bin:
            break

    if not quantize_bin:
        print("  WARNING: llama-quantize binary not found, keeping f16 GGUF only.")
        print(f"  Build llama.cpp first: cd {llama_cpp_path} && cmake -B build && cmake --build build")
        quant_gguf = f16_gguf
    else:
        cmd_quant = [quantize_bin, f16_gguf, quant_gguf, quant_type]
        result = subprocess.run(cmd_quant, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR in quantize: {result.stderr[-500:]}")
            return None
        # Remove f16 intermediate
        if os.path.exists(quant_gguf) and quant_gguf != f16_gguf:
            os.remove(f16_gguf)

    elapsed = time.time() - t0
    final_size = os.path.getsize(quant_gguf)
    print(f"  Done! {final_size / 1024**3:.2f} GB, {elapsed:.0f}s")
    print(f"  File: {quant_gguf}")
    return {"method": "gguf", "quant_type": quant_type, "size_gb": round(final_size / 1024**3, 2),
            "time_s": round(elapsed), "file": quant_gguf}


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Quantize merged model for deployment")
    sub = parser.add_subparsers(dest="method", required=True)

    # AWQ
    p_awq = sub.add_parser("awq", help="AWQ W4A16 quantization (for vLLM)")
    p_awq.add_argument("--input", required=True, help="Merged model directory")
    p_awq.add_argument("--output", default=None, help="Output directory (auto-generated if omitted)")
    p_awq.add_argument("--bits", type=int, default=4)
    p_awq.add_argument("--group-size", type=int, default=128)

    # GPTQ
    p_gptq = sub.add_parser("gptq", help="GPTQ W4A16 quantization (for vLLM)")
    p_gptq.add_argument("--input", required=True, help="Merged model directory")
    p_gptq.add_argument("--output", default=None, help="Output directory (auto-generated if omitted)")
    p_gptq.add_argument("--bits", type=int, default=4)
    p_gptq.add_argument("--group-size", type=int, default=128)

    # GGUF
    p_gguf = sub.add_parser("gguf", help="GGUF quantization (for llama.cpp / Ollama)")
    p_gguf.add_argument("--input", required=True, help="Merged model directory")
    p_gguf.add_argument("--output", default=None, help="Output directory (auto-generated if omitted)")
    p_gguf.add_argument("--type", default="Q4_K_M", dest="quant_type",
                        help="GGUF quant type (default: Q4_K_M)")
    p_gguf.add_argument("--llama-cpp", default=None, dest="llama_cpp_path",
                        help="Path to llama.cpp clone")

    args = parser.parse_args()

    input_path = os.path.join(BASE_DIR, args.input) if not os.path.isabs(args.input) else args.input
    output_path = resolve_output(input_path, args.output, args.method)

    if not os.path.isdir(input_path):
        print(f"ERROR: Input not found: {input_path}")
        sys.exit(1)

    if args.method == "awq":
        result = quantize_awq(input_path, output_path, args.bits, args.group_size)
    elif args.method == "gptq":
        result = quantize_gptq(input_path, output_path, args.bits, args.group_size)
    elif args.method == "gguf":
        result = convert_gguf(input_path, output_path, args.quant_type, args.llama_cpp_path)

    if result:
        meta_path = os.path.join(output_path, "quant_info.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
