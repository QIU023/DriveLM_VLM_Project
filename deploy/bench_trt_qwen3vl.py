"""TRT-LLM 1.3.0rc15 latency benchmark on B.5'' (Qwen3-VL-4B) ckpt.

Measures: TTFT, per-token decode, full 14-token trajectory generation, throughput.
Compares to HF bf16 baseline at deploy/trt_bench/B5prime_hf_bf16.json (different model
since HF baseline was on Qwen2.5-VL 3-cam, this is Qwen3-VL 1-cam; trade is
"deployable" for "1-cam vs 3-cam" — different model class).
"""
import json
import time
from pathlib import Path


def percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def main():
    ckpt = "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"
    out_path = "/workspace/DriveLM_VLM_Project/deploy/trt_bench/B5pp_trt_qwen3vl_bf16.json"

    print(f"[bench] loading {ckpt} via TRT-LLM ...")
    from tensorrt_llm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm = LLM(model=ckpt, dtype="bfloat16", tensor_parallel_size=1)
    t_load = time.perf_counter() - t0
    print(f"[bench] loaded in {t_load:.1f}s")

    # Fixed prompt (text-only, mimics planning prompt prefix structure)
    prompt = (
        "You are a self-driving system. Given the current scene context, "
        "predict the next 6 ego waypoints as <traj_start> bin tokens. "
        "Current scene: straight road, 12 m/s, no obstacles. <traj_start>"
    )
    N_WARMUP = 3
    N_RUNS = 20
    MAX_NEW = 14  # trajectory token sequence length

    sp_one = SamplingParams(max_tokens=1, temperature=0.0)
    sp_full = SamplingParams(max_tokens=MAX_NEW, temperature=0.0)

    print(f"[bench] warmup {N_WARMUP} runs ...")
    for _ in range(N_WARMUP):
        _ = llm.generate([prompt], sp_full)

    # 1. TTFT (max_new=1)
    ttft = []
    for _ in range(N_RUNS):
        t = time.perf_counter()
        _ = llm.generate([prompt], sp_one)
        ttft.append(time.perf_counter() - t)

    # 2. Full 14-token generation
    full = []
    for _ in range(N_RUNS):
        t = time.perf_counter()
        _ = llm.generate([prompt], sp_full)
        full.append(time.perf_counter() - t)

    # 3. Per-token decode = (full - ttft) / (MAX_NEW - 1)
    decode = [(f - t) / (MAX_NEW - 1) for f, t in zip(full, ttft)]
    throughput = [MAX_NEW / f for f in full]

    results = {
        "ckpt": ckpt,
        "backend": "TRT-LLM 1.3.0rc15 PyTorch backend",
        "model_type": "qwen3_vl",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "n_warmup": N_WARMUP,
        "n_runs": N_RUNS,
        "max_new_tokens": MAX_NEW,
        "load_seconds": t_load,
        "TTFT_ms": {
            "mean": 1000 * sum(ttft) / len(ttft),
            "p50": 1000 * percentile(ttft, 50),
            "p99": 1000 * percentile(ttft, 99),
        },
        "per_token_decode_ms": {
            "mean": 1000 * sum(decode) / len(decode),
            "p50": 1000 * percentile(decode, 50),
            "p99": 1000 * percentile(decode, 99),
        },
        "full_traj_ms": {
            "mean": 1000 * sum(full) / len(full),
            "p50": 1000 * percentile(full, 50),
            "p99": 1000 * percentile(full, 99),
        },
        "throughput_toks_per_s": {
            "mean": sum(throughput) / len(throughput),
            "p50": percentile(throughput, 50),
        },
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bench] saved → {out_path}")
    print(f"  TTFT    mean: {results['TTFT_ms']['mean']:.1f} ms (p50 {results['TTFT_ms']['p50']:.1f} / p99 {results['TTFT_ms']['p99']:.1f})")
    print(f"  decode  mean: {results['per_token_decode_ms']['mean']:.2f} ms / token")
    print(f"  full    mean: {results['full_traj_ms']['mean']:.1f} ms / {MAX_NEW} tokens")
    print(f"  thrpt   mean: {results['throughput_toks_per_s']['mean']:.1f} tok/s")


if __name__ == "__main__":
    main()
