"""TTFT + decode-throughput benchmark for the TRT-LLM VLA engine.

The MEASUREMENT logic here is version-independent; only the `generate_step`
wiring depends on your TRT-LLM version (Executor API vs the example run.py).
Fill in `make_runner()` per the example you located in README §1, then run:

    python deploy/benchmark.py --engine-dir /models/engine_5090 \
        --image sample.jpg --prompt "..." --warmup 3 --iters 20

Reports:
  * TTFT (time-to-first-token): prefill latency — dominated by vision tokens +
    context length. This is what the visual-token compression work moves.
  * decode throughput (tok/s): steady-state autoregressive rate — memory-
    bandwidth bound at batch=1 (the automotive regime; see README §5).
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List


def make_runner(engine_dir: str):
    """Return a callable that runs ONE generate and yields tokens one at a time.

    VERIFY / WIRE per your TRT-LLM version. Two common options:

      (a) High-level LLM API:
            from tensorrt_llm import LLM, SamplingParams
            llm = LLM(model=engine_dir)
            # multimodal input wiring is version-specific
            for out in llm.generate_async(..., streaming=True): yield out

      (b) Example run.py / Executor: import the example's runner module and
          drive it token-by-token.

    Must yield once per generated token so TTFT = time to the FIRST yield.
    """
    raise NotImplementedError(
        "Wire make_runner() to your TRT-LLM version's multimodal generate "
        "(see README §1 to locate the example), yielding one token per step."
    )


def time_one(gen_step: Callable[[], "iterator"], max_new_tokens: int):
    """Returns (ttft_s, decode_tok_per_s, n_tokens)."""
    t0 = time.perf_counter()
    ttft = None
    n = 0
    t_first = None
    for _tok in gen_step():
        now = time.perf_counter()
        if ttft is None:
            ttft = now - t0          # prefill -> first token
            t_first = now
        n += 1
        if n >= max_new_tokens:
            break
    t_end = time.perf_counter()
    decode_s = max(t_end - (t_first or t_end), 1e-9)
    decode_tps = (n - 1) / decode_s if n > 1 else 0.0
    return ttft or 0.0, decode_tps, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="Describe the driving scene.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    runner = make_runner(args.engine_dir)  # callable -> token iterator

    def gen_step():
        return runner(image=args.image, prompt=args.prompt,
                      max_new_tokens=args.max_new_tokens)

    for _ in range(args.warmup):
        time_one(gen_step, args.max_new_tokens)

    ttfts: List[float] = []
    tpses: List[float] = []
    for _ in range(args.iters):
        ttft, tps, _n = time_one(gen_step, args.max_new_tokens)
        ttfts.append(ttft)
        tpses.append(tps)

    print(f"iters={args.iters}  max_new_tokens={args.max_new_tokens}")
    print(f"TTFT     : mean {1e3*statistics.mean(ttfts):.1f} ms  "
          f"p50 {1e3*statistics.median(ttfts):.1f} ms")
    print(f"decode   : mean {statistics.mean(tpses):.1f} tok/s  "
          f"p50 {statistics.median(tpses):.1f} tok/s")


if __name__ == "__main__":
    main()
