"""Benchmark inference throughput via OpenAI-compatible API.

Measures: TTFT (Time to First Token), generation speed (tokens/s),
end-to-end latency, and GPU memory via nvidia-smi.
Supports concurrent requests for throughput testing.

Works with any OpenAI-compatible server: vLLM, llama.cpp, Ollama, SGLang.

Start server first:
    # vLLM (WSL2/Linux)
    python -m vllm.entrypoints.openai.api_server \
        --model models/qwen25vl-3b-drivelm-baseline-awq --port 8000

    # llama.cpp (Windows/Linux)
    llama-server -m model.gguf --mmproj mmproj.gguf --port 8000

    # Ollama
    ollama serve  # default port 11434

Usage:
    python scripts/benchmark_throughput.py \
        --api-base http://localhost:8000/v1 \
        --model qwen25vl-3b-drivelm-baseline-awq \
        --concurrency 1,2,4 \
        --num-requests 20 \
        --image-dir data/nuscenes/samples/CAM_FRONT

    # Text-only (no images)
    python scripts/benchmark_throughput.py \
        --api-base http://localhost:8000/v1 --model MODEL --no-image

    # Ollama
    python scripts/benchmark_throughput.py \
        --api-base http://localhost:11434/v1 --model MODEL
"""

import argparse
import asyncio
import base64
import json
import os
import random
import subprocess
import statistics
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Representative DriveLM-style prompts for benchmarking
BENCHMARK_PROMPTS = [
    "What are the important objects in the current scene? Those objects will be considered for the future planning.",
    "What is the moving status of the object directly ahead of the ego vehicle?",
    "Is there a traffic light visible? What is its current status?",
    "What is the safe action for the ego vehicle to take in this situation?",
    "Describe the weather and road conditions visible in this driving scene.",
    "Are there any pedestrians or cyclists near the ego vehicle? What are their intentions?",
    "What lane is the ego vehicle currently in? Should it change lanes?",
    "Identify potential hazards in the current driving scene.",
    "What is the speed and direction of the vehicle in front?",
    "Based on the current scene, what should the ego vehicle's planning decision be?",
]


def encode_image_b64(image_path):
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_test_images(image_dir, n=10):
    """Load n random images from directory as base64 strings."""
    if not image_dir or not os.path.isdir(image_dir):
        return []
    files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return []
    random.shuffle(files)
    images = []
    for f in files[:n]:
        path = os.path.join(image_dir, f)
        images.append(encode_image_b64(path))
    return images


def build_messages(prompt, image_b64=None):
    """Build OpenAI-compatible chat messages."""
    content = []
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def get_gpu_memory_mb():
    """Get current GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            values = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
            return values[0] if values else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


async def benchmark_single(session, url, model, messages, max_tokens):
    """Send one streaming request and measure timing metrics."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }

    t_send = time.perf_counter()
    t_first_token = None
    output_tokens = 0

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return {"error": f"HTTP {resp.status}: {error_text[:200]}"}

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            output_tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except Exception as e:
        return {"error": str(e)}

    t_end = time.perf_counter()

    if t_first_token is None:
        return {"error": "No tokens generated"}

    gen_time = t_end - t_first_token
    return {
        "ttft": round((t_first_token - t_send) * 1000, 1),  # ms
        "latency": round((t_end - t_send) * 1000, 1),  # ms
        "output_tokens": output_tokens,
        "tokens_per_sec": round(output_tokens / gen_time, 1) if gen_time > 0 else 0,
        "gen_time_ms": round(gen_time * 1000, 1),
    }


async def run_benchmark(api_base, model, prompts, images, max_tokens, num_requests, concurrency):
    """Run benchmark with given concurrency level."""
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    url = f"{api_base.rstrip('/')}/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(i):
        async with semaphore:
            prompt = prompts[i % len(prompts)]
            img = images[i % len(images)] if images else None
            messages = build_messages(prompt, img)
            return await benchmark_single(session, url, model, messages, max_tokens)

    gpu_before = get_gpu_memory_mb()

    connector = aiohttp.TCPConnector(limit=concurrency + 2)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        t_start = time.perf_counter()
        tasks = [bounded(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_start

    gpu_after = get_gpu_memory_mb()

    # Filter successful results
    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not ok:
        return {"concurrency": concurrency, "error": "All requests failed", "errors": errors[:3]}

    ttfts = [r["ttft"] for r in ok]
    latencies = [r["latency"] for r in ok]
    tps_list = [r["tokens_per_sec"] for r in ok]
    total_tokens = sum(r["output_tokens"] for r in ok)

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successful": len(ok),
        "failed": len(errors),
        "wall_time_s": round(wall_time, 2),
        "throughput_rps": round(len(ok) / wall_time, 2),
        "total_tokens": total_tokens,
        "aggregate_tps": round(total_tokens / wall_time, 1),
        "ttft_ms": {
            "p50": round(statistics.median(ttfts), 1),
            "p95": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1) if len(ttfts) >= 2 else ttfts[0],
            "mean": round(statistics.mean(ttfts), 1),
        },
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) >= 2 else latencies[0],
            "mean": round(statistics.mean(latencies), 1),
        },
        "tokens_per_sec": {
            "p50": round(statistics.median(tps_list), 1),
            "mean": round(statistics.mean(tps_list), 1),
        },
        "gpu_memory_mb": {
            "before": gpu_before,
            "after": gpu_after,
        },
        "errors_sample": [e["error"] for e in errors[:3]] if errors else [],
    }


def print_results(results_list, model_name):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS: {model_name}")
    print(f"{'=' * 70}")

    for r in results_list:
        c = r["concurrency"]
        if "error" in r:
            print(f"\n  Concurrency={c}: {r['error']}")
            continue

        print(f"\n  Concurrency={c} ({r['successful']}/{r['num_requests']} ok, {r['wall_time_s']}s wall)")
        print(f"  ┌─────────────────┬──────────┬──────────┬──────────┐")
        print(f"  │ Metric          │   P50    │   P95    │   Mean   │")
        print(f"  ├─────────────────┼──────────┼──────────┼──────────┤")
        print(f"  │ TTFT (ms)       │ {r['ttft_ms']['p50']:>7.1f}  │ {r['ttft_ms']['p95']:>7.1f}  │ {r['ttft_ms']['mean']:>7.1f}  │")
        print(f"  │ Latency (ms)    │ {r['latency_ms']['p50']:>7.1f}  │ {r['latency_ms']['p95']:>7.1f}  │ {r['latency_ms']['mean']:>7.1f}  │")
        print(f"  │ Tokens/s        │ {r['tokens_per_sec']['p50']:>7.1f}  │    -     │ {r['tokens_per_sec']['mean']:>7.1f}  │")
        print(f"  └─────────────────┴──────────┴──────────┴──────────┘")
        print(f"  Throughput: {r['throughput_rps']:.2f} req/s | {r['aggregate_tps']:.1f} tokens/s aggregate")
        if r["gpu_memory_mb"]["after"]:
            print(f"  GPU memory: {r['gpu_memory_mb']['after']} MB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference throughput via OpenAI-compatible API")
    parser.add_argument("--api-base", required=True, help="API base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--model", required=True, help="Model name as registered in the server")
    parser.add_argument("--concurrency", default="1,2,4", help="Comma-separated concurrency levels (default: 1,2,4)")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of requests per concurrency level")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
    parser.add_argument("--image-dir", default=None, help="Directory with test images (default: data/nuscenes/samples/CAM_FRONT)")
    parser.add_argument("--no-image", action="store_true", help="Text-only benchmark (no images)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load images
    images = []
    if not args.no_image:
        image_dir = args.image_dir or os.path.join(BASE_DIR, "data", "nuscenes", "samples", "CAM_FRONT")
        if os.path.isdir(image_dir):
            print(f"Loading test images from {image_dir}...")
            images = load_test_images(image_dir, n=min(20, args.num_requests))
            print(f"  {len(images)} images loaded")
        else:
            print(f"WARNING: Image dir not found: {image_dir}")
            print("  Running text-only benchmark. Use --image-dir or --no-image to suppress.")

    mode = "multimodal" if images else "text-only"
    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]

    print(f"\nBenchmark config:")
    print(f"  API:         {args.api_base}")
    print(f"  Model:       {args.model}")
    print(f"  Mode:        {mode} ({len(images)} images)")
    print(f"  Requests:    {args.num_requests} per level")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Max tokens:  {args.max_tokens}")

    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run benchmarks
    all_results = []
    for c in concurrency_levels:
        print(f"\n--- Running concurrency={c} ---")
        result = asyncio.run(
            run_benchmark(args.api_base, args.model, BENCHMARK_PROMPTS, images,
                          args.max_tokens, args.num_requests, c)
        )
        all_results.append(result)

    # Display
    print_results(all_results, args.model)

    # Save
    output_path = args.output or os.path.join(BASE_DIR, "benchmark_results.json")
    full_report = {
        "model": args.model,
        "api_base": args.api_base,
        "mode": mode,
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
