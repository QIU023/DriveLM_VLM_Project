"""Compare quantization results from multiple eval_results JSON files.

Usage:
  python scripts/compare_quant.py eval_results_ckpt46000_bf16.json eval_results_checkpoint-46000_max100.json [...]
"""
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_result(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        # Auto-discover eval_results_*.json in project root
        files = sorted(
            f for f in os.listdir(BASE_DIR)
            if f.startswith("eval_results_") and f.endswith(".json")
        )
        if not files:
            print("No eval_results_*.json files found. Pass paths as arguments.")
            return
        paths = [os.path.join(BASE_DIR, f) for f in files]
    else:
        paths = sys.argv[1:]

    results = []
    for p in paths:
        full = os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p
        if not os.path.exists(full):
            print(f"SKIP (not found): {p}")
            continue
        data = load_result(full)
        results.append((os.path.basename(full), data))

    if not results:
        print("No results to compare.")
        return

    # Collect all categories
    all_cats = set()
    for _, data in results:
        all_cats.update(k for k in data["summary"] if k != "overall")
    cats = sorted(all_cats) + ["overall"]

    # Header
    print("\n" + "=" * 90)
    print("QUANTIZATION COMPARISON")
    print("=" * 90)

    # Per-category tables
    for cat in cats:
        print(f"\n--- {cat.upper()} ---")
        print(f"  {'Config':<42s} {'N':>5} {'Exact':>6} {'Acc%':>7} {'AvgTok':>7} {'AvgTime':>8}")
        print(f"  {'-'*42} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
        for fname, data in results:
            tag = data.get("tag", fname)
            s = data["summary"].get(cat)
            if not s:
                continue
            label = f"{tag} | {fname.replace('eval_results_','').replace('.json','')}"
            print(f"  {label:<42s} {s['n']:>5} {s['exact_match']:>6} {s['accuracy']:>6.1f}% "
                  f"{s['avg_tokens']:>7.1f} {s['avg_time_s']:>7.2f}s")

    # Summary table for copy-paste
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (for project_goal.md)")
    print("=" * 90)
    print(f"\n| {'Config':<35s} | {'Accuracy':>8s} | {'Avg Tokens':>10s} | {'Avg Time':>8s} | {'GPU Mem':>7s} |")
    print(f"|{'-'*37}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*9}|")
    for fname, data in results:
        tag = data.get("tag", "?")
        s = data["summary"]["overall"]
        # GPU memory not stored in results, mark as N/A
        print(f"| {tag:<35s} | {s['accuracy']:>7.1f}% | {s['avg_tokens']:>10.1f} | {s['avg_time_s']:>7.2f}s | {'N/A':>7s} |")

    print()


if __name__ == "__main__":
    main()
