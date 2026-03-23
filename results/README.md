# Visual Token Compression — DriveLM Evaluation Results

## Setup

- **Base Model**: Qwen2.5-VL-3B-Instruct
- **Fine-tuning**: LoRA (r=16, alpha=32, dropout=0.05), 1 epoch, lr=2e-4
- **Dataset**: DriveLM nuScenes val set (18,898 samples, full evaluation)
- **Evaluation**: Exact match accuracy, greedy decoding (do_sample=False), max_new_tokens=512
- **GPU**: NVIDIA GH200 480GB (bf16, no quantization)
- **Image Resolution**: min 256×28×28, max 512×28×28

## Methods

| Method | Config | Compression | Description |
|--------|--------|-------------|-------------|
| **Baseline** | `baseline.yaml` | 1x (none) | No visual token compression, full 480 tokens |
| **FasterVLM** | `fastervlm_c4.yaml` | 4x | Top-25% tokens by L2 norm importance |
| **PruMerge** | `prumerge_c4.yaml` | 4x | Prune low-importance + merge into nearest kept |
| **PyramidDrop** | `pyramiddrop_c4.yaml` | 4x | Two-stage progressive dropping by importance |
| AvgPool | `avg_pool_c4.yaml` | 4x | 2x2 spatial average pooling (not yet trained) |

## Overall Accuracy (Full val set, N=18,898)

| Method | Compress | Visual Tokens | Accuracy | Avg Tokens | Avg Time | Eval Time |
|--------|:--------:|:------------:|:--------:|:----------:|:--------:|:---------:|
| **Baseline** | 1x | 480 | **56.7%** | 13.4 | 1.40s | 441 min |
| **FasterVLM** | 4x | 120 | **56.9%** | 13.2 | 1.38s | 436 min |
| **PruMerge** | 4x | 120 | **57.4%** | 13.4 | 1.40s | 441 min |
| **PyramidDrop** | 4x | 120 | **57.3%** | 13.2 | 1.31s | 414 min |

## Per-Category Breakdown (Full eval)

| Category | N | Baseline | FasterVLM | PruMerge | PyramidDrop |
|----------|----:|:--------:|:---------:|:--------:|:-----------:|
| behavior | 187 | 44.9% | 44.4% | **51.9%** | 48.1% |
| perception | 8,012 | 47.3% | **48.0%** | 48.4% | 48.5% |
| planning | 4,430 | 48.7% | 48.7% | **49.3%** | 49.1% |
| prediction | 6,269 | 74.6% | 74.4% | **74.8%** | 74.6% |
| **OVERALL** | **18,898** | 56.7% | 56.9% | **57.4%** | 57.3% |

## TTFT / Latency Benchmark (GH200, bf16, 30 image samples)

> Measures visual encoder → compression → prefill → first token timing with real images.

| Method | Visual Tokens | Vision (ms) | Compress (ms) | Prefill (ms) | **TTFT P50 (ms)** | GPU (GB) |
|--------|:------------:|:-----------:|:-------------:|:------------:|:------------------:|:--------:|
| Baseline | 480→480 | 107 | 0 | 98 | **205** | 7.28 |
| FasterVLM | 480→120 | 106 | 1 | 98 | **205** | 7.28 |
| PruMerge | 480→120 | 107 | 125 | 99 | **327** | 7.28 |
| PyramidDrop | 480→120 | 110 | 1 | 101 | **211** | 7.28 |

## FasterVLM Compression Scaling (accuracy vs ratio)

> Using baseline LoRA + FasterVLM compression at inference time (no re-training).

| Ratio | Visual Tokens | Accuracy | Delta vs 1x |
|:-----:|:------------:|:--------:|:-----------:|
| 1x | 480 | 51.2% | — |
| 2x | 240 | 51.0% | -0.2% |
| 4x | 120 | 51.0% | -0.2% |
| 8x | 60 | 50.7% | -0.5% |
| 16x | 30 | 48.8% | -2.4% |

## 4070Ti Deployment Benchmark (GGUF Q4_K_M + llama.cpp, text-only)

> LLM backbone throughput — no vision encoder (GGUF 不含 ViT)。token 压缩差异不体现在此。

| Model | Concurrency | TTFT P50 (ms) | Tokens/s P50 | Aggregate TPS | GPU (MB) |
|-------|:-----------:|:-------------:|:------------:|:-------------:|:--------:|
| baseline | 1 | 142 | 170 | 72 | 5,264 |
| baseline | 4 | 181 | 121 | 196 | 5,234 |
| fastervlm | 1 | 144 | 173 | 78 | 5,326 |
| fastervlm | 4 | 165 | 124 | 202 | 5,308 |
| prumerge | 1 | 141 | 173 | 74 | 5,238 |
| prumerge | 4 | 182 | 114 | 197 | 5,219 |
| pyramiddrop | 1 | 141 | 171 | 72 | 4,897 |
| pyramiddrop | 4 | 196 | 120 | 185 | 4,900 |

## Analysis

### Key Findings

1. **4x compression = zero accuracy loss** — All three methods (FasterVLM, PruMerge, PyramidDrop) match or **exceed** baseline accuracy at 4x compression. PruMerge leads at 57.4% (+0.7% over baseline).

2. **PruMerge best accuracy, FasterVLM/PyramidDrop best latency** — PruMerge's merge step (125ms) adds overhead that negates its TTFT advantage. FasterVLM and PyramidDrop add <2ms compression overhead.

3. **Prefill time unchanged at 480 tokens** — At this resolution (480 visual tokens), the 4x compression from 480→120 tokens saves ~360 token slots but prefill is already fast (~98ms). The benefit scales with higher resolution where visual tokens can reach 1000+.

4. **Robust to extreme compression** — FasterVLM scaling shows only 2.4% accuracy drop even at 16x (480→30 tokens), demonstrating high redundancy in visual tokens for DriveLM driving QA.

5. **4070Ti deployment viable** — GGUF Q4_K_M (1.8GB) achieves 170 tok/s single-request on consumer GPU with ~5GB VRAM footprint. Model fits comfortably in 12GB with room for concurrent requests.

### Interview Talking Points

- "4x token 压缩在 DriveLM 精度不降反升（+0.7%），说明原始 visual token 存在大量冗余"
- "FasterVLM 压缩只需 1ms，PruMerge 精度最高但压缩开销 125ms，存在 accuracy-latency tradeoff"
- "从 bf16 训练 → Q4_K_M 量化 → llama.cpp 部署，模型从 7GB 压缩到 1.8GB，在消费级 4070Ti 上 170 tok/s"
- "FasterVLM 16x 极端压缩（480→30 tokens）只掉 2.4% 精度，说明 driving QA 任务视觉信息高度冗余"

---

*Last updated: 2026-03-21*
*Full evaluation: 18,898 samples, TTFT benchmark: 30 image samples with warmup*
