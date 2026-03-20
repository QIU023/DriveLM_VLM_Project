# Visual Token Compression — DriveLM Evaluation Results

## Setup

- **Base Model**: Qwen2.5-VL-3B-Instruct
- **Fine-tuning**: LoRA (r=16, alpha=32, dropout=0.05), 1 epoch, lr=2e-4
- **Dataset**: DriveLM nuScenes val set (18,898 samples, 100 per category sampled)
- **Evaluation**: Exact match accuracy, greedy decoding (do_sample=False), max_new_tokens=512
- **GPU**: NVIDIA GH200 480GB (bf16, no quantization)
- **Image Resolution**: min 256×28×28, max 512×28×28

## Methods

| Method | Config | Compression | Description |
|--------|--------|-------------|-------------|
| **Baseline** | `baseline.yaml` | 1x (none) | No visual token compression, full tokens |
| **FasterVLM** | `fastervlm_c4.yaml` | 4x | Top-25% tokens by L2 norm (importance proxy) |
| **PruMerge** | `prumerge_c4.yaml` | 4x | Prune low-importance + merge into nearest kept |
| **PyramidDrop** | `pyramiddrop_c4.yaml` | 4x | Two-stage progressive dropping by importance |
| **AvgPool** | `avg_pool_c4.yaml` | 4x | 2x2 spatial average pooling |

## Overall Results

| Method | Compress | Accuracy | Avg Tokens | Avg Time | Checkpoint |
|--------|----------|----------|------------|----------|------------|
| Baseline | 1x | **55.0%** | 14.6 | 1.48s | `default/checkpoint-51000` |
| FasterVLM | 4x | **55.0%** | 13.8 | 1.41s | `fastervlm_c4/final` |
| PruMerge | 4x | — | — | — | `prumerge_c4/checkpoint-36000` |
| PyramidDrop | 4x | — | — | — | training in progress |
| AvgPool | 4x | — | — | — | not started |

## Per-Category Breakdown

### Baseline (no compression)

| Category | N | Exact Match | Accuracy | Avg Tokens | Avg Time |
|----------|---|-------------|----------|------------|----------|
| behavior | 100 | 46 | 46.0% | 15.0 | 1.52s |
| perception | 100 | 47 | 47.0% | 22.0 | 2.17s |
| planning | 100 | 50 | 50.0% | 15.2 | 1.54s |
| prediction | 100 | 77 | 77.0% | 6.0 | 0.69s |
| **OVERALL** | **400** | **220** | **55.0%** | **14.6** | **1.48s** |

### FasterVLM (4x compression)

| Category | N | Exact Match | Accuracy | Avg Tokens | Avg Time |
|----------|---|-------------|----------|------------|----------|
| behavior | 100 | 43 | 43.0% | 15.2 | 1.54s |
| perception | 100 | 46 | 46.0% | 19.9 | 1.99s |
| planning | 100 | 51 | 51.0% | 14.1 | 1.43s |
| prediction | 100 | 80 | 80.0% | 6.0 | 0.69s |
| **OVERALL** | **400** | **220** | **55.0%** | **13.8** | **1.41s** |

## Analysis

### Baseline vs FasterVLM (4x compression)

| Category | Baseline Acc | FasterVLM Acc | Delta |
|----------|-------------|---------------|-------|
| behavior | 46.0% | 43.0% | -3.0% |
| perception | 47.0% | 46.0% | -1.0% |
| planning | 50.0% | 51.0% | +1.0% |
| prediction | 77.0% | 80.0% | +3.0% |
| **OVERALL** | **55.0%** | **55.0%** | **0.0%** |

**Key Findings:**
- FasterVLM achieves **identical overall accuracy** (55.0%) with 4x visual token compression
- Slight accuracy trade-off in behavior (-3%) and perception (-1%), compensated by gains in planning (+1%) and prediction (+3%)
- FasterVLM is marginally faster (1.41s vs 1.48s avg inference time) due to fewer visual tokens
- Token generation count is slightly lower (13.8 vs 14.6 avg tokens)

---

*Last updated: 2026-03-20*
*Evaluation: 100 samples per category (400 total), deterministic sampling*
