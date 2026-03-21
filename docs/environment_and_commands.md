# Environment & Command Reference

## Hardware

- **GPU**: NVIDIA GH200 480GB (96GB HBM3 GPU memory)
- **Arch**: aarch64 (ARM)
- **OS**: Linux 6.8.0-1046-nvidia-64k

## Conda Environment

```bash
# Activate
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate qwen25vl

# Or (if conda init already ran)
conda activate qwen25vl
```

**Python 3.10** | Key packages:

| Package | Version |
|---------|---------|
| torch | 2.10.0+cu128 |
| transformers | 5.3.0 |
| peft | 0.18.1 |
| accelerate | 1.13.0 |
| bitsandbytes | 0.49.2 |
| pillow | 12.0.0 |
| matplotlib | 3.10.8 |
| numpy | 2.2.6 |
| safetensors | 0.7.0 |
| pyyaml | 6.0.3 |
| triton | 3.6.0 |
| wandb | 0.25.1 |
| qwen-vl-utils | 0.0.14 |
| torchvision | 0.25.0+cu128 |
| tokenizers | 0.22.2 |

<details>
<summary>Full conda list</summary>

```
accelerate                  1.13.0
annotated-types             0.7.0
av                          17.0.0
bitsandbytes                0.49.2
certifi                     2026.2.25
contourpy                   1.3.2
cuda-bindings               12.9.4
cuda-pathfinder             1.2.2
cycler                      0.12.1
filelock                    3.20.0
fonttools                   4.62.1
fsspec                      2025.12.0
gitdb                       4.0.12
gitpython                   3.1.46
jinja2                      3.1.6
kiwisolver                  1.5.0
markupsafe                  3.0.2
matplotlib                  3.10.8
mpmath                      1.3.0
networkx                    3.4.2
numpy                       2.2.6
nvidia-cublas-cu12          12.8.4.1
nvidia-cuda-cupti-cu12      12.8.90
nvidia-cuda-nvrtc-cu12      12.8.93
nvidia-cuda-runtime-cu12    12.8.90
nvidia-cudnn-cu12           9.10.2.21
nvidia-cufft-cu12           11.3.3.83
nvidia-cufile-cu12          1.13.1.3
nvidia-curand-cu12          10.3.9.90
nvidia-cusolver-cu12        11.7.3.90
nvidia-cusparse-cu12        12.5.8.93
nvidia-cusparselt-cu12      0.7.1
nvidia-nccl-cu12            2.27.5
nvidia-nvjitlink-cu12       12.8.93
nvidia-nvshmem-cu12         3.4.5
nvidia-nvtx-cu12            12.8.90
packaging                   25.0
peft                        0.18.1
pillow                      12.0.0
protobuf                    6.33.5
psutil                      7.2.2
pydantic                    2.12.5
pydantic-core               2.41.5
pyparsing                   3.3.2
python                      3.10.20
python-dateutil             2.9.0.post0
pyyaml                      6.0.3
qwen-vl-utils               0.0.14
regex                       2026.2.28
requests                    2.32.5
safetensors                 0.7.0
sentry-sdk                  2.54.0
sympy                       1.14.0
tokenizers                  0.22.2
torch                       2.10.0+cu128
torchvision                 0.25.0+cu128
transformers                5.3.0
triton                      3.6.0
typing-extensions           4.15.0
wandb                       0.25.1
```

</details>

## Directory Layout

```
DriveLM/
├── configs/                          # YAML configs (layer 1: hardware, layer 2: experiments)
│   ├── gh200.yaml                    # GH200 base: bf16, bs=4, no quant
│   ├── 4070ti.yaml                   # 4070 Ti: 4-bit quant, bs=1
│   ├── baseline.yaml                 # No compression (inherits gh200)
│   ├── avg_pool_c4.yaml              # AvgPool 4x compression
│   ├── fastervlm_c4.yaml             # FasterVLM 4x compression
│   ├── prumerge_c4.yaml              # PruMerge 4x compression
│   ├── pyramiddrop_c4.yaml           # PyramidDrop 4x compression
│   └── run_all.sh                    # Run all 5 experiments sequentially
├── scripts/
│   ├── train_lora.py                 # LoRA fine-tuning (reads --config YAML)
│   ├── demo_inference.py             # Quick inference demo (10 samples)
│   ├── eval_full.py                  # Full evaluation (per-category accuracy)
│   ├── visual_compress.py            # 4 compression methods implementation
│   ├── benchmark_visual_compression.py  # End-to-end latency benchmark
│   ├── benchmark_ttft.py             # TTFT (prefill-only) benchmark
│   ├── eval_compression_scaling.py   # Accuracy vs compression ratio curve
│   ├── visualize_token_retention.py  # Token retention heatmap visualization
│   ├── convert_data.py               # DriveLM → Qwen conversation format
│   ├── merge_lora.py                 # Merge LoRA into base model
│   ├── quantize_model.py             # Post-training quantization
│   └── download_drivelm_data.py      # Download DriveLM dataset
├── data/                             # Raw DriveLM data
│   ├── QA_dataset_nus/v1_1_train_nus.json
│   └── nuscenes/samples/             # 24,432 images (6 cameras)
├── data_processed/                   # Processed conversation format
│   ├── train.json                    # 359,057 samples (95%)
│   ├── val.json                      # 18,898 samples (5%)
│   └── train_mini.json               # 500 samples (quick test)
├── checkpoints_qwen25/               # LoRA checkpoints (~50MB each)
│   ├── default/checkpoint-46000      # Baseline (no compression)
│   ├── default/checkpoint-51000      # Baseline final
│   ├── fastervlm_c4/final            # FasterVLM 4x
│   ├── prumerge_c4/final             # PruMerge 4x
│   └── pyramiddrop_c4/final          # PyramidDrop 4x
├── results/                          # Eval & benchmark results (JSON + logs)
├── visualizations/                   # Token retention heatmaps (PNG)
├── logs/                             # Training & eval logs
└── docs/                             # Documentation
```

## Commands

### 1. Data Preparation

```bash
# Download DriveLM dataset from HuggingFace
python scripts/download_drivelm_data.py

# Convert raw DriveLM → Qwen conversation format
python scripts/convert_data.py
# Output: data_processed/{train,val,train_mini}.json
```

### 2. LoRA Training

```bash
# Quick test (500 samples)
python scripts/train_lora.py --config configs/gh200.yaml --mini

# Full training — baseline (no compression), 1 epoch
python scripts/train_lora.py --config configs/gh200.yaml

# Full training — with visual token compression
python scripts/train_lora.py --config configs/baseline.yaml          # no compression
python scripts/train_lora.py --config configs/fastervlm_c4.yaml      # FasterVLM 4x
python scripts/train_lora.py --config configs/prumerge_c4.yaml       # PruMerge 4x
python scripts/train_lora.py --config configs/pyramiddrop_c4.yaml    # PyramidDrop 4x
python scripts/train_lora.py --config configs/avg_pool_c4.yaml       # AvgPool 4x

# Run all 5 compression experiments sequentially
bash configs/run_all.sh

# Override hyperparams via CLI
python scripts/train_lora.py --config configs/gh200.yaml --bs 8 --lr 2e-4 --epochs 2

# Background training with logging
nohup python -u scripts/train_lora.py --config configs/gh200.yaml \
    2>&1 | tee logs/train_full.log &
```

### 3. Inference (Quick Demo)

```bash
# Inference with final LoRA checkpoint
python scripts/demo_inference.py --config configs/gh200.yaml

# Inference with specific checkpoint
python scripts/demo_inference.py --config configs/gh200.yaml \
    --lora checkpoints_qwen25/default/checkpoint-46000

# Base model only (no LoRA)
python scripts/demo_inference.py --config configs/gh200.yaml --no-lora

# More samples, shorter output
python scripts/demo_inference.py --config configs/gh200.yaml --n 20 --max-tokens 256
```

### 4. Full Evaluation (Per-Category Accuracy)

```bash
# Evaluate baseline checkpoint (100 samples per category, 400 total)
python scripts/eval_full.py \
    --config configs/baseline.yaml \
    --lora checkpoints_qwen25/default/checkpoint-46000 \
    --max-per-cat 100

# Evaluate compression checkpoints
python scripts/eval_full.py \
    --config configs/fastervlm_c4.yaml \
    --lora checkpoints_qwen25/fastervlm_c4/final \
    --max-per-cat 100

python scripts/eval_full.py \
    --config configs/prumerge_c4.yaml \
    --lora checkpoints_qwen25/prumerge_c4/final \
    --max-per-cat 100

python scripts/eval_full.py \
    --config configs/pyramiddrop_c4.yaml \
    --lora checkpoints_qwen25/pyramiddrop_c4/final \
    --max-per-cat 100
```

### 5. Benchmarks

#### 5a. End-to-End Latency Benchmark

```bash
# Compare all compression methods (30 samples, max 128 output tokens)
python scripts/benchmark_visual_compression.py \
    --config configs/gh200.yaml \
    --experiments \
        baseline=checkpoints_qwen25/default/checkpoint-46000 \
        fastervlm=checkpoints_qwen25/fastervlm_c4/final:fastervlm:4 \
        prumerge=checkpoints_qwen25/prumerge_c4/final:prumerge:4 \
        pyramiddrop=checkpoints_qwen25/pyramiddrop_c4/final:pyramiddrop:4 \
    --n 30 --max-tokens 128
# Output: benchmark_visual_compression.json
```

#### 5b. TTFT (Time-to-First-Token) Benchmark

```bash
# Prefill-only latency (generates 1 token, isolates prefill speedup)
python scripts/benchmark_ttft.py \
    --config configs/gh200.yaml \
    --experiments \
        baseline=checkpoints_qwen25/default/checkpoint-46000 \
        fastervlm=checkpoints_qwen25/fastervlm_c4/final:fastervlm:4 \
        prumerge=checkpoints_qwen25/prumerge_c4/final:prumerge:4 \
        pyramiddrop=checkpoints_qwen25/pyramiddrop_c4/final:pyramiddrop:4 \
    --n 30
# Output: results/benchmark_ttft.json
```

#### 5c. Compression Ratio Scaling Curve

```bash
# Accuracy vs compression ratio (1x→16x) using baseline checkpoint
python scripts/eval_compression_scaling.py \
    --config configs/gh200.yaml \
    --lora checkpoints_qwen25/default/checkpoint-46000 \
    --ratios 1,2,4,8,16 \
    --samples-per-cat 100
# Output: results/compression_scaling.json
```

### 6. Visualization

```bash
# Token retention heatmaps (which spatial regions are kept/dropped)
python scripts/visualize_token_retention.py \
    --config configs/gh200.yaml \
    --methods fastervlm,prumerge,pyramiddrop \
    --ratio 4 --n 5
# Output: visualizations/retention_*.png, visualizations/token_importance_heatmap.png
```

### 7. Model Export

```bash
# Merge LoRA adapter into base model
python scripts/merge_lora.py \
    --config configs/gh200.yaml \
    --lora checkpoints_qwen25/default/checkpoint-46000 \
    --output merged_model/

# Quantize merged model
python scripts/quantize_model.py --model merged_model/ --bits 4
```

## Config System

Two-layer YAML config with inheritance:

```
gh200.yaml (Layer 1: hardware)     ← base config
  ├── baseline.yaml (Layer 2)      ← base_config: gh200.yaml
  ├── fastervlm_c4.yaml            ← base_config: gh200.yaml, compress_method: fastervlm
  ├── prumerge_c4.yaml             ← base_config: gh200.yaml, compress_method: prumerge
  ├── pyramiddrop_c4.yaml          ← base_config: gh200.yaml, compress_method: pyramiddrop
  └── avg_pool_c4.yaml             ← base_config: gh200.yaml, compress_method: avg_pool
```

CLI args (`--bs`, `--lr`, `--epochs`, `--compress-method`, `--compress-ratio`) override YAML values.

## Experiment Format (for benchmarks)

```
name=lora_path[:compress_method:compress_ratio]

# Examples:
baseline=checkpoints_qwen25/default/checkpoint-46000
fastervlm=checkpoints_qwen25/fastervlm_c4/final:fastervlm:4
prumerge=checkpoints_qwen25/prumerge_c4/final:prumerge:4
```

## Results Files

| File | Description |
|------|-------------|
| `results/eval_baseline_full.json` | Baseline accuracy (400 samples) |
| `results/eval_fastervlm_c4_full.json` | FasterVLM accuracy |
| `results/eval_prumerge_c4_full.json` | PruMerge accuracy |
| `results/eval_pyramiddrop_c4_full.json` | PyramidDrop accuracy |
| `results/benchmark_ttft.json` | TTFT latency benchmark |
| `results/compression_scaling.json` | Accuracy vs compression ratio curve |
| `benchmark_visual_compression.json` | End-to-end latency benchmark |
| `results/README.md` | Results summary table |

---

*Last updated: 2026-03-21*
