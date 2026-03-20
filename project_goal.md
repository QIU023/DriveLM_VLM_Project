# DriveLM Efficient VLM 项目目标

> 面试叙事核心：在自动驾驶场景下，从 LoRA 微调 → token 压缩 → 量化推理 → 部署 serving，完整走通"小模型高效落地"全链路。

---

## 当前进度总览

| 阶段 | 状态 | 设备 | 说明 |
|------|------|------|------|
| Layer 1: LoRA 微调 | **Epoch 1 完成** (ckpt-46000) | GH200 | 继续训练中，等更好的 ckpt |
| Layer 2: Visual Token 压缩 | **2/4 完成** | GH200 | fastervlm_c4 ✅, prumerge_c4 ✅, pyramiddrop 进行中 |
| **Layer 3: 本地推理/部署** | **脚本就绪，待执行** | **4070Ti** | 3 个 LoRA 的 merge→量化→benchmark 全链路 |
| Layer 4: Benchmark 报告 | 待开始 | 两者 | 所有实验完成后整理 |

---

## Layer 1: LoRA 微调 (GH200) — Epoch 1 Done

**已完成：**
- Qwen2.5-VL-3B + LoRA (r=16, alpha=32) 在 DriveLM v1.1 上完成第一轮训练
- checkpoint-46000 已保存，可用于推理
- 训练配置：bf16, bs=4, grad_accum=2, lr=1e-4, max_seq=2048

**进行中：**
- GH200 继续跑后续 epoch，等待 loss 收敛 / 更好的 checkpoint
- 后续拿到 best ckpt 后直接替换到 Layer 3 链路中重新评测

---

## Layer 3: 4070Ti 本地量化部署全链路 — 当前任务

> **原则：本地不跑 raw torch 推理，所有推理/评测通过推理框架完成。**
> 目标：3 个 LoRA (baseline / fastervlm_c4 / prumerge_c4) 合并 → 量化导出 → 推理框架部署 → throughput 测试。
> 精度评测在 GH200 上完成，4070Ti 只测性能。后续有更好的 ckpt 只需替换路径重跑。

### Step 3.1: LoRA 合并 & 模型导出 — 脚本就绪

三个 LoRA → 三个完整模型：

| LoRA | 来源 | 合并后目录 | 状态 |
|------|------|-----------|------|
| baseline | `checkpoints_qwen25/checkpoint-46000` | `models/qwen25vl-3b-drivelm-baseline-merged/` | ✅ 已合并 (已有 `models/qwen25vl-3b-drivelm-merged/`) |
| fastervlm_c4 | `checkpoints_qwen25/fastervlm_c4/checkpoint-XXX` | `models/qwen25vl-3b-drivelm-fastervlm_c4-merged/` | 待合并 |
| prumerge_c4 | `checkpoints_qwen25/prumerge_c4/checkpoint-XXX` | `models/qwen25vl-3b-drivelm-prumerge_c4-merged/` | 待合并 |

```bash
# 方案 A: 在 GH200 上合并（ckpt 在那边），再 SCP merged 模型到本地
python scripts/batch_merge.py \
    baseline=checkpoints_qwen25/checkpoint-46000 \
    fastervlm_c4=checkpoints_qwen25/fastervlm_c4/checkpoint-XXX \
    prumerge_c4=checkpoints_qwen25/prumerge_c4/checkpoint-XXX

# 方案 B: 只 SCP LoRA ckpt (~50MB each) 到本地，本地 CPU 合并
scp -r ubuntu@gh200:~/DriveLM/checkpoints_qwen25/fastervlm_c4/checkpoint-XXX checkpoints_qwen25/fastervlm_c4/
scp -r ubuntu@gh200:~/DriveLM/checkpoints_qwen25/prumerge_c4/checkpoint-XXX checkpoints_qwen25/prumerge_c4/
python scripts/batch_merge.py \
    fastervlm_c4=checkpoints_qwen25/fastervlm_c4/checkpoint-XXX \
    prumerge_c4=checkpoints_qwen25/prumerge_c4/checkpoint-XXX
# baseline 已有，会自动跳过
```

### Step 3.2: 量化导出（AWQ / GPTQ / GGUF）— 脚本就绪

```bash
# 单个模型量化
python scripts/quantize_model.py awq  --input models/qwen25vl-3b-drivelm-baseline-merged
python scripts/quantize_model.py gptq --input models/qwen25vl-3b-drivelm-baseline-merged
python scripts/quantize_model.py gguf --input models/qwen25vl-3b-drivelm-baseline-merged

# 批量量化: 3 模型 × 3 格式 = 9 个量化模型
bash scripts/batch_quantize.sh          # 全部
bash scripts/batch_quantize.sh awq      # 只做 AWQ
bash scripts/batch_quantize.sh baseline # 只做 baseline 的 3 种格式
```

依赖安装:
```bash
pip install autoawq       # AWQ
pip install auto-gptq     # GPTQ
pip install gguf           # GGUF (还需 llama.cpp 源码)
```

输出目录: `models/qwen25vl-3b-drivelm-{name}-{awq,gptq,gguf}/`

### Step 3.3: 推理框架部署

```bash
# vLLM (需 WSL2/Linux, Windows 不支持)
python -m vllm.entrypoints.openai.api_server \
    --model models/qwen25vl-3b-drivelm-baseline-awq --port 8000

# llama.cpp (Windows 原生支持)
llama-server -m models/qwen25vl-3b-drivelm-baseline-gguf/model-q4_k_m.gguf \
    --port 8000

# Ollama (Windows 原生, 最简单)
ollama serve   # port 11434
```

| 框架 | 平台 | 模型格式 | 多模态 | 备注 |
|------|------|---------|--------|------|
| vLLM | WSL2/Linux | AWQ/GPTQ | ✅ 原生支持 | PagedAttention, continuous batching |
| llama.cpp | Windows | GGUF | ✅ 需 mmproj | 轻量, CPU offload 可选 |
| Ollama | Windows | GGUF | ✅ | 一键部署, 最简单 |

### Step 3.4: Throughput Benchmark — 脚本就绪

```bash
# 启动 server 后运行 benchmark
python scripts/benchmark_throughput.py \
    --api-base http://localhost:8000/v1 \
    --model qwen25vl-3b-drivelm-baseline-awq \
    --concurrency 1,2,4 \
    --num-requests 20 \
    --image-dir data/nuscenes/samples/CAM_FRONT

# Text-only 模式 (无图片)
python scripts/benchmark_throughput.py \
    --api-base http://localhost:8000/v1 \
    --model MODEL --no-image

# Ollama
python scripts/benchmark_throughput.py \
    --api-base http://localhost:11434/v1 \
    --model MODEL
```

测量指标: TTFT (ms), Tokens/s, 端到端延迟, GPU 显存, 并发 throughput (P50/P95)

### Step 3.5: TensorRT-LLM（时间允许）

- [ ] 导出量化模型到 TensorRT-LLM engine
- [ ] kernel fusion + INT4/FP8 进一步优化
- [ ] 对比 vLLM vs TRT-LLM vs llama.cpp 全维度 benchmark

---

## Layer 2: Visual Token 压缩 (GH200) — 核心亮点

> 在 visual encoder 输出之后、送入 LLM 之前，压缩视觉 token。
> 已实现 4 种方法，通过 YAML config 切换，1 epoch 训练 + 对比。

**已实现的压缩方法 (`scripts/visual_compress.py`)**：
1. **avg_pool** — 2x2 空间平均池化，保留 grid 结构
2. **fastervlm** — L2 norm 重要性选择 top-K token (FasterVLM, 2024)
3. **prumerge** — 剪枝低重要性 token + 合并到最近邻 (LLaVA-PruMerge, 2024)
4. **pyramiddrop** — 两阶段渐进式丢弃 (PyramidDrop, 2024 简化版)

**实验配置** (`configs/`):
```bash
python scripts/train_lora.py --config configs/baseline.yaml       # 无压缩 baseline
python scripts/train_lora.py --config configs/avg_pool_c4.yaml    # avg_pool 4x
python scripts/train_lora.py --config configs/fastervlm_c4.yaml   # fastervlm 4x
python scripts/train_lora.py --config configs/prumerge_c4.yaml    # prumerge 4x
python scripts/train_lora.py --config configs/pyramiddrop_c4.yaml # pyramiddrop 4x
bash configs/run_all.sh                                           # 顺序跑全部 5 组
```

- 所有实验 1 epoch，lr=2e-4，每 200 步验证
- Checkpoint 按实验名保存到 `checkpoints_qwen25/{experiment}/`
- 面试话术："高分辨率输入下视觉 token 可达上千个，LLM prefill 和 KV cache 线性增长。我探索了几种 token 压缩策略，在精度损失 X% 的情况下把推理延迟降了 Y%。"

---

## Layer 4: 系统性 Benchmark 报告

### 已完成: GGUF Q4_K_M + llama.cpp Throughput (4070Ti 12GB, text-only)

> 测试条件: llama-server b8429, CUDA 12.4, -ngl 99, -c 2048, 4 slots, 20 requests/level, max_tokens=128
> 注意: 此为纯 LLM backbone throughput。三个模型的 token 压缩差异只体现在 visual encoder 阶段（prefill），LLM decode 速度基本一致。

**三模型 × 三并发度对比**

| 模型 | 并发 | TTFT P50 (ms) | TTFT P95 (ms) | Tokens/s P50 | Aggregate TPS | Latency P50 (ms) | GPU (MB) |
|------|:----:|:-------------:|:-------------:|:------------:|:-------------:|:-----------------:|:--------:|
| baseline | 1 | 142 | 412 | 170 | 72 | 225 | 5264 |
| baseline | 2 | 159 | 340 | 148 | 123 | 265 | 5241 |
| baseline | 4 | 181 | 344 | 121 | 196 | 313 | 5234 |
| fastervlm | 1 | 144 | 357 | 173 | 78 | 240 | 5326 |
| fastervlm | 2 | 181 | 360 | 154 | 126 | 295 | 5315 |
| fastervlm | 4 | 165 | 318 | 124 | 202 | 312 | 5308 |
| prumerge | 1 | 141 | 356 | 173 | 74 | 233 | 5238 |
| prumerge | 2 | 174 | 347 | 153 | 123 | 269 | 5234 |
| prumerge | 4 | 182 | 366 | 114 | 197 | 338 | 5219 |

**关键发现:**
- 单请求 decode 速度: ~170 tokens/s (三个模型一致，因为 LLM 权重相同结构)
- 4 并发聚合 throughput: ~196-202 tokens/s
- TTFT: ~140ms (单请求) → ~180ms (4 并发)
- 显存占用: ~5.2GB (模型 1.8GB + KV cache + compute buffer)，12GB 卡还剩 ~7GB
- Token 压缩的加速效果需要在多模态推理 (带图片 prefill) 场景下才能体现

**量化方案对比 (4070Ti 12GB)**

| 配置 | 模型大小 | BPW | TTFT (ms) | Tokens/s | 峰值显存 | 框架 | 状态 |
|------|----------|:---:|:---------:|:--------:|:--------:|------|:----:|
| Merged fp16 (GH200) | 6.99 GB | 16.0 | — | — | — | torch | 仅 GH200 |
| **GGUF Q4_K_M** | **1.8 GB** | **4.99** | **142** | **170** | **5.2 GB** | **llama.cpp** | **✅ Done** |
| AWQ W4A16 | ~2 GB | ~4 | — | — | — | vLLM | ❌ 库不兼容 transformers 5.x |
| GPTQ W4A16 | ~2 GB | ~4 | — | — | — | vLLM | ❌ 库不兼容 transformers 5.x |

> AWQ/GPTQ: autoawq 已 deprecated，auto-gptq 不兼容 transformers 5.3.0。后续可在 GH200 上用 llm-compressor 量化，SCP 到本地用 vLLM(WSL2) 部署。

**端到端链路对比**

| 配置 | 模型大小 | Tokens/s | 端到端延迟 | 部署复杂度 | 状态 |
|------|----------|:--------:|:---------:|:---------:|:----:|
| LoRA bf16 (GH200) | 6.99 GB | — | — | 高（需大卡） | GH200 评测 |
| **GGUF Q4_K_M + llama.cpp (4070Ti)** | **1.8 GB** | **170** | **225 ms** | **低** | **✅ Done** |
| GGUF + Ollama (4070Ti) | 1.8 GB | — | — | 最低（一键） | 待测 |
| AWQ + vLLM (4070Ti WSL2) | ~2 GB | — | — | 中 | 待做 |

---

## 设备分工

| 设备 | 用途 | 理由 |
|------|------|------|
| GH200 (96GB HBM3) | LoRA 微调、token 压缩实验、全精度评测 baseline | 大显存做训练和 bf16 完整推理 |
| 4070Ti (12GB VRAM) | 量化导出、推理框架部署、性能评测 | 模拟真实部署场景，不跑 raw torch |

---

## 面试叙事要点

1. **为什么选 3B 不选 7B？** — 目标是高效部署，3B 在 12GB 消费级显卡上可量化到 ~2GB，体现工程落地能力
2. **为什么在 4070Ti 上做部署？** — 刻意模拟资源受限的真实部署环境，而非用大卡掩盖效率问题
3. **量化不只是跑通** — AWQ/GPTQ/GGUF 三种方案横评，能说清精度-速度 tradeoff
4. **推理框架选型** — vLLM vs llama.cpp vs Ollama，不同场景下的最优选择
5. **完整链路** — 从数据处理 → 微调 → LoRA 合并 → 量化导出 → 框架部署 → API 评测，每一步都可展开讲细节
