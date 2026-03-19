# DriveLM Efficient VLM 项目目标

> 面试叙事核心：在自动驾驶场景下，从 LoRA 微调 → token 压缩 → 量化推理 → 部署 serving，完整走通"小模型高效落地"全链路。

---

## 当前进度总览

| 阶段 | 状态 | 设备 | 说明 |
|------|------|------|------|
| Layer 1: LoRA 微调 | **Epoch 1 完成** (ckpt-46000) | GH200 | 继续训练中，等更好的 ckpt |
| Layer 2: Visual Token 压缩 | 待开始 | GH200 | 核心创新点，依赖 Layer 1 best ckpt |
| **Layer 3: 本地推理/部署** | **当前任务** | **4070Ti** | 先打通全链路，后续换 ckpt 即可 |
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
> 目标：LoRA 合并 → 量化导出 → 推理框架部署 → 通过 API 做性能 & 精度评测。
> 后续有更好的 ckpt 只需替换路径重跑一遍。

### Step 3.1: LoRA 合并 & 模型导出

- [x] checkpoint-46000 已拉到本地
- [ ] 将 LoRA adapter 合并回 base model（`merge_and_unload()`），导出完整权重
- [ ] 验证合并后模型输出与 base+adapter 一致
- [ ] 输出目录：`models/qwen25vl-3b-drivelm-merged/`

### Step 3.2: 量化导出（多格式）

- [ ] **AWQ (W4A16)** — 用 autoawq 做 calibration + 量化，导出 safetensors
- [ ] **GPTQ (W4A16)** — 用 auto-gptq 做 calibration + 量化，导出 safetensors
- [ ] **GGUF (Q4_K_M)** — 用 llama.cpp 的 convert 工具导出，供 llama.cpp / Ollama 使用
- [ ] 每种格式记录：文件大小、量化耗时
- [ ] 输出目录：`models/qwen25vl-3b-drivelm-{awq,gptq,gguf}/`

### Step 3.3: 推理框架部署

- [ ] **vLLM**：加载 AWQ/GPTQ 量化模型，启动 OpenAI-compatible API server
- [ ] **llama.cpp / Ollama**：加载 GGUF 模型，启动本地 server
- [ ] 验证框架能正确处理 Qwen2.5-VL 的多模态输入（image + text）
- [ ] 如果 vLLM Windows 不支持，备选 SGLang 或 WSL2
- [ ] 记录每种框架的启动配置和显存占用

### Step 3.4: 性能 & 精度评测（通过框架 API）

- [ ] 编写评测客户端脚本，调用框架的 OpenAI-compatible API
- [ ] 在 val.json 上跑精度评测（分类别 accuracy: perception / prediction / planning / behavior）
- [ ] 性能指标：TTFT (Time to First Token)、生成速度 (tokens/s)、端到端延迟、显存峰值
- [ ] 对比矩阵：AWQ vs GPTQ vs GGUF × vLLM vs llama.cpp
- [ ] 并发压测：1/2/4 并发下的 throughput 变化

### Step 3.5: TensorRT-LLM（时间允许）

- [ ] 导出量化模型到 TensorRT-LLM engine
- [ ] kernel fusion + INT4/FP8 进一步优化
- [ ] 对比 vLLM vs TRT-LLM vs llama.cpp 全维度 benchmark

---

## Layer 2: Visual Token 压缩 (GH200) — 核心亮点

> 依赖 Layer 1 的 best checkpoint 作为 baseline，在此基础上做 token 压缩实验。

- 在 visual encoder 输出之后、送入 LLM 之前，加 token selection / pooling 模块
- 将视觉 token 从 256+ 砍到 64/32，测 accuracy-latency tradeoff
- Qwen2.5-VL 动态分辨率方案天然适合此实验
- 面试话术："高分辨率输入下视觉 token 可达上千个，LLM prefill 和 KV cache 线性增长。我探索了几种压缩策略，精度损失 X% 的情况下推理延迟降了 Y%。"

---

## Layer 4: 系统性 Benchmark 报告

最终整理成对比表：

**量化方案对比 (4070Ti 12GB)**

| 配置 | 模型大小 | Accuracy | TTFT | Tokens/s | 峰值显存 | 框架 |
|------|----------|----------|------|----------|----------|------|
| Merged bf16 (baseline, GH200) | ~6GB | | | | | torch |
| AWQ W4A16 | ~2GB | | | | | vLLM |
| GPTQ W4A16 | ~2GB | | | | | vLLM |
| GGUF Q4_K_M | ~2GB | | | | | llama.cpp |

**推理框架对比 (同一量化模型)**

| 框架 | TTFT | Tokens/s | 并发 Throughput | 显存占用 | 备注 |
|------|------|----------|----------------|----------|------|
| vLLM (AWQ) | | | | | PagedAttention, continuous batching |
| llama.cpp (GGUF) | | | | | CPU offload 可选 |
| Ollama (GGUF) | | | | | 一键部署体验 |
| TensorRT-LLM | | | | | 时间允许再做 |

**端到端链路对比**

| 配置 | Accuracy | 端到端延迟 | 部署复杂度 |
|------|----------|-----------|-----------|
| LoRA (bf16, GH200) | | | 高（需大卡） |
| AWQ + vLLM (4070Ti) | | | 中 |
| GGUF + Ollama (4070Ti) | | | 低（一键） |
| + Token Compress (未来) | | | 中 |

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
