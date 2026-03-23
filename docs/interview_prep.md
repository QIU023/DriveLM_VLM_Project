# DriveLM 项目面试准备 — 技术知识点 & 复习资料

## 1. VLM 架构 (必须能画出来)

**整体 pipeline:**
```
Image → ViT (Vision Encoder) → Projection Layer → [Compression Module] → LLM Backbone → Text Output
```

**核心知识点：**

### a) Vision Transformer (ViT)
- 图像切 patch → linear embedding → transformer encoder → visual tokens
- Qwen2.5-VL 用的是动态分辨率 ViT，不同图片产生不同数量 visual tokens
- 你的实验中 min/max pixels 配置决定了 480 tokens

> 复习：[An Image is Worth 16x16 Words (ViT 原论文)](https://arxiv.org/abs/2010.11929)，重点看 Section 3

### b) Projection Layer（视觉-语言对齐）
- ViT 输出的 visual token embedding 维度 ≠ LLM embedding 维度
- Projection layer (MLP / linear) 把 visual tokens 映射到 LLM 的 embedding space
- **你的压缩模块插在 projection 之后、LLM 之前** — 面试必须能解释为什么选这个位置

> 面试高频问题："为什么不在 ViT 内部压缩？" → 因为 ViT 内部压缩需要改 attention mask / positional encoding，在 projection 后压缩对 ViT 是无侵入的，且 visual tokens 已经在 LLM space 里，可以用 LLM 的 attention score 做 importance 判断

---

## 2. 三种 Visual Token Compression Methods

**必须能讲清楚每种方法的一句话原理 + 区别：**

| Method | 核心思路 | 复杂度 | 你的数据 |
|---|---|---|---|
| **FasterVLM** | 按 L2 norm 排序，保留 top-K 重要 tokens | O(n log n) | 1ms，zero accuracy loss |
| **PruMerge** | Prune 低重要性 tokens + merge 到最近的保留 token | O(n²) | 125ms overhead，精度最高 |
| **PyramidDrop** | 分阶段逐步 drop，每阶段用 attention score 筛选 | O(n log n) | 1ms，效果接近 PruMerge |

**关键 tradeoff 能讲清楚：**
- FasterVLM/PyramidDrop：几乎零延迟，精度好
- PruMerge：merge 操作带来 125ms overhead，但信息保留最完整（因为 pruned tokens 的信息 merge 回了 kept tokens）

> 复习论文：
> - FasterVLM: [Efficient Multimodal Large Language Models via Visual Token Dropping](https://arxiv.org/abs/2405.14870)
> - PruMerge: [PruMerge: Token Reduction for Efficient VLMs](https://arxiv.org/abs/2403.15388)
> - PyramidDrop: [PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction](https://arxiv.org/abs/2410.17247)

---

## 3. LoRA (Low-Rank Adaptation)

**必会知识：**
- 原理：冻结原始权重 W，训练低秩分解 ΔW = BA，其中 B∈R^(d×r), A∈R^(r×k)
- **rank=16** 意味着只训练 16 维的低秩更新，参数量 << full fine-tuning
- **alpha=32**：scaling factor，实际更新 = (alpha/rank) × BA = 2× BA
- 你只保存 LoRA adapter (~50MB)，不保存完整模型

**面试高频问题：**
- "为什么 rank 选 16？" → 经验值，DriveLM 是 domain-specific QA 不需要太大 rank；rank 太大过拟合 + 训练慢
- "LoRA 加在哪些层？" → 通常加在 attention 的 Q/K/V/O projection（检查一下你的 config 里 `target_modules` 是什么）
- "LoRA vs QLoRA？" → QLoRA = 4-bit quantized base model + LoRA。你在 GH200 上用 bf16 不需要 QLoRA，但 4070Ti config 里 `quantize: true` 就是类似 QLoRA 的思路

> 复习：[LoRA 原论文](https://arxiv.org/abs/2106.09685)，重点 Section 4 (method) + Table 2 (rank ablation)

---

## 4. GGUF 量化 + llama.cpp 部署

### GGUF Q4_K_M 含义拆解
- **GGUF**：llama.cpp 的模型格式（替代旧的 GGML），存储量化后的权重 + metadata
- **Q4**：4-bit 量化
- **K**：k-quant，按 block 做混合精度量化（重要层用更高精度）
- **M**：medium quality preset（比 Q4_K_S 精度高，比 Q4_K_L 小）

### 与 AWQ/GPTQ 的区别（面试必问）

| | GGUF (llama.cpp) | AWQ | GPTQ |
|---|---|---|---|
| 量化方式 | Block-wise k-quant | Activation-aware weight quant | One-shot weight quant |
| 推理框架 | llama.cpp (CPU/GPU) | vLLM / SGLang | vLLM / SGLang |
| 优势 | 消费级 GPU 友好，纯 C++ | 精度好，GPU 推理快 | 成熟，GPU 推理快 |
| 你的场景 | ✅ 4070Ti 部署 | 未做 | 未做 |

### llama.cpp 关键概念
- C/C++ 实现的 LLM 推理引擎，支持 CPU + GPU offload
- 你用的是 llama.cpp server（HTTP API），不是纯 CLI
- **170 tok/s** 是 decode throughput（逐 token 生成速度）
- **142ms TTFT** 是 prompt processing + first token 的延迟

> 面试问题："为什么选 llama.cpp 而不是 vLLM？" → 4070Ti 只有 12GB，vLLM 对显存要求更高；llama.cpp 对消费级硬件优化更好，支持 CPU offload 作为 fallback

---

## 5. 推理性能指标

**必须能定义这几个指标：**

| 指标 | 含义 | 你的数据 |
|---|---|---|
| **TTFT** (Time to First Token) | 从请求到第一个 token 生成的延迟 | 142ms (4070Ti), 205ms (GH200) |
| **Tokens/s** (Decode throughput) | 每秒生成的 token 数 | 170 tok/s |
| **Prefill time** | 处理所有 input tokens 的时间 | ~98ms |
| **Aggregate TPS** | 并发场景下总吞吐 | 196 tok/s @ concurrency=4 |

**TTFT = Vision encoding + Compression + Prefill + Sampling first token**

> 面试问题："压缩后 TTFT 为什么没有明显下降？" → 480 tokens 本身就很少，prefill 已经很快（~98ms）。压缩的收益在高分辨率场景（1000+ tokens）才会显著。这也是你 profile 的 insight 之一。

---

## 6. Pareto Frontier（accuracy-efficiency trade-off）

- 你做了 FasterVLM 1x/2x/4x/8x/16x 的 scaling 实验
- Pareto frontier = 在给定 efficiency 下能达到的最优 accuracy 的边界
- 面试能画出这张图：X 轴 = visual tokens (或 compression ratio)，Y 轴 = accuracy
- **Key insight**：曲线非常平（2x→8x 几乎没掉），说明 driving QA 的 visual token 冗余度极高

---

## 7. 项目整体 Narrative（面试讲故事）

### 30 秒版本
> "I built an end-to-end efficient VLM pipeline for autonomous driving QA. Starting with Qwen2.5-VL-3B, I LoRA fine-tuned it on DriveLM, then integrated 3 visual token compression methods and profiled the accuracy-efficiency Pareto curve — finding that 4x compression loses zero accuracy and even 16x only drops 2.4%. For deployment, I quantized the model to 4-bit GGUF and served it on a consumer 4070Ti via llama.cpp, achieving 170 tokens/s at 142ms TTFT."

### 面试官可能的 follow-up 方向
1. "你怎么决定压缩放在哪一层？" → 见上面 §1b
2. "为什么不用更大的模型？" → 项目目标就是 efficient VLM，3B 是 sweet spot
3. "如果要上生产环境你会改什么？" → vLLM/SGLang 替换 llama.cpp，batched inference，KV cache 优化
4. "和你之前的持续学习论文有什么关系？" → 下一步计划是 VLM continual learning，用 feature-level + logit-level KD 防止灾难性遗忘

---

## 复习优先级

§1 VLM 架构 > §2 三种压缩方法 > §3 LoRA > §4 GGUF/llama.cpp > §5 性能指标 > §6 Pareto > §7 Narrative

---

*Generated: 2026-03-21*
