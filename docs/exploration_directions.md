# VLM 项目探索方向规划

> **定位**：Efficient VLM + 蒸馏方向，专注小模型 + 快速微调。不做通用多模态预训练，而是通过项目深入理解跨模态交互机制。目标覆盖：自动驾驶、多模态推荐、视频理解。

---

## 第一层：领域微调（基座能力）

### 方向 1：DriveLM LoRA 微调 ✅ 进行中
- Qwen2.5-VL-3B + LoRA 在 DriveLM nuScenes 数据上的 QA 微调
- 涵盖 perception / prediction / planning / behavior 四类任务
- **定位**：基座项目，证明你能跑通 VLM 微调全流程
- **求职话术**："我在 DriveLM 数据集上对 Qwen2.5-VL 做了 LoRA 微调，让通用 VLM 适配了自动驾驶场景的结构化 QA 任务，在 perception/planning 等类别上相比 base model 有显著提升。"

---

## 第二层：Efficient VLM（核心技术深度）

### 方向 2：Visual Token 压缩 — 已规划
- 在 visual encoder 输出之后、LLM 之前加 token selection / pooling 模块
- 把视觉 token 从 256+ 砍到 64/32，测 accuracy-latency tradeoff
- Qwen2.5-VL 的动态分辨率方案天然适合做这个实验
- **求职话术**："Qwen2.5-VL 在高分辨率输入下视觉 token 可达上千个，LLM 的 prefill 开销和 KV cache 都随之线性增长。我探索了几种 token 压缩策略，在精度损失 X% 的情况下把推理延迟降了 Y%。"

### 方向 3：跨模态 Adapter 结构探索
**动机**：VLM 中 visual encoder 和 LLM 之间的 projector/adapter 是跨模态交互的核心瓶颈。大多数开源 VLM（LLaVA、Qwen-VL）用的是简单的 MLP projector，这里有很大的优化空间。这个方向直接展示你对**跨模态交互机制**的理解深度。

**具体做法**：
- 对比不同 projector 结构：Linear / MLP / Q-Former / Perceiver Resampler / Cross-Attention
- 在 Qwen2.5-VL 上替换 projector，固定 visual encoder 和 LLM，只训练 projector + LoRA
- 关键指标：参数量、训练速度、DriveLM QA 准确率、推理延迟
- 重点分析：不同结构对 perception vs planning 类任务的影响差异（perception 更依赖细粒度视觉特征，planning 更依赖语义压缩）

**工作量**：3-4 天

**求职话术**："我系统对比了 5 种跨模态 projector 结构在驾驶 VLM 上的表现。发现 Perceiver Resampler 在保持精度的同时，训练效率比 MLP projector 高 X%，因为它天然支持视觉 token 降采样，同时完成了特征对齐和压缩。"

**为什么重要**：这个方向直接回答了面试高频问题——"VLM 中图像和文本是怎么交互的？你觉得当前方案的瓶颈在哪？"

---

### 方向 4：VLM 知识蒸馏（大→小，跨场景迁移）
**动机**：Efficient VLM 的两大支柱是压缩和蒸馏。方向 2 做压缩，这个做蒸馏。更关键的是，蒸馏不仅是自动驾驶场景的需求——多模态推荐和视频理解同样需要把大模型能力迁移到小模型。

**具体做法**：

**阶段 A — 同域蒸馏（自动驾驶）**：
- Teacher：Qwen2.5-VL-7B（或 API 调用 72B）在 DriveLM 上生成高质量 response
- Student：Qwen2.5-VL-3B + LoRA
- 对比三种训练方式：原始 GT / teacher response / 混合
- 蒸馏方法：response-level KD（最简单）→ logit-level KD（需要同架构）→ feature-level KD（对齐中间层）

**阶段 B — 跨域蒸馏（迁移到推荐/视频）**：
- 用阶段 A 蒸馏得到的 3B 模型，在新领域做 few-shot 微调
- 验证蒸馏是否提升了跨域迁移能力（vs 直接从 base model 微调）
- 新领域数据：商品图文匹配（推荐）、短视频摘要（视频理解）

**工作量**：阶段 A 2-3 天，阶段 B 额外 2 天

**求职话术**："我构建了一个两阶段蒸馏 pipeline：先在自动驾驶域用 7B teacher 蒸馏 3B student，然后验证蒸馏后的 student 在电商图文匹配任务上的 few-shot 迁移能力比直接微调高 X%。这说明蒸馏不仅压缩了模型，还提升了跨模态表征的泛化性。"

---

### 方向 5：LoRA 变体与高效微调方法对比
**动机**：LoRA 只是 PEFT 的一种。面试时如果被问"为什么选 LoRA 而不是其他方法"，需要有实验支撑。同时这个方向天然适用于所有下游场景。

**具体做法**：
- 在 DriveLM 上对比：LoRA / QLoRA / DoRA / LoRA+ / rsLoRA / AdaLoRA（自适应 rank）
- 控制变量：相同参数预算（如都是 10M 可训参数），对比收敛速度和最终精度
- 额外实验：不同 rank（4/8/16/32/64）的 scaling law
- 分析哪些层最重要：只训 attention vs 只训 MLP vs 全部

**工作量**：2-3 天（大部分是改 config 重跑）

**求职话术**："我在 VLM 微调中系统对比了 6 种 PEFT 方法。发现 DoRA 在相同参数预算下比 LoRA 收敛快 15%，而 AdaLoRA 自适应分配 rank 能在参数减半的情况下保持 95% 的精度。"

---

## 第三层：场景拓展（证明方法的通用性）

### 方向 6：多模态推荐 — 商品图文理解
**动机**：多模态推荐是VLM落地最快的商业场景之一。电商、短视频、广告投放都需要理解"图片+文字"的联合语义。这个方向让你的 efficient VLM 能力从自动驾驶扩展到推荐系统。

**具体做法**：
- 数据集：Amazon Reviews（商品图+评论）或 Shopee 商品匹配数据集
- 任务：
  - 图文匹配：给商品图+标题，判断是否匹配（二分类）
  - 商品描述生成：给商品图，生成卖点描述
  - 跨模态检索：图搜文 / 文搜图
- 用方向 1 训好的 LoRA 作为起点，做 continual fine-tuning
- 关键：对比 Qwen2.5-VL-3B vs CLIP vs BLIP2 在推荐任务上的效率-精度 tradeoff

**工作量**：2-3 天

**求职话术**："我把 efficient VLM 的方法论从自动驾驶迁移到了电商推荐场景。用 Qwen2.5-VL-3B + LoRA 做商品图文匹配，在 X 数据集上达到了 CLIP-Large 的精度，但推理速度快 Y 倍，因为小模型 + token 压缩的双重优势。"

---

### 方向 7：视频理解 — 时序多模态推理
**动机**：视频理解是多模态的终极场景——既有空间信息（每帧图像），又有时序信息（帧间变化）。这个方向把你的 token 压缩能力用到最极致的场景，因为视频的 token 量是图像的数十倍。

**具体做法**：
- 数据集：ActivityNet-QA / NExT-QA / MVBench（视频 QA benchmark）
- 关键挑战：一个视频抽 8-16 帧，每帧几百个 visual token → 总共数千个 token → LLM 爆炸
- 做法 1（Efficient 角度）：用方向 2 的 token 压缩，每帧只保留 16-32 个 token，8 帧也只有 128-256 个 token
- 做法 2（时序建模角度）：在压缩后的帧 token 之间加 temporal attention 或 temporal pooling
- 对比：均匀抽帧 vs 关键帧选择（用 CLIP score 选信息量大的帧）

**工作量**：3-5 天

**求职话术**："视频理解中每帧数百个 visual token、多帧叠加后 token 总量爆炸。我把图像场景下验证过的 token 压缩策略扩展到视频，配合关键帧选择，在 MVBench 上用 3B 模型达到了 7B 模型的 90% 精度，推理速度快 3 倍。"

**为什么重要**：视频 VLM 的效率问题比图像更突出 10 倍，你的 token 压缩工作在这里有最大的发挥空间。

---

### 方向 8：Grounded VLM — 视觉定位 + 语言理解
**动机**：纯 QA 输出文本，但实际部署（自动驾驶、推荐中的商品定位、视频中的目标追踪）都需要输出坐标。DriveLM 数据自带 bbox 标注，改动最小。

**具体做法**：
- 改造 QA 格式，让模型输出结构化坐标：`<bbox>x1,y1,x2,y2</bbox>`
- 训练 VLM 同时做 referring（给描述找目标）和 grounding（给目标出描述+坐标）
- 用 DriveLM 的 `key_object_infos` 中的 2d_bbox 作为训练标签
- 额外：在推荐场景中做商品区域定位（给出商品在图中的位置）

**工作量**：2-3 天

**求职话术**："通用 VLM 只输出文本，但部署场景需要精确空间定位。我扩展了微调数据格式支持 bbox 输出，在 DriveLM 上实现了 grounded driving QA，同时验证了这个能力可以迁移到电商场景的商品定位。"

---

## 第四层：系统工程（Serving & 部署）

### 方向 9：VLM 推理加速与多 LoRA Serving
**动机**：有了 efficient 模型还不够，需要高效部署。这个方向展示你的工程能力，对大厂 infra 岗和自动驾驶部署岗是硬需求。

**具体做法**：
- 用 vLLM / SGLang 部署 Qwen2.5-VL + LoRA
- 实现 multi-LoRA serving：一个 base model 热切换多个 adapter（自动驾驶 / 推荐 / 视频）
- Benchmark：throughput、latency、首 token 延迟、并发用户数
- 对比：HF naive inference vs vLLM vs SGLang
- 额外：量化部署（GPTQ/AWQ）对 LoRA 微调模型的影响

**工作量**：1-2 天

**求职话术**："我用 vLLM 部署了 Qwen2.5-VL，实现了 LoRA adapter 热切换来服务自动驾驶、推荐、视频理解三个场景。单 base model + 3 个 LoRA 的方案比部署 3 个独立模型节省 70% 显存，throughput 相比 HF inference 提升 5 倍。"

---

## 优先级与时间规划

| 优先级 | 方向 | 核心价值 | 工作量 | 覆盖场景 |
|--------|------|----------|--------|----------|
| **P0** | 1. DriveLM LoRA | 基座，跑通全流程 | ✅ 已在做 | 自动驾驶 |
| **P0** | 2. Token 压缩 | Efficient 核心亮点 | 3-4 天 | 全场景通用 |
| **P0** | 4. 知识蒸馏 | Efficient 第二支柱 | 4-5 天 | 全场景通用 |
| **P1** | 3. 跨模态 Adapter | 展示架构理解深度 | 3-4 天 | 全场景通用 |
| **P1** | 7. 视频理解 | Token 压缩最佳验证场 | 3-5 天 | 视频理解 |
| **P1** | 5. PEFT 方法对比 | 微调方法论深度 | 2-3 天 | 全场景通用 |
| **P2** | 6. 多模态推荐 | 场景拓展到推荐 | 2-3 天 | 推荐系统 |
| **P2** | 8. Grounded VLM | 输出坐标，改动小 | 2-3 天 | AD + 推荐 |
| **P2** | 9. 推理加速 | 工程能力展示 | 1-2 天 | 全场景通用 |

---

## 推荐执行路线

### 核心主线（必做，2-3 周）

```
方向 1（DriveLM LoRA）    ← 已完成/进行中
       ↓
方向 2（Token 压缩）      ← 核心创新点
       ↓
方向 4（知识蒸馏）         ← 第二个创新点
       ↓
方向 9（推理加速部署）     ← 工程闭环
```

这条线走完，你有一个完整的故事：**微调 → 压缩 → 蒸馏 → 部署**，覆盖了 Efficient VLM 的全栈。

### 深度支线（选做，展示理解深度）

```
方向 3（跨模态 Adapter）   ← 证明你理解 VLM 内部机制
方向 5（PEFT 对比）        ← 证明你的方法选择有实验依据
```

### 场景拓展支线（选做，拓宽面）

```
方向 6（多模态推荐）       ← 把 Efficient VLM 能力迁移到推荐
方向 7（视频理解）         ← Token 压缩在视频上的极致验证
方向 8（Grounded VLM）     ← 最小改动获得坐标输出能力
```

---

## 面试叙事框架

> "我的研究方向是 **Efficient VLM**——如何让视觉语言模型在保持能力的同时变得更小、更快、更容易部署。
>
> 我从自动驾驶场景切入（方向 1），在 DriveLM 数据集上微调 Qwen2.5-VL-3B，建立了 baseline。然后从两个维度做效率优化：**token 压缩**（方向 2）减少视觉 token 数量降低计算开销，**知识蒸馏**（方向 4）把大模型能力迁移到小模型。
>
> 为了验证这些方法的通用性，我把同一套技术栈迁移到了电商推荐的图文匹配（方向 6）和视频理解（方向 7），证明 efficient VLM 的方法论是跨场景的。
>
> 最后，我用 vLLM 实现了 multi-LoRA serving（方向 9），一个 base model 通过热切换 adapter 服务多个场景，完成了从算法到部署的闭环。"

这个叙事的强项在于：**不是零散的项目堆砌，而是一条有逻辑的技术主线**。
