# Full-Modal AD-VLA Research & 5D Parallelism Strategy

Survey of open-source autonomous-driving Vision-Language-Action (VLA) approaches
beyond pure RGB, plus a deep dive on how 5D parallelism (DP / TP / SP / PP / CP)
becomes load-bearing once we move from RGB-only to a full-sensor stack.

Audience: interview prep + future engineering plan. Reflects analysis on the
24K nuScenes + 3B Qwen2.5-VL pilot we ran (R1' / A.0 / A.1 done; A.2/A.3 in
flight), specifically the observation that **1-cam → 3-cam gave no L2 win at
this scale**, which is the empirical motivation for everything below.

---

## 1. Open-source multi-modal AD-VLA landscape

| Work | Modalities | Fusion mechanism | Data scale | Code |
|---|---|---|---|---|
| **UniAD** (CVPR'23 Best) | 6-cam → BEV | Sparse query + BEV decoder (perception→prediction→planning unified) | nuScenes 28K | https://github.com/OpenDriveLab/UniAD |
| **VAD** (CVPR'23) | cam + HD map | Vectorized map representation | nuScenes | https://github.com/hustvl/VAD |
| **DriveLM** (Shanghai AI Lab) | 6-cam only | Graph-of-thought QA + standard VLM | nuScenes-QA 70K | https://github.com/OpenDriveLab/DriveLM |
| **DriveMLM** (Shanghai AI Lab'24) | cam + LiDAR + map | Token-stream concat (each modality independent encoder → projector) | nuScenes + private | https://github.com/OpenGVLab/DriveMLM |
| **OmniDrive** (NVIDIA'24) | 6-cam + LiDAR + map | Q-Former-style cross-attention pool over multi-modal feature pool | nuScenes-QA + Waymo | https://github.com/NVlabs/OmniDrive |
| **LMDrive** (HK CUHK'23) | cam + LiDAR | Early fusion + LLaMA-2 | CARLA closed-loop sim | https://github.com/opendilab/LMDrive |
| **DriveVLM** (Tsinghua / Li Auto'24) | cam + map + ego | Dual-system (fast deterministic + slow VLM) | nuScenes + 100K+ private | https://github.com/Tsinghua-MARS-Lab/DriveVLM (partial) |
| **Senna** (Tsinghua'24) | cam (VLM) + BEV (planner) | Two-stage: VLM emits high-level intent → BEV planner emits trajectory | nuScenes + 1M private | https://github.com/hustvl/Senna |
| **EMMA** (Waymo'24) | cam + map text | Pure vision + text, no LiDAR; Gemini-based | Waymo 100M+ | closed-source |
| **OpenEMMA** (academic clone'24) | cam + map | Mirrors EMMA design | nuScenes | https://github.com/taco-group/OpenEMMA |
| **DriveGPT4** (HKU'23) | cam + language | Single-view + chain-of-thought | small | https://github.com/PJLab-ADG/DriveGPT4 |
| **GPT-Driver** (USC'23) | cam + CoT planning text | Camera → caption → LLM plans | nuScenes | https://github.com/PointsCoder/GPT-Driver |

**Closed but influential**: Tesla FSD v12 (end-to-end neural), Waymo Driver, Wayve LINGO-1/2.

---

## 2. Fusion mechanism taxonomy

Five archetypes; each work above maps onto one of these.

### A. Token-stream concatenation (DriveMLM, OmniDrive variant)
Each modality has its own encoder; outputs are projected to LM hidden dim and
concatenated as a contiguous visual-token block in the LM context.

```
LiDAR  → PointPillars → BEV feat → Linear projector → 64 tokens ┐
6-cam  → ViT          → 2x2 merger → Linear projector → 768 ts  ├─ concat → LM context
HD Map → vector → render BEV → ViT → Linear projector → 64 ts  ┘
3D box → JSON-serialize into prompt text                        ─ prompt prefix
```

- ✓ Flexible; modality dropout via masked tokens is natural
- ✗ Visual-token count explodes (768 + 64 + 64 = 896 visual per frame × N frames)

### B. BEV unification (UniAD, VAD)
All sensors project to a shared bird's-eye-view grid; the BEV image is then
encoded by a single ViT. Map sits naturally in BEV.

- ✓ Compact, naturally aligned
- ✗ Loses fine-grained camera detail (small distant objects washed out)

### C. Cross-modal Q-Former / Perceiver (OmniDrive, DriveLM-Agent)
A fixed set of learned queries (e.g. 64–128) cross-attends to a flat pool of
multi-modal features. The query stack is what the LM sees.

- ✓ Fixed query count (no token explosion); modality dropout extremely natural
- ✗ Extra learned params (~80M for 64q × 6L Q-Former)

### D. Sparse object query (BEVFormer, Sparse4D, UniAD perception head)
3D object queries cross-attend directly to multi-view + LiDAR features; the
queries themselves are the world-state representation.

- ✓ Object-level reasoning; UniAD uses this for the perception→planning pipeline
- ✗ Detection-oriented; awkward to expose as a chat interface

### E. Tool-use / staged perception (LMDrive, GPT-Driver, Senna)
LLM never sees raw sensors. A perception module (BEV or 2D) emits structured
scene descriptions → fed to LLM as text. LLM only does planning.

- ✓ Clean decoupling; LLM doesn't need to know about sensor formats
- ✗ Loses end-to-end gradient; perception errors propagate as text errors

**Industry convergence (2024)**: A or C for full-modal training; E for fast
deployment on edge HW (perception model is small + LLM is small + decoupling
allows different update cadences). Senna's dual-system is the hybrid trend.

---

## 3. Modality dropout training (robustness)

The "always-on multi-modal" assumption breaks in production: HD maps are
missing in unmapped regions; radar/LiDAR drop occasionally; one camera can be
blocked by mud.

**Standard fix**: during training, randomly mask one or more modality streams
per sample (10–30% probability). Encoder output for the dropped modality is
replaced by zero tokens or a small set of learned "empty" tokens.

Inference-time fallback: the missing modality slot uses the same empty token,
and the model has learned this distribution.

References: DriveMLM §3.3 explicitly trains with modality dropout; M3DETR uses
similar idea for 3D detection robustness.

On our pipeline this maps cleanly: the dataloader rolls a per-modality
Bernoulli mask, and the projector's input either runs the encoder or replaces
its output with zeros / a learned bias before the modality's projector → LM
inlining step.

---

## 4. Training cost on 24K nuScenes — empirical estimates

Numbers are extrapolations from our A.1/A.2 step-time measurements
(1.85–4.5 s/step on 8 × 5090) plus encoder FLOPs from MMDetection3D /
PointPillars / VAD published benchmarks.

| Configuration | est. s/step | wall (3 epoch ~2240 steps) | Per-rank GPU mem |
|---|---|---|---|
| **Current A.1** (1-cam × 4f, Q-Former) | ~1.9 | ~1.5 h | 26 GB |
| **Current A.2** (1-cam × 4f, PixelShuffle, +50% pixels) | ~3.1 | ~2.0 h | 28 GB |
| + LiDAR (PointPillars BEV → 64 tokens) | ~3.5 | ~2.3 h | 29 GB |
| + HD map (vector → BEV render → 64 tokens) | ~4.0 | ~2.5 h | 30 GB |
| + 6-cam (vs 1-cam) | ~5–6 | ~3.5–4 h | 32–34 GB ⚠️ |
| + 3D bbox (JSON-serialize into prompt) | ~3.5 | ~2.5 h | 28 GB (LM seq grows) |
| **Full-modal (6cam + LiDAR + map + bbox), 4 frames** | ~8–10 | ~6–7 h | ~38 GB **OOM on 5090 32GB** — needs AC or grad-accum |
| **Full-modal × 8 frames** (longer history) | ~14–16 | ~10–12 h | ~46 GB **definitely OOM**, requires CP / PP |

**Encoder pretraining cost (if not using off-the-shelf)**:
- PointPillars 3D detection from scratch on nuScenes: ~2–3 days × 8-GPU. NOT viable in our budget — **must use mmdetection3d / CenterPoint pretrained checkpoints**.
- HD-map render baseline: cheap (~30 lines of nuScenes API code).

---

## 5. Qwen2.5-VL architectural fit — what works, what doesn't

| Capability | Native in Qwen2.5-VL | What's needed for full-modal |
|---|---|---|
| 2D spatial reasoning (M-RoPE h/w dims) | ✓ strong | — |
| Temporal reasoning (M-RoPE t dim) | ✓ handles video | — |
| **3D point-cloud / BEV spatial reasoning** | ✗ no native | External LiDAR encoder (PointPillars / CenterPoint, frozen) + projector adapter — same pattern as the 2x2 patch merger |
| **HD map vector understanding** | ✗ no native | Either: (a) render BEV image and let ViT eat it, OR (b) serialize lane segments as text (token-heavy) |
| **Cross-modal alignment (LiDAR ↔ text)** | ✗ NOT pretrained | Stage-1 alignment pretraining on LiDAR-text pairs (100K–1M pair scale) — this is what we lack |
| **Modality dropout robustness** | ✗ no native | Bake into training loop (Bernoulli mask in dataloader) |
| **Long-context for full-modal sequence (8K–32K tokens)** | partial | M-RoPE OK in theory; CP (Context Parallel) needed at compute level — see §6 |

**The load-bearing gap**: cross-modal alignment. Qwen2.5-VL was pretrained on
image-text pairs. Adding a new modality (LiDAR) without alignment pretraining
is asking the model to learn the encoding from scratch on the downstream task.
At our 24K scale this is hopeless — already proven by **1-cam → 3-cam giving
no L2 gain** in our A.0 experiment (more vision modality, same model, no win).
Full-modal would fail harder.

---

## 6. 5D parallelism for multi-modal VLA training

This is where things get interesting. RGB-only at 4 frames + 1 cam fits in DP
on 8 GPUs. Full-modal at 8 frames + 6 cams + LiDAR + map at 7B+ scale does
**not**. The five dimensions of parallelism each map onto a different bottleneck.

### 6.1 The bottleneck shifts as modalities are added

| Setup | Seq len (visual + text) | Param count | Bottleneck |
|---|---|---|---|
| 1-cam × 4f, 3B | ~600 tokens | 3B | activation memory (DP-only fine) |
| 6-cam × 4f, 3B | ~3400 tokens | 3B | activation + attention compute |
| 6-cam × 4f + LiDAR + map, 3B | ~4500 tokens | 3B + 15M encoders | attention compute (quadratic in seq) |
| 6-cam × 8f + full-modal, 7B | ~9000 tokens | 7B + 15M | param mem **and** attention compute |
| 6-cam × 16f + full-modal, 32B | ~18000 tokens | 32B + 15M | everything |

### 6.2 Each dimension's role

**DP (Data Parallel)** — the trivial one
- Replicate the model on each rank; each rank does 1/N of the batch.
- Caps at activation+param-memory fit on one rank.
- Effective up to ~3B model + 4 frames + 1 cam on 32 GB cards (what we did).

**TP (Tensor Parallel)** — shard inside a layer
- Split attention heads and MLP hidden dim across ranks; all-reduce after each
  attention/MLP block.
- Standard for 7B+ LMs (Megatron-style).
- Heavy NCCL traffic — best within a node (NVLink).
- For multi-modal: only matters for the LM stages. Vision encoders are
  small enough to live within DP.

**PP (Pipeline Parallel)** — split by layers
- Different layer ranges live on different ranks; activations flow through a
  pipeline of micro-batches.
- Useful for 7B+ models that don't fit param + optimizer state on one rank.
- **Multi-modal twist**: each modality encoder can be its own *parallel* PP
  stage 0 (vision-encoder rank, LiDAR-encoder rank, map-encoder rank all feed
  the same downstream LM PP stage 1). Saves vision-encoder compute from
  blocking the critical path.

**SP (Sequence Parallel)** — shard the sequence within layers
- Within a TP region, split the sequence dim for LayerNorm + dropout + the
  non-matmul parts of MLP, where the activations are the largest tensors.
- Reduces activation memory roughly proportional to TP rank count.
- For multi-modal, the long visual-token concatenation makes activations huge
  → SP becomes more valuable than in pure-text LMs.

**CP (Context Parallel)** — shard the sequence across attention
- Split the sequence into chunks; each rank holds 1/N of K, V; attention
  computation rotates K, V across ranks (Ring Attention) or builds a tree.
- O(N²) attention compute is divided by CP rank count.
- **Becomes essential when seq len > 16K** — exactly where multi-modal long-
  history VLA lives.
- For our 1-cam × 4f at ~600 tokens, CP is wasted overhead.
- For 6-cam × 16f + full-modal at ~18K, CP is the *only* way to fit attention.

### 6.3 Recommended 5D layout for full-modal VLA at scale

Concrete example: **7B Qwen2.5-VL-like model + 6 cam + LiDAR + map + 8-frame
history** on a 64-GPU cluster (8 nodes × 8 GPUs).

```
Total ranks       : 64
DP × TP × PP × CP : 4 × 2 × 4 × 2 = 64
SP                : enabled (uses TP comm group, no extra rank dim)
```

Layout:
- **PP stage 0**: vision + LiDAR + map encoders (in parallel sub-stages, same
  rank slice). Each modality runs concurrently on 2 TP × 2 CP × 4 DP = 16
  ranks. Output: per-modality projected tokens.
- **PP stage 1–3**: LM transformer blocks (24 layers / 3 stages = 8 layers each
  for a 7B model). TP=2 × CP=2 × DP=4 = 16 ranks per stage.
- CP chunks the ~9K-token visual-token block across 2 ranks → 4.5K per rank
  for attention.
- SP shards the activations in the non-matmul ops of each TP region.

This layout falls naturally out of torchtitan's mesh API
(`init_device_mesh(("dp","pp","tp","cp"), (4,4,2,2))`).

### 6.4 What 5D does **not** solve

- **Cross-modal alignment**: still needs pretraining data, not compute. 5D
  lets you *afford* large-scale alignment pretraining (100K–1M pair); it
  doesn't replace it.
- **Modality dropout robustness**: dataloader-level, orthogonal to parallelism.
- **Encoder pretraining cost**: PointPillars / map encoders are tiny relative
  to the LM and aren't the parallelism target.

### 6.5 Why we punted 5D for the current 24K experiments

- Our 3B + 1-cam × 4f setup fits in DP on 8 × 5090. CP/PP overhead would lose
  more than it saves.
- 24K data is too small to need 7B+; 3B is already over-parametrized at this
  scale (see A.1 LM-loss-better-but-L2-worse evidence).
- For deployment on RTX 5090 → DRIVE Thor (single-device inference), TP/PP
  at *training* time doesn't change the production engine.

**5D becomes the right call at the next step up**: fleet-scale data (≥100K
samples) + 7B–32B model + full-modal + 8–16 frame history. That's the regime
where the bottleneck shifts from "can I fit one sample on one GPU" to "can I
afford the O(N²) attention on a 16K-token multi-modal context".

---

## 7. Verdict for our current pipeline

**Stay on RGB-only depth for the 24K demo**. Specifically:
- finish A.1 / A.2 / A.3 projector ablation (proves the fusion-bottleneck design space)
- training-free 2-D compression Pareto on the winning ckpt (proves the prefill-efficiency frontier)
- NVFP4 + TRT-LLM engine on 5090 (proves the prod-grade deployment story)

**Full-modal + 5D is the natural next chapter**, not a 24K demo. The story to
tell an interviewer:

> "At 24K scale we proved the projector design space and the prefill
> efficiency frontier on RGB. The same trunk extends to LiDAR / map / 3D bbox
> via additional encoders into the projector — but cross-modal alignment
> needs Stage-1 pretraining on 100K–1M pair data, which is a fleet-scale
> capex, not a 24K demo. Once you commit to that scale, the parallelism
> profile shifts: DP-only stops working when sequence length hits 16K under
> full-modal long-history, and CP + SP become load-bearing. PP+TP layout for a
> 7B+ multi-modal LM is standard Megatron territory; the multi-modal twist is
> that each modality encoder lives in its own PP stage 0 sub-pipeline so the
> vision compute doesn't block the LM critical path."

This answers both the "have you done multi-modal?" and "do you understand
distributed?" lines of questioning, without requiring us to actually run a
full-modal 7B training that would fail empirically at our data scale.

---

## 8. Related repos worth citing in interview

- **torchtitan** — 5D parallelism reference impl (https://github.com/pytorch/torchtitan)
- **OmniDrive** — multi-modal cross-attention pool (closest to our projector-replacement series)
- **DriveMLM** — token-stream concat + explicit modality dropout
- **UniAD** — BEV unification (the perception→prediction→planning gold standard)
- **mmdetection3d** — LiDAR encoder zoo (PointPillars, CenterPoint, BEVFormer)
- **Megatron-LM** — TP+PP+SP reference (older but canonical)
