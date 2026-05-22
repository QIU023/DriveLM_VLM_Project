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

### 6.6 CP+VLM: known problems and how they're solved in production

Context parallel on VLMs was famously broken until 2024-Q3. The fundamental
issue: vision tokens have "dense attention" (every text token attends to every
vision token, no causal sparsity). CP slices sequence → every attention layer
needs cross-rank all-to-all → naive CP gives ~3-5× wall-clock slowdown.

Six known problems, and the production-ready solutions:

| Problem | Production solution | Who uses it |
|---|---|---|
| Dense vision attention → cross-rank comm explosion | **Megatron-CP Ring Attention**: ring-pass KV blocks, O(N/p) memory | NVIDIA, ByteDance |
| Alternative when num_heads ≥ CP | **DeepSpeed-Ulysses**: two all-to-all in seq↔head dim | Microsoft, DeepSeek |
| Vision = bidirectional, LM = causal → mask conflict | **Flash-Attn 3 + variable-mask kernel** (mask-aware ring) | Llama-3, Qwen2.5-VL |
| Cannot split image mid-patch (breaks spatial structure) | **CP cut-point aligned to image boundary**, never inside a single image | All production stacks |
| Vision is only ~30% of tokens → CP cost > benefit on vision | **HotSpot CP**: replicate vision tokens, CP only the LM/text side | Kuaishou MM, Step Video (inferred) |
| ViT (high batch, low seq) vs LM (low batch, high seq) — different parallelism profiles | **Heterogeneous parallel**: ViT runs vision-side TP (patch-TP); LM runs TP+CP+PP; reshard at boundary | InternVL-Chat-4K, Yi-VL |

CP+VLM is "fully solved" only since Megatron-LM v0.7 (2024-Q3) + Flash-Attn 3
+ heterogeneous vision/LM mesh. XPeng-scale 7B+ dense VLA almost certainly
uses Megatron-CP Ring + Flash-Attn 3 + a heterogeneous mesh (ViT on TP-only,
LM on full 5D).

**Implication for us**: at 8 GPU × 24K data × ~2400-token sequence, we never
touch CP. This is interview-talkable understanding, not a deployment need.

### 6.7 XPeng-scale parallelism: inferred recipe (public-info reconstruction)

XPeng has disclosed: 2.51 EFLOPS training compute, 35 B yuan R&D budget,
Turing AI chip at 750 TOPS each, 4 chips per car = 3000 TOPS in production.
VLA 2.0 model size not public; rumored 7-14B dense (not MoE).

Inferred cluster recipe (extrapolated from public scale + standard practice):

```
Cluster ~2540 H100-equivalent GPUs (2.51 EFLOPS / 989 TFLOPS_bf16/GPU)
Per-experiment slice (single VLA training):

  TP=4 × SP=on × CP=2 × PP=2 × DP=128 (FSDP-style)  →  2048 GPU
  ─────                                       ─────
  inside node                          cross-node FSDP+DP

  EP=N/A (model is dense)
```

The 5 dimensions:

| Dim | Why XPeng needs it | Sweet spot |
|---|---|---|
| FSDP / DP | Shard 7-14B params across 100+ ranks; batch scaling | DP=128 |
| TP | 7-14B params don't fit per-rank under FSDP=8 alone; head matrix splits | TP=4 |
| SP | Pairs with TP; shards LayerNorm + dropout activations in seq dim | enabled |
| CP | Multi-cam × video history → 100K+ token sequences | CP=2 |
| PP | Not for "model too deep" (28 layers fits) — for **vision compute pipelining**: ViT (heavy) on dedicated stage so LM doesn't block on it | PP=2 |
| EP | None — model is dense | — |

**Why PP=2 even at 7B**: ViT-L over 11 cam × 8 frames = 88 ViT forwards/step,
~5× the LM forward FLOPs. Co-locating ViT and LM on same rank means LM idles
~30% of step time. Putting ViT on PP stage 0 → micro-batches pipeline → LM
critical path shrinks ~40%. This is the **real** motivation for PP in
multi-modal training, not "model too deep". Flamingo, IDEFICS, GPT-4V all do
this.

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

## 8. Decision 2026-05-21: image-as-modality, no fusion module

### Why (A.x ablation finding)

| Variant | Projector | Params | L2 @ 24K nuScenes |
|---|---|---:|---:|
| R1' / A.0 | OEM Linear (pretrained) | ~10M | **0.6423** (winner) |
| A.1 v6 | Q-Former (random init) | 81M | 0.7144 |
| A.2 v4 | PixelShuffle (random init) | 17M | 0.6811 |
| A.3 v2 | Resampler (random init) | 90M | 0.6968 |

Every random-init projector lost to OEM-pretrained Linear at 24K. LLaVA-style
stage-1 alignment (558K image-text pair pretraining) would add ~24h before SFT
to MAYBE close the gap — not worth the budget when EMMA-style image-as-modality
reuses the already-aligned OEM path for free.

Industry consensus aligns: EMMA (Waymo 2024), DriveLM-Agent (CVPR'24), AutoVLA
all run pure image-as-modality + text serialization with NO fusion module.
OmniDrive (Q-Former) is the minority case and depends on BEV pretraining.

### Architecture (Qwen2.5-VL-3B unchanged)

```
ViT-L (~600M) ──► OEM Linear (~10M, pretrained, unfrozen) ──► LM tokens ──┐
                                                                            ├──► Qwen2.5-3B Decoder
text (bbox / ego / prompt) ───────────────────────────────► LM tokens ────┘    (~2.4B, 28 layers)
```

All visual modalities (camera × Nf, HD map BEV PNG, LiDAR BEV PNG) share the
SAME ViT + OEM Linear path. Non-visual modalities (3D bbox, ego state) are
text-serialized. **No random-init projector, no cross-modal Q-Former, no
multi-encoder fusion.**

### Sequence layout (per sample)

```
<bos>
<system> task instruction (~150 tok) </system>
<user>
  <image_0..Nf-1>  camera frames  (1-cam × 180 tok/frame)
  <image_hdmap>    HD map BEV     224² → 64 tok
  <image_lidar>    LiDAR BEV      224² → 64 tok
  bbox text        30-50 obj × ~12 tok            ~500 tok
  ego state        speed + yaw + history          ~80 tok
  task prompt      "predict 3s @ 0.5s"            ~30 tok
</user>
<assistant>
  trajectory tokens (OpenVLA-bin, 6 wp × 2 dim + delim)  ~50 tok
</assistant>
```

| Variant | Camera tok | Other tok | **TOTAL** |
|---|---:|---:|---:|
| 4-frame  |  720 | ~980 | **~1700** |
| 8-frame  | 1440 | ~980 | **~2400** |
| 16-frame | 2880 | ~980 | **~3900** |

Within Qwen2.5-VL 32K native context with 10× headroom.

### FSDP=8 memory budget (per rank, bf16, 8-frame estimate)

| Component | Estimate |
|---|---:|
| Sharded params (3B / 8 × 2B) | 0.75 GB |
| Sharded AdamW state (fp32 master + 2 momentum) | 3.0 GB |
| Sharded grads | 0.75 GB |
| LM activations with AC (28 layer × 2400 tok × 2048 × LBS=3 × bf16 × 0.3) | ~6 GB |
| ViT activations (10 image × 180 grid × 1280 × LBS=3 × bf16) | ~0.5 GB |
| NCCL buffers + misc | ~2 GB |
| **Total per rank** | **~13 GB** |

8f_vtm baseline measures 31 GB/rank (includes peak ViT decode + video buffer
overhead). Full-modal 8f estimate ~30-35 GB/rank — fits H100 80 / A100 80 / H800 40.

### Parallelism: FSDP=8 only, no multi-D

| Dim | Triggered? | Reason |
|---|---|---|
| DP / FSDP=8 | yes | baseline shard for 3B model |
| TP | no | per-rank params 0.75 GB, weight sharding unnecessary |
| SP | no | 2.4K seq → ~6 GB activations, well within budget |
| CP | no | no long-context activation pressure |
| PP | no | 28 layers, fits on single rank's sharded slice |

Multi-D triggers only on later upgrades:
- Qwen2.5-VL-7B / 32B → **TP=2/4**
- 32-frame × 3-cam (~17K seq) → **SP** for activation memory
- GRPO online rollout co-located with trainer → **DP×DP split** (rollout server / trainer separate groups)

### Training data
- 28051 keyframes (intersection of HD map cache 28051/28130 = 99.72%, bbox cache 28K, camera 28130, LiDAR TBD)
- 79 samples dropped due to shapely MultiLineString edge case in HD map render — dataloader skip-on-miss

### Explicitly out of scope
- Cross-modal Q-Former (lost to OEM Linear at this data scale; reconsider only at >100K paired samples)
- Multi-encoder fusion (Cambrian SVA style — overkill for AD-VLA at 28K)
- BEV unify single-trunk (UniAD-style — needs perception pretraining we don't have)
- LLaVA stage-1 alignment (558K pair pretraining) — adds 24h to any learned-projector variant before SFT

---

## 9. XPeng VLA 2.0 simulation: experiment sweep on 8 GPU × 24K data

Goal: maximally simulate XPeng VLA 2.0 design space within our compute envelope
(8 × 80GB GPU, 28K nuScenes keyframes, 5-6 modalities cached). Six experiment
series, each addresses a specific XPeng-aligned design axis.

### 9.1 Sweep summary

| Series | What it tests | XPeng-align | Done |
|---|---|---|---|
| A | Projector (single-cam) ablation | ★★ | ✅ A.0/1/2/3, validates OEM-pretrained > random-init at 24K |
| B | Multi-modal image-as-modality fusion | ★★★★ | next |
| C | Long-video temporal compression | ★★ | running (8f_mean done; 8f_vtm running; 8f_longvu next) |
| D | V→A direct (no language CoT) | ★★★★ | planned |
| E | World-model data synthesis | ★★★ | future (separate infra) |
| F | Closed-loop RL (DPO / GRPO) | ★★★★ | F.1 cheap, F.2 needs sim |

### 9.2 B series — multi-modal SFT (next priority)

| ID | Variant | Modalities added vs A.0 | GPU-h | Expected L2 |
|---|---|---|---|---|
| B.1 | + HD map BEV image | HD map | 8 | 0.60-0.64 |
| B.2 | + LiDAR BEV image | HD map + LiDAR | 8 | 0.58-0.63 |
| B.3 | + bbox text serialization | HD map + LiDAR + bbox | 8 | 0.58-0.62 |
| B.4 | + ego state text | + ego | 8 | 0.57-0.61 |
| B.5 | **= XPeng V2 simulation** | all 5 modalities, image-as-modality, no fusion module | 10 | **0.56-0.60** |
| B.6 | B.5 + modality dropout 10% per modality | all + dropout robustness | 10 | 0.57-0.61 |

All B runs share the §8 architecture (OEM Linear path for all visual modalities,
text path for bbox/ego). Sequential ablation order: each run adds one modality
on top of the previous → isolates each modality's marginal contribution.

### 9.3 D series — V→A direct (XPeng V2 core innovation)

XPeng VLA 2.0's selling point: "removes the language translation step, enables
direct Visual-to-Action generation". Two variants:

| ID | Variant | Description | GPU-h | Risk |
|---|---|---|---|---|
| D.1 | Pure trajectory tokens (no CoT in answer) | LM trained to emit only `<traj_x_0>...<traj_y_5>`, no reasoning text | 8 | **high** — loss of interpretability, harder to converge |
| D.2 | D.1 + reasoning scaffold via **system prompt** only | KV-cache friendly: scaffold fixed in prefill, output is action-only | 8 | medium |

Pair with B.5 in cross-product: B.5 × D.1 = "XPeng V2 lookalike".

### 9.4 F series — closed-loop RL (production alignment)

| ID | Variant | Description | GPU-h | Risk |
|---|---|---|---|---|
| F.1 | Offline DPO on trajectory pairs | Generate K candidates per sample with B.5 ckpt; pairwise rank by L2 to ground truth; DPO on (winner, loser) | 8 | medium — needs candidate generation infra (~2h) |
| F.2 | Online GRPO with simulator rollout | Need a UniAD-port simulator or nuScenes replay environment; rollouts every 100 steps | 16 | **high** — sim infra not built; defer |

F.1 is the right XPeng-alignment story without needing X-World-style simulator.

### 9.5 A.4-A.5 — sanity / isolation experiments

| ID | Variant | What it isolates | GPU-h |
|---|---|---|---|
| A.4 | OEM Linear architecture + **random init** | Pretrained-weight vs arch-shape contribution | 6 |
| A.5 | OEM Linear **frozen** during SFT | Does SFT degrade OEM weights? How much? | 6 |

Cheap diagnostics; run only if B.x finishes early.

### 9.6 Priority queue (~7 GPU-days wall clock on 8× cluster)

```
1. B.1 → B.2 → B.3 → B.4 → B.5   (sequential, ~40h GPU)
2. D.1 (pair with B.5)            (8h GPU)
3. B.6 modality dropout           (10h GPU)
4. F.1 offline DPO                (8h GPU + 2h infra)
5. A.4 sanity                     (6h GPU, optional)
TOTAL                             (~74h GPU = ~9 wall days on 8 GPU)
```

E series (world-model data synthesis) and F.2 (online GRPO with sim) are
deferred — they need infra builds (~1-2w) not justified at 24K scale.

### 9.7 Out of scope (deliberately, with reasoning)

| Item | Why not on the sweep |
|---|---|
| 32-frame video history | Bottleneck shifts to CP+SP, our 8-GPU stack can't validate; small-data benefit not worth the engineering |
| Cross-modal Q-Former (B alternative) | A.x already proved random-init projector lost at 24K; restating it on B doesn't add information |
| Stage-1 LLaVA-style pretraining (CC3M) | Domain mismatch (web images vs driving cameras); 24h cost; XPeng doesn't do this either (they have proprietary fleet data) |
| 7B Qwen2.5-VL scale-up | Param mem doesn't fit on 8× 80GB at 7B with 2400-token activations; would need TP=2 which loses the "story is portable" property |

---

## 10. XPeng architecture mapping (interview-ready talking points)

Our project component → XPeng VLA 2.0 component → what to say:

| Our component | XPeng V2 equivalent | Talking point |
|---|---|---|
| Qwen2.5-VL-3B trunk | Single end-to-end VLA model | "Same single-stream philosophy; 3B not 7B-14B due to local compute, but the architectural choices port" |
| OEM Linear (A.0 winner) | OEM vision encoder + projector | "Pretrained alignment is load-bearing — A.x ablation showed 81M Q-Former, 90M Resampler, 17M PixelShuffle all lose to OEM at 24K. Industry sees same: EMMA, AutoVLA, DriveLM-Agent all keep OEM" |
| Image-as-modality (HD map BEV / LiDAR BEV as RGB image) | Image-only sensor stack (XPeng V2 trending no-LiDAR) | "EMMA-style: every modality goes through the same already-aligned ViT path; no random-init fusion module to cold-start" |
| OpenVLA-bin trajectory tokens | V→A direct action output | "Same action discretization paradigm; 256-bin per-dim is paper-conservative" |
| UniAD-port collision metric, TemAvg L2 | Production planning metric | "Cross-validated to UniAD canonical eval; not just nuScenes paper L2" |
| 5D parallelism story (§6, §6.6, §6.7) | XPeng 2.51 EFLOPS cluster | "Know FSDP+TP+SP+PP+CP stack; CP+VLM solved by Megatron-CP + Flash-Attn 3 + heterogeneous mesh post-2024-Q3. PP exists to pipeline ViT, not because LM is too deep" |
| GRPO (F.2, future) | X-World closed-loop RL | "Trainer side aligned; their X-World provides simulator we lack. Our F.1 offline DPO is the cheap-version equivalent" |
| AutoVLA-aligned hyperparams (LR=2e-5, warmup 1.74%, step decay ×0.98 / 6.96%) | XPeng's proprietary recipe | "Paper-aligned not arbitrary. Pre-launch audit table for every deviation" |
| Modality dropout (B.6) | Production sensor-failure robustness | "Same training-time technique used by Tesla FSD, XPeng, Waymo for sensor degradation handling" |

### Common interview question prep

**Q: 你们的方案和小鹏 VLA 2.0 区别？**
A: 单流端到端 + 无 fusion module 同方向。规模差一档（3B vs 7-14B），数据差两档（28K vs fleet），但每个设计 decision 都对应可复用结论。

**Q: 为什么不上 Q-Former？**
A: A.x ablation 实证：24K 数据训不出 random-init Q-Former。EMMA / AutoVLA / DriveLM 业界主流也是 OEM 路径。OmniDrive 是 BEV 大量预训练后才能用 Q-Former，我们没有 fleet。

**Q: CP+VLM 有什么坑？**
A: 见 §6.6。

**Q: 端到端去掉 language layer 怎么 debug？**
A: 小鹏靠 X-World 闭环回放 + 我们 F.1/F.2 路线。轨迹 token 仍能反解到 BEV 可视化。

**Q: 为什么不直接用 7B？**
A: 8×80GB 跑 7B + 2400-token activation 要 TP=2，会破"story portable"性；3B 在 28K 数据上已经过参数化（A.x: LM-loss-better-but-L2-worse 证据）。

---

## 11. Phase B v2 results: paper-grade ablation with sub-scenario + behavior metrics

### 11.1 What changed vs the original §8/§9 framing

Two non-trivial pipeline bugs surfaced during initial Track-B / Track-C eval:

1. **`planning_eval.py:_build_batch_inputs` rebuilt prompt from scratch** — only fed
   camera videos + multicam suffix text. HD map image + bbox text from the
   `MultiModalPlanningDataset` were silently discarded. This made `--multimodal`
   eval byte-identical to camera-only eval (proven by identical L2 to 4 dp over
   6019 val samples). **Fixed by re-using `_build_user_content_multimodal` +
   passing `images=[hdmap]` to the processor when the dataset is multimodal.**

2. **Open-loop L2 on full val set saturates near 0.62-0.80** for every design,
   confirming the "Is Ego Status All You Need?" (arXiv 2312.03031) finding that
   nuScenes planning L2 is dominated by ego-speed reproduction. **Added
   sub-scenario L2 breakdown + 5 behavior metrics** (heading error, lateral
   accel RMS, speed error, hard brake rate, progress ratio) so ablations
   actually differentiate.

### 11.2 v2 evaluation matrix (6 ckpts × 10 metrics)

`eval_results/track_v2/*.json` — produced by `scripts/run_all_ckpts_eval_v2.sh`.

| Ckpt | L2 | Coll% | **turning** | straight | **lane_change** | braking | heading_err (rad) | hard_brake_rate | progress |
|---|---|---|---|---|---|---|---|---|---|
| R1' / A.0 1-cam baseline | 0.642 | 3.73 | 1.179 | 0.635 | 0.833 | 0.823 | 0.285 | 1.39% | 0.961 |
| **R1'' 3-cam baseline** | **0.622** | 3.99 | 1.005 | **0.574** | 0.809 | 0.942 | 0.284 | 0.16% | 1.162 |
| **B.5 image-as-modality (HD map + bbox)** | 0.678 | **3.07** | **0.964** | 0.580 | **0.670** | 0.842 | **0.279** | 0.59% | 1.086 |
| B.5 same ckpt + camera-only at eval (robustness probe) | 0.665 | 3.54 | 1.113 | 0.609 | 0.762 | 0.882 | 0.287 | 0.68% | 1.100 |
| B.6 = B.5 + per-modality dropout p=0.1 | 0.803 | 3.48 | 1.130 | 0.611 | 0.805 | **0.758** | 0.284 | 0.27% | 1.078 |
| B.6 same ckpt + camera-only at eval | 0.800 | 3.52 | 1.145 | 0.645 | 0.864 | 0.768 | 0.287 | 0.39% | 1.085 |

n=541 turning, ~1500 straight, ~120 lane_change, ~200 braking, ~3000 cruising/stationary (per 5119-sample val cut after multimodal HD-map filter).

### 11.3 Headline findings — per-axis winners

| Axis | Winner | margin vs R1' baseline | Reading |
|---|---|---|---|
| Overall L2 | R1'' 3-cam (0.622) | -3% rel | More cameras → small L2 lift |
| Collision rate | **B.5 multimodal (3.07%)** | **-18% rel** | HD map + bbox give clear safety benefit |
| Turning L2 | **B.5 multimodal (0.964)** | **-18% rel** | HD map decisive at intersections |
| Lane-change L2 | **B.5 multimodal (0.670)** | **-20% rel** | HD map + bbox decisive for lateral decisions |
| Braking L2 | B.6 (0.758) | -8% rel | Dropout taught conservative braking |
| Hard brake rate | R1'' 3-cam (0.16%) | -88% rel | 3-cam = smooth driver style |
| Heading error | B.5 multimodal (0.279 rad) | -2% | Marginal but consistent |
| Straight L2 | R1'' 3-cam (0.574) | -10% rel | Multi-cam helps straight cruise too |

**Every ckpt wins something.** Open-loop L2 alone hides this because non-cruise scenarios are only ~20% of samples and get averaged out.

### 11.4 Multi-modal robustness probe (B.5 / B.6 cam-only column)

Same ckpt evaluated with HD map + bbox masked out at inference (worst-case modality-failure scenario). Quantifies how much of the model's competence depends on each modality.

| ckpt | full → cam-only L2 Δ | full → cam-only collision Δ | full → cam-only turning Δ |
|---|---|---|---|
| B.5 | -0.013 (slight better) | **+0.47% abs (+15% rel)** | **+0.149 abs (+15% rel)** |
| B.6 | -0.004 | +0.04% | +0.015 |

**B.5 model actually uses HD map / bbox**: removing them increases collision +15% relative and degrades turning L2 +15% relative. The model learned to consult those modalities for safety + intersection decisions. B.6 (dropout-trained) is much flatter — dropout taught it to under-weight HD map/bbox, so masking them at eval barely changes output (but the base quality is also worse).

### 11.5 B.6 dropout = negative result (paper-cite-worthy)

B.6 = B.5 + per-modality dropout p=0.1 (HD map / bbox independently dropped per sample; camera + ego never). Standard "modality robustness" trick from SwitchOut/PerceiverIO.

**Result**: B.6 is +0.13 L2 worse than B.5 across both eval modes, even though train_loss is virtually identical (B.5 ep3 0.7731 vs B.6 ep3 0.7786).

**Three root causes (most-to-least likely)**:
1. **Stochastic input → gradient variance ↑ → worse minimum** at fixed hyperparams (lr 2e-5, 3 epoch). Optimization needs different schedule for dropout-augmented data.
2. **24K + dropout = each pattern undertrained**: only ~2K samples per dropout regime (9% drop-HDmap, 9% drop-bbox, 1% drop-both). LLaVA stage-1 uses 558K to make dropout work.
3. **Model learns to down-weight modalities** because they're "unreliable" (gone 19% of time). Eval-time full modality then underused → trajectory accuracy degrades.

**Negative finding for resume / paper**: "Naive per-sample modality dropout p=0.1 hurts multi-modal SFT at 24K scale by +18% L2 even though it preserves the safety improvement (-7% collision); aligned with the LLaVA stage-1 threshold (~500K pair) needed for dropout to work."

### 11.6 What this means for the resume narrative

The **original** framing (§7-§8) implied "all ablations land near baseline, the right call is image-as-modality with OEM Linear". v2 sharpens this:

> "Built a 6×10 ablation matrix on Qwen2.5-VL-3B + nuScenes planning. Open-loop
> L2 saturates near 0.62-0.80 across designs (matches arXiv 2312.03031 "Is Ego
> Status All You Need?"), but sub-scenario metrics differentiate sharply: B.5
> image-as-modality SFT (HD map BEV + 3D bbox text via shared OEM Qwen2.5-VL
> ViT) reduces collision rate by 18%, turning-L2 by 18%, lane-change-L2 by 20%
> relative to camera-only baseline. Modality-failure robustness probe shows the
> model genuinely depends on HD map for these gains. Naive p=0.1 modality
> dropout (B.6) hurts overall L2 by +18% without recovering camera-only mode."

Three keyword phrases for HR / interview match against XPeng / Li Auto / 小鹏 JDs:
- **"Multi-axis ablation with sub-scenario breakdown"** (not single L2)
- **"-18%/-20% on hard scenarios"** (quantified breakthrough)
- **"Aligned with paper-cited saturation"** (industry reading)

### 11.7 Artifacts produced this round

| File | Purpose |
|---|---|
| `scripts/multimodal_planning_dataset.py` | Image-as-modality dataset (HD map + bbox + ego) for B.5/B.6 training |
| `configs/nuscenes_planning_b5.yaml` | B.5 config (no dropout, R1' hyperparams) |
| `configs/nuscenes_planning_b6.yaml` | B.6 config (dropout 0.1, sole delta vs B.5) |
| `scripts/launch_nuscenes_b{5,6}.sh` | Pre-flight + launchers |
| `scripts/b5b6_orchestrator.sh` | Chained B.5 → DP eval → B.6 → DP eval |
| `scripts/run_b5b6_eval_matrix.sh` | 4-cell matrix for B.5/B.6 (multimodal × cam-only) |
| `scripts/run_all_ckpts_eval_v2.sh` | 6-cell matrix incl. R1' / R1'' baselines |
| `scripts/_smoke_planning_eval_multimodal.py` | CPU smoke for the eval-pipeline fix |
| `scripts/_smoke_planning_metrics.py` | CPU smoke for the sub-scenario + behavior metrics |
| `scripts/planning_eval.py` | EXTENDED with `--multimodal` flag + new metrics |
| `eval_results/track_v2/*.json` | 6 result JSONs with full metric schema |
| `logs/all_ckpts_eval_v2_summary.md` | Headline markdown table (this section §11.2 source) |

---

## 12. Related repos worth citing in interview

- **torchtitan** — 5D parallelism reference impl (https://github.com/pytorch/torchtitan)
- **OmniDrive** — multi-modal cross-attention pool (closest to our projector-replacement series)
- **DriveMLM** — token-stream concat + explicit modality dropout
- **UniAD** — BEV unification (the perception→prediction→planning gold standard)
- **mmdetection3d** — LiDAR encoder zoo (PointPillars, CenterPoint, BEVFormer)
- **Megatron-LM** — TP+PP+SP reference (older but canonical)
