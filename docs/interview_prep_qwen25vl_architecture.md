# Interview Prep — Qwen2.5-VL Architecture, DriveLM Fit, Distributed & Deployment

Target roles: XPeng JD2 (VLA Foundation Model, most on-point), JD3 (E2E
Perception/Planning), JD1 (CV Engineer, C++ deployment — referral).

The resume gets you the interview; **this doc is what wins it.** Read for the
*causal chains*, not the facts — interviewers push on "why".

---

## Part 1 — Qwen2.5-VL architecture (4 anchors)

### Anchor 1: M-RoPE is the core idea
Position ids are decomposed into **(temporal, height, width)**:
- **Text**: all three equal → collapses to ordinary 1D RoPE.
- **Image**: height/width vary across patches, temporal constant.
- **Video**: temporal increments per frame.

One mechanism handles **both** variable 2D spatial layout (needed for VQA visual
grounding) **and** video temporal order (needed for planning motion history). The
`second_per_grid_ts` field in the code feeds the temporal dimension; it's why the
same model does single-image QA and multi-frame video planning without arch
changes.

> Likely follow-up: "How does the model know where an object is for grounding?"
> → M-RoPE preserves 2D structure so attention can localize; Qwen2.5-VL was
> pretrained with bbox/point grounding, so DriveLM's coordinate grounding tags
> transfer.

### Anchor 2: Where multimodal fusion actually happens
Path: **ViT** (native dynamic resolution, NaViT-style; window attention in most
layers + full attention in 4) → **2×2 patch merger (an MLP projector)** that
groups adjacent patches, concatenates (4× channels), and projects to the LLM
hidden dim (this is the spatial downsample + alignment step) → visual tokens are
**inlined as a contiguous block** between text tokens via
`<|vision_start|> … <|vision_end|>`.

The **projector is the fusion bottleneck** — cheapest place to intervene. That's
exactly why the parallel projector-ablation branches exist
(perceiver-resampler / Q-Former / pixel-shuffle / resampler): they swap this
module to change the visual-token count or the fusion inductive bias.

### Anchor 3: VQA → VLA distribution gap (interviewers love this)
- **VQA**: map visual regions → language. Transfers well from pretraining.
  Driving-specific pain: tiny distant agents (visual-token budget vs small-object
  recall trade-off) and single CAM_FRONT losing 360° context.
- **VLA**: output is no longer language but a **trajectory**. We tokenize
  continuous (Δx, Δy) into **256-bin/dim discrete action tokens** (OpenVLA/RT-2
  style) placed in spare vocab slots — **no embedding resize**, reuses the LM
  head. The hard part is the **action distribution is heavily peaked on
  "constant-speed straight"** → naive cross-entropy collapses to predicting the
  mean (straight line).

  **The causal chain to be able to recite:**
  1. peaked action distribution → mode collapse
  2. → add **ego-speed conditioning** to the prompt ("Ego Status All You Need")
     so the model isn't guessing speed from pixels
  3. → **tighten bins to [−2, 40] m forward / [−8, 8] m lateral** so quantization
     resolution concentrates where data actually lives, instead of wasting bins
     on a [−50, 50] range
  4. → **evaluate on L2 + collision rate (UniAD/VAD protocol), NOT token
     accuracy** — token-acc is misleading when the distribution is peaked

### Anchor 4: Scale-up — how to train each module
- **Vision tower**: freeze for small data (current setup). Unfreeze only with
  fleet-scale data + new domains (night/rain/fisheye), and at a **lower LR
  (~1e-6, an order below the LLM)** to avoid catastrophic forgetting.
- **Projector**: cheapest alignment lever — retrain whenever visual-token count
  changes (compression) or a modality is added.
- **LLM**: **LoRA for VQA** (close to text pretraining) vs **full-SFT for VLA
  planning** (action distribution is far from text, needs capacity to shift the
  distribution). This LoRA/full-SFT split is itself a defensible design decision.
- **Action head**: discrete bins lose precision on rare maneuvers → at scale,
  consider a **continuous regression head or a diffusion head** (JD3 mentions
  diffusion). RT-2/OpenVLA (discrete) vs π0/diffusion-policy (continuous) is a
  good comparison to volunteer.

---

## Part 2 — How Qwen2.5-VL fits DriveLM (modality & distribution challenges)

| Challenge | Why it's hard | How the architecture / training adapts |
|---|---|---|
| Small distant agents | limited visual-token budget | native dynamic res helps; but capping pixels trades off small-object recall — profiled in the compression work |
| Single CAM_FRONT | no depth, no 360° | known limitation; multi-cam (3-cam AutoVLA) + future BEV/occupancy (JD3 gap) |
| Action mode collapse | peaked toward straight | ego-speed conditioning + tight bins + L2/collision metric |
| Temporal reasoning | planning needs motion history | M-RoPE temporal dim + multi-frame video input |
| Token explosion (video) | N frames × spatial tokens blows context + compute | cross-frame compression (temporal pool / VTM / LongVU) |
| Domain gap | web-pretrained vision ≠ driving | freeze vision early; unfreeze w/ fleet data at low LR |

---

## Part 2.5 — Visual-token compression as a 2-D design space

The single biggest prefill lever (Part 4 shows driving VLA is prefill-bound). The
key reframe: visual tokens have **two orthogonal redundancy axes**, and the
project's two compression lines map onto them.

```
total visual tokens = (tokens per frame) × (frames) × (cameras)
                            ↑ SPATIAL          ↑ TEMPORAL    ↑ (cross-camera)
```

| axis | redundancy | methods (this repo) | granularity |
|---|---|---|---|
| **Spatial (intra-frame)** | sky/road texture within a frame | FasterVLM, PruMerge, PyramidDrop, SATS-CRP | per-frame |
| **Temporal (inter-frame)** | adjacent 2 Hz frames ~overlap | temporal-pool, VTM, LongVU | across frames |

**They are orthogonal and composable** — spatial-prune each frame, *then*
temporally merge (spatial-first is cleaner; temporal merge scrambles per-frame
attention signals the spatial methods rely on). 4× spatial × 2× temporal = 8×.

**Why temporal feels marginal on driving (4 frames):** at 4 frames @ 2 Hz the
temporal axis has only ~2× headroom; **spatial dominates the short-clip case**
(140 tok/frame × 4 frames → spatial is most of the budget). Temporal/cross-camera
compression is the foundation for the **long-video (16f+) and multi-cam (3–6)**
direction, where token count explodes (6 cam × 8 f × 140 = 6720). Say this plainly
or an interviewer asks "why cross-frame on a 4-frame clip?"

**Do the single-image (VQA) methods transfer to video planning?** Yes, applied
**per-frame** (spatial axis); most are **training-free** (FasterVLM = vision-encoder
attention; PruMerge = similarity merge; PyramidDrop = layer-wise drop), so you
**re-evaluate the already-trained planning model with the compressor inserted —
no retraining** (`scripts/planning_eval_compress.py`). They miss temporal
redundancy → that's why video-aware methods exist. SATS-CRP (if train-aware) is
the one exception that costs compute.

**FLOPs / sequence composition (3B, d≈2048, prefill ≈ 2·P·N):**

| config | visual tok | text | output | total seq | visual % | prefill FLOPs |
|---|---|---|---|---|---|---|
| baseline (1 cam, 4 f) | 560 | ~100 | ~14 | ~660 | 85% | ~4 TFLOP |
| spatial 4× | 140 | ~100 | ~14 | ~240 | 58% | ~1.5 TFLOP |
| spatial 4× + temporal 2× | 70 | ~100 | ~14 | ~170 | 41% | ~1 TFLOP |

Input is **overwhelmingly visual (prefill); output is tiny (a ~14-token
trajectory)** → compute is prefill- and visual-token-bound → compressing visual
tokens is the dominant FLOP lever (quantitatively, "compression ≫ MoE"). Bonus:
fewer tokens → smaller KV cache → the short decode is faster too. The attention
N² term means the win is **super-linear** at multi-cam / long-video scale.

**Pipeline location matters (where you compress decides what you save):**
- **pre-encoder** (frame/resolution selection, e.g. LongVU) → saves ViT encoder + LLM
- **post-encoder, pre-LLM** (2×2 merger / FasterVLM / VTM) → saves LLM prefill + KV, *not* the encoder
- **in-LLM** (PyramidDrop layer drop) → saves later-layer compute

**Real-time prompt + a deployment trick:** the VLA prompt is system + [video] +
fixed instruction + ego-status (+ navigation command in prod). The **text is a
near-static template**; only the frames + ego-state change per step. Current layout
(system → video → instruction) blocks prefix-KV-caching because the changing video
sits before the instruction — **move all static text ahead of the video** to cache
the static prefix.

**Action-attention pruning (a novel, training-free contribution for VLA):** vanilla
FasterVLM prunes by `[CLS] → visual` attention (task-agnostic). For VLA, prune by
**`action-token → visual` attention** — keep the visual tokens the trajectory
prediction actually attends to (task / planning-relevant). Training-free, and it
extends SATS-CRP's region-awareness from VQA "region relations" to planning
"action-relevant regions."

**Where do planning-region labels come from? (no new annotation needed.)** All
derivable from data nuScenes already has, or from the model itself:
1. **future-trajectory projection** — project the (already-labeled) future waypoints
   into CAM_FRONT via calibration → the drivable corridor the ego will traverse
2. **agent boxes** — the `gt_boxes` already used for collision eval, projected → agent regions
3. **HD-map drivable area** — projected → road region
4. **action→vision attention** — training-free saliency from the trained model (= the
   action-attention pruning signal above)

(1)+(3) give a geometric prior; (4) gives model saliency; agreement between the
projected corridor and the attention heatmap is itself an interpretability figure.

---

## Part 3 — Distributed training (the reasoning, not the code)

**For the 3B DriveLM VLA: FSDP FULL_SHARD is correct and sufficient.** Don't add
TP/PP/CP — they're for 30B+ or very long context; on 3B they only add
communication overhead.

- **EP (expert parallel)**: not needed — the VLA is **dense, no MoE**, and this is
  the **correct standard for real-time driving VLA** (OpenVLA, AutoVLA, π0, EMMA,
  DriveVLM are all dense backbones). Do **NOT** claim "MoE beats the bandwidth wall
  at prod scale" — that's wrong for driving (see Part 4 for the physics: driving's
  action path is short → prefill-bound, not decode-bound, so a decode-throughput
  lever like FFN-MoE targets the wrong bottleneck; and on edge, all expert weights
  stay resident → no memory-capacity saving + routing adds latency variance, bad
  for hard real-time). MoE in driving is a 2025 *research* line and uses
  **structured experts** (DriveMoE's scene/camera + skill MoE; AutoMoT's fast-slow
  Mixture-of-Transformers), not generic token-routing FFN-MoE. Generic MoE-VLM
  (DeepSeek-VL2, Aria) is for broad-task long-video understanding, not narrow
  real-time control.
- **TP**: shards each layer's matmuls; all-reduce every layer → needs NVLink.
  Overkill for 3B; bad on no-NVLink consumer GPUs.
- **PP**: splits layers into stages; communicates only at boundaries → tolerates
  PCIe; throughput-oriented with pipeline bubbles.
- **CP (context parallel)**: the only one with a **real open problem for VLMs** —
  visual tokens are a contiguous block, and the 2×2 merger + window attention
  can't be sharded along the sequence the way text can. Splitting the visual
  block across CP ranks breaks fusion. Relevant **only for long-video (16f+)**
  when context blows up. This is a genuinely novel-ish research framing:
  "CP for long-video VLA with a non-shardable vision tower."

**The two-project bridge** (answers JD2's fleet-data + cross-modal-alignment ask):
- DriveLM VLA = VLA modeling + multimodal fusion + compression (well-trained,
  has results).
- AttnResidualTorchTitan = real 5D parallelism + self-implemented PP infra
  (the "scale across GPU clusters" credential).
- **The闭环**: adding LiDAR / CAN bus / maps / fleet video → more encoders +
  params + longer multimodal context → *then* TP/PP/CP become necessary, plugging
  into the PP infra already built. The scale-up is a *reasoned design*, not a
  re-implementation (limited compute → don't re-run 5D on DriveLM).

---

## Part 4 — Production deployment (Orin/Thor, TRT-LLM, the bandwidth wall)

### The one-liner
"30B dense on Orin is a **batch=1 memory-bandwidth wall** — single-chip INT4 gives
~8–10 tok/s. But the binding constraint for driving VLA is **prefill** (the action
path emits a short trajectory, so it's prefill-bound, not decode-bound). Landing it
needs **small dense model + visual-token compression (cuts prefill) + W4A4/FP4 +
KV-cache quant + distillation** — NOT FFN-MoE (wrong bottleneck + no edge memory
saving + routing latency variance). Dual-Orin runs a **functional pipeline (not TP
— no NVLink, no shared memory)**; true 30B-class VLA waits for **Thor's FP4**."

### Why automotive inference is uniquely hard: batch = 1
One ego vehicle, fixed sensor rate, can't batch across time (each frame must be
planned before the next matters). Datacenter serving hides weight-load latency by
batching → compute-bound. **Car-side can't → memory-bandwidth-bound, the worst
case for GPU utilization.** This is *the* reason on-vehicle driving models are
**small dense + heavily quantized** (not MoE — see the Levers section for why).

### The math (dual Orin-X: 254 TOPS, 64 GB, ~204 GB/s each)
Decode tok/s ≈ effective_BW / weight_bytes_per_token:

| precision | 30B weights | theoretical tok/s | realistic (~65% BW + KV) |
|---|---|---|---|
| FP16 | 60 GB | — | won't fit (KV + vision + OS) |
| INT8 | 30 GB | 6.8 | ~4–5 |
| INT4 | 15 GB | 13.7 | ~8–10 |

- 50-token plan @ ~10 tok/s ≈ **5 s** vs ~100 ms budget → **~50× too slow**
- prefill of ~1k tokens alone ≈ **0.5 s** on Orin dense compute → already blows budget

### Multi-chip arrangement
Two Orins **don't share memory** and have **no NVLink** → **can't TP**. Arrange as
a **functional pipeline**: Orin-A = vision encoder + perception, Orin-B =
LLM/planning, passing only **compressed visual tokens** across the link. Pipeline
to hide latency (encode frame t+1 while decoding plan for t). This is cross-chip
**PP / functional partition**, not TP.

### Levers (and how this repo maps)
Driving's action path is **prefill-bound** (short trajectory output), so optimize
prefill first:
- **Small dense model** → standard for real-time driving VLA (OpenVLA/AutoVLA/π0/
  EMMA are all dense); FFN-MoE is the *wrong* tool here (decode-throughput lever on
  a prefill-bound, edge-memory-constrained, hard-real-time workload)
- **Visual-token compression** (this repo) → prefill ∝ tokens, VLA is vision-heavy
  → biggest prefill win + minimizes inter-chip transfer
- **W4A4 / FP4 + KV quant** → fewer LPDDR bytes
- **Distillation** (this repo's SATS-CRP KD) → large teacher → small dense student

> MoE in driving exists only as 2025 *research*, and as **structured experts**, not
> generic FFN routing: DriveMoE (arXiv 2505.16278) = Scene/camera Vision MoE +
> Skill Action MoE (for rare maneuvers — addresses the peaked-action problem);
> AutoMoT (2603.14851) = fast-slow Mixture-of-Transformers (async reasoning vs
> action). Production stays dense.

### Orin → Thor + the 5090 FP4 preview
Thor (Blackwell, **FP4-native**, ~1000 TOPS INT8 / ~2000 TFLOPS FP4, much higher
BW) is the chip for 30B-class VLA. **NVFP4 is Blackwell-native**, and the **RTX
5090 (sm_120) is also Blackwell** → deploying the VLA in **FP4 on the 5090 is a
faithful local preview of the Thor production path** (same datatype). Ada
(4070Ti) has no FP4 tensor cores — that's why deployment moved to the 5090.

---

## Part 5 — TRT engine + C++ runtime (JD1 / the internship gap)

(Prior internship: stopped at ONNX checkpoint; never did TRT + C++. This is that
gap, filled.)

### Engine = AOT-compiled, arch-locked artifact
Build-time passes:
1. **Kernel/layer fusion** (conv+bias+act → 1 kernel; fused attention/GEMM)
2. **Tactic auto-tuning** — benchmarks kernel variants **on the actual GPU**,
   picks fastest → **why the engine is arch-specific** (sm_120 tactics ≠ sm_89)
3. **Per-layer precision** (FP4/INT8/...), inserts quant/dequant
4. **Static memory planning** — buffer reuse via lifetime analysis → **zero
   malloc at runtime**

### TRT C++ runtime
`IRuntime → deserializeCudaEngine(.engine) → ICudaEngine →
createExecutionContext()`. Bind device buffers with `setTensorAddress`, pick a
dynamic-shape optimization profile with `setInputShape`, launch async with
`enqueueV3(stream)`, wrap with H2D/D2H `cudaMemcpyAsync` + `cudaStreamSynchronize`.
Links `libnvinfer` + `libnvinfer_plugin` + CUDA runtime. The Python API is a thin
pybind wrapper over this.

### TRT-LLM adds an LLM runtime on top
`libtensorrt_llm` adds **paged KV cache, in-flight/continuous batching, sampling,
the autoregressive decode loop**. Modern C++ entry point:
`tensorrt_llm::executor::Executor` (predecessors `GptManager`/`GptSession`);
Triton's trtllm backend wraps it. Core plugins: `gpt_attention_plugin` (fused
attention + KV read/write), `gemm_plugin`. Multimodal: vision engine runs first,
its embeddings are spliced into the LLM input at the visual-token placeholders.

### Multi-GPU (datacenter, distinct from car-side)
- **Single model across GPUs** (set at `trtllm-build` time): `--tp_size` (per-layer
  all-reduce via MPI+NCCL, needs NVLink) or `--pp_size` (stage boundaries, PCIe-ok,
  bubbles). For 3B: **single GPU — don't shard.**
- **Throughput** (many requests): N independent engine instances, one per GPU,
  zero inter-GPU comm, near-linear QPS scaling (Triton model instances).
- **Consumer Blackwell (5090) has no NVLink** → never TP across them; use
  data-parallel instances, or PP if a model doesn't fit.
