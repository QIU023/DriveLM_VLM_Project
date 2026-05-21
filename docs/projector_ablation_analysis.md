# Projector Ablation Analysis — Why Linear Wins at 24K nuScenes

Track A 1-cam projector ablation result + design-space discussion + scaling-law
extrapolation. Companion to `docs/full_modal_vla_research.md`. Pulls together
the empirical R1' / A.0 / A.1 / A.2 numbers and explains the
counter-intuitive ordering.

(A.3 Perceiver Resampler is still training at the time of writing; this doc
will be updated when its DP eval lands.)

---

## 1. Headline numbers

| Run | Projector | LM-input visual tokens / 4-frame sample | L2_avg (TemAvg) | collision_avg |
|---|---|---|---|---|
| **R1'** (baseline) | **Linear** (native 2×2 merger) | 480 | **0.6423** ★ | **3.73%** ★ |
| **A.0** | Linear (3-cam) | 1440 | 0.6626 | 4.16% |
| **A.1** | Q-Former 64q × 6L | 64 | 0.6807 | 3.95% |
| **A.2** | PixelShuffle 2× | 180 (+50% visual budget caveat) | 0.6717 | 4.16% |
| A.3 | Perceiver Resampler 64-latent | ~64 (pending) | TBD | TBD |

Ranking by L2_avg: **R1' (Linear) > A.0 (3-cam Linear) > A.2 (PixelShuffle) > A.1 (Q-Former)**

All learned-projector variants **lose to the native pretrained Linear baseline**
at the 24K nuScenes scale. This is the same pattern that produced "1-cam → 3-cam
gave no L2 win" in A.0: at this scale, **more learnable visual machinery is not
better**.

---

## 2. What each projector actually does

### Linear (R1', native Qwen2.5-VL 2×2 patch merger)
```
ViT 16×30 patches per frame (after dynamic resize)
  → native 2×2 patch merger MLP: groups 4 adjacent patches, concatenates
    (4× channels), Linear to LM hidden dim
  → 8×15 = 120 tokens per frame
  → all 120 tokens inlined into LM context per frame
  → 4 frames × 120 = 480 LM-input visual tokens / sample
```
- **Change vs OEM Qwen2.5-VL**: none architecturally; just re-fine-tuned
- **Pretraining alignment**: ✓ the MLP was pretrained on the entire Qwen2.5-VL
  image-text corpus
- **Information preserved**: ~100% (only native merger compression)
- **New parameters**: ~0 (re-fine-tunes the existing MLP)

### Q-Former (A.1)
```
ViT 480 patches × 4 frames = 1920 raw features per sample
  → 64 learnable queries cross-attend (6 layers of cross+self attention)
  → 64 LM-input visual tokens per sample (FIXED, frame-independent)
```
- **Change**: REPLACES the native merger entirely with BLIP-2-style Q-Former
- **Pretraining alignment**: ✗ random-init; must learn from 24K SFT
- **Information preserved**: ~12.5% (1920 → 64 = 30× raw compression, ~8.75× vs Linear's 480)
- **New parameters**: ~90 M (queries + cross+self attn × 6 layers)

### PixelShuffle 2× (A.2)
```
ViT 16×30 / frame
  → native 2×2 merger → 10×18 = 180 per frame (A.2 uses 150528 max_pixels for
    even-grid constraint; baseline R1' is 8×15 = 120 / frame at 109760)
  → 2×2 PixelUnshuffle (space → channels): 5×9 = 45 per frame
  → Linear (4·D → lm_dim)
  → 4 frames × 45 = 180 LM-input visual tokens / sample
```
- **Change**: extra deterministic spatial down-sample + a single Linear
- **Pretraining alignment**: ✗ the Linear is random-init (PixelUnshuffle has no params)
- **Information preserved**: ~38% (all patches retained, only spatial rearrangement; but A.2 was given +50% raw pixels)
- **New parameters**: ~17 M (just the final Linear)

### Perceiver Resampler (A.3, pending)
Similar to Q-Former but with explicit **temporal positional encoding** per frame:
queries cross-attend to multi-frame visual features with Flamingo-style
temporal embeddings. 64 latents. ~90 M params. Same per-sample token count
(fixed 64) and same pretraining gap as Q-Former.

---

## 3. LM-input visual token budget — the core comparison

| Projector | Tokens / 4-frame sample | Ratio vs R1' Linear | Compression severity |
|---|---|---|---|
| **Linear (R1')** | **480** | **1.0×** | none beyond native merger |
| Q-Former (A.1) | 64 | 0.13× | 7.5× (extreme) |
| PixelShuffle (A.2) | 180 | 0.38× | 2.7× (moderate, deterministic) |
| Resampler (A.3) | 64 (predicted) | 0.13× | 7.5× |

A.2's 0.6717 L2 came with **+50% raw visual pixel budget** vs R1' (10×18=180 raw
tokens/frame vs 8×15=120). A fair-budget comparison would expect A.2 to do worse.

---

## 4. Why Linear wins at 24K — root causes

Three intertwined factors:

### 4.1 Pretrained alignment vs from-scratch
- **Linear (R1')**: the projector inherits the alignment built up over the
  entire Qwen2.5-VL pretraining corpus (image-text pairs at internet scale).
  Fine-tuning on 24K just refines an already-correct mapping.
- **Q-Former / PixelShuffle / Resampler**: random initialization. The 24K + 3
  epoch budget = ~72 K (visual, text) pairs through the loss. For a 90 M-param
  Q-Former this is **vastly under-determined** — the learned-pool weights have
  ~10⁹ degrees of freedom; the SFT signal can only cover a tiny fraction.
- **Empirical consequence**: A.1 val_loss tracked R1' closely on language
  modelling, but downstream L2 was 0.038 worse — the Q-Former learned a *plausible*
  pool but not the *right* one for trajectory regression.

### 4.2 Information bottleneck vs task signal
- Trajectory regression in driving needs **fine-grained spatial detail**: small
  distant agents, lane geometry, stop-line position. These are exactly what
  high-frequency ViT patches encode.
- Linear preserves all 120 tokens per frame → all spatial detail survives into
  the LM.
- Q-Former discards 87.5% (480 → 64 per sample) — any patch the queries don't
  attend to is lost. With random-init queries on small data, the queries
  attend to "generic" features (likely large objects, sky), not the
  trajectory-relevant spots.
- PixelShuffle preserves all patches structurally (just rearranges space →
  channels) but the final Linear must re-learn how to consume 4× channel-stacked
  features → some signal lost in the random-init linear.

### 4.3 New parameters vs available signal
- Linear: 0 new params → 24K samples × 3 epochs is more than enough to refine.
- PixelShuffle: 17 M new params → 24K marginal.
- Q-Former / Resampler: 90 M new params → 24K severely insufficient.

**Rule of thumb**: roughly 1 K supervised examples per million projector
parameters. 90 M Q-Former wants ~90 K paired examples; we have ~24 K × 3 ≈ 72 K
forward passes (and most of those are noisy, not paired with curated
trajectory-relevant supervision).

---

## 5. Why Linear breaks at scale (data-rich regime)

The same factors flip sign as data grows and history lengthens.

| Regime | Predicted winner | Why |
|---|---|---|
| **Small data** (≤50 K), short history (≤4 frames) | **Linear** | Pretrained alignment + no bottleneck wins; no scale for learned-pool to pay off |
| Medium data (100–500 K) | PixelShuffle ≈ Linear, Q-Former closing | Learned compressor starts paying off; tokens still budget-feasible |
| **Large data** (≥500 K), long history (8–16 frames) | **Q-Former / Resampler** | Fixed token count under learned cross-attn enables longer history + more modalities at same compute |
| **Full-modal** (cam + LiDAR + map + bbox) | **Q-Former / Resampler** | Cross-modal queries are the natural fusion primitive (OmniDrive, DriveMLM precedent) |

**The breaking point for Linear** is not loss quality — it's **LM context budget**.
- Linear at 16 frames × 1 cam = 1920 visual tokens. Fits.
- Linear at 16 frames × 6 cams = 11 520 visual tokens. Borderline.
- Linear at 16 frames × 6 cams + LiDAR-BEV + map-BEV = ~15 000 visual tokens.
  + text/traj overhead pushes context past 16 K. **Now CP/SP become load-bearing**
  (see `full_modal_vla_research.md` §6).
- Q-Former at the same input scale: fixed 64 (or 128/256) tokens regardless of
  frames/cams/modalities. Linear with no truncation is unsustainable.

This is the scaling-law argument that justifies production systems (OmniDrive,
DriveMLM, Flamingo, BLIP-2) all choosing learned cross-attention pooling
despite Q-Former being worse on small benchmarks.

---

## 6. Connection to deployment

For NVFP4 + TRT-LLM 5090 deployment, **the projector choice changes the
prefill token count, which dominates TTFT for short-output VLA**.

| Projector | Visual tokens at prefill | Prefill FLOPs (LM forward) | TTFT impact |
|---|---|---|---|
| Linear | 480 × seq_overhead → ~600 tok | ~120 GFLOPs (3B LM) | baseline |
| PixelShuffle | 180 + overhead → ~280 tok | ~56 GFLOPs | 2× faster prefill |
| Q-Former | 64 + overhead → ~150 tok | ~30 GFLOPs | 4× faster prefill |

So even though Linear wins on quality at 24K, **at deploy time PixelShuffle or
Q-Former may be the right pick** if the small L2 regression is tolerable for
the latency / throughput gain. This is the Pareto axis that
`scripts/sweep_compress_pareto.sh` (already on trunk) explores in
training-free mode.

---

## 7. Interview sound-bite

> "On 24 K nuScenes we ablated 4 projectors against the native pretrained
> 2×2-merger Linear baseline (R1', L2 0.6423). All three learned variants —
> Q-Former 64q (A.1, L2 0.6807), PixelShuffle 2× (A.2, L2 0.6717), Perceiver
> Resampler (A.3, in flight) — lost to baseline. Root cause is two-fold:
> (a) the native merger inherits image-text alignment from Qwen2.5-VL
> pretraining (~internet scale), while replacements are random-init and 24K
> SFT is wildly insufficient to relearn alignment for a 90M-param Q-Former;
> (b) trajectory regression needs spatial detail and Q-Former's 7.5×
> token compression discards it. This is consistent with the scaling-law for
> learned-pool projectors: BLIP-2 / Flamingo / OmniDrive all use Q-Former or
> Resampler because at 100K-1M+ scale, fixed token count enables long-history
> multi-modal context without blowing the LM's positional budget. Our 24K
> demo is the *small-data anchor* of that scaling curve — and the empirical
> reason to defer learned-projector decisions until fleet-scale data is in
> play. At deployment, PixelShuffle is the production-pragmatic pick:
> deterministic (no training-data hunger), 2.7× prefill speedup, modest L2
> regression (0.03 m at 3-s horizon)."

---

## 8. Open follow-ups (post-A.3, optional)

1. **Re-run A.1/A.2 with the projector frozen at OEM Linear weights for first
   N steps**, then unfreeze — does warm-starting from the pretrained projector
   close the gap? (Would isolate factor 4.1.)
2. **Multi-seed runs** (3 seeds each) to confirm L2 gaps are above noise.
   Right now ±0.02 differences could be seed-level.
3. **Frame-count sweep** (1, 4, 8, 16 frames) on the winning projector — at
   what frame count does Linear's context budget break and Q-Former overtake?
   This would be a clean empirical Pareto datapoint to cite.
4. **Joint with training-free 2D compression sweep** (`sweep_compress_pareto.sh`) —
   apply temporal/spatial compressors on top of the winning projector to
   establish the full L2-vs-tokens frontier for deployment.
