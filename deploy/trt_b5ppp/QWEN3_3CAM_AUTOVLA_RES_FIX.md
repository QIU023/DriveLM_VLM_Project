# B.5''' Qwen3-VL-4B 3-cam — AutoVLA video-resolution token-budget fix

Date: 2026-05-26
Config: `configs/nuscenes_planning_3cam_qwen3vl_multimodal.yaml`
Backup before edit: `configs/nuscenes_planning_3cam_qwen3vl_multimodal.yaml.bak_pre_autovla_res`

## The bug

AutoVLA's OFFICIAL Qwen2.5-VL-3B SFT setting is `min_pixels == max_pixels == 109760`
(a deterministic resolution-for-coverage budget). On Qwen2.5-VL — patch-14, merge-2
→ 28 px / merged-token — that yields ~240 post-merge visual tokens per camera. Our
B.5' Qwen2.5-VL-3B 3-cam model measured **240 tok/cam** at this setting and is
AutoVLA-faithful.

The Qwen3-VL-4B 3-cam config inherited the SAME `109760`. But Qwen3-VL is **patch-16,
merge-2 → 32 px / merged-token**, so the same pixel count only resolves to ~36
post-merge tok/cam. The **pixel number does not transfer across patch sizes — the
TOKEN count must be preserved**. The config was feeding the video tower a thumbnail.

Empirically confirmed (real 1600x900 × 4-frame nuScenes clips):

| video_*_pixels | resized/frame | video_grid_thw | post-merge tok/cam |
|----------------|---------------|----------------|--------------------|
| 109760 (old)   | 96x192        | [2, 6, 12]     | **36**  ← bug      |
| 524288 (new)   | 256x480       | [2, 16, 30]    | **240** ← parity   |

## Why two separate knobs

Qwen3-VL has two processors with DIFFERENT pixel semantics:

- `image_processor` (HD-map BEV): `smart_resize` caps **per-frame** pixels. Keep
  `min_pixels = max_pixels = 109760` exactly as B.5' (HD-map unchanged → 121 tok).
- `video_processor` (`Qwen3VLVideoProcessor`): `smart_resize` caps the **TOTAL CLIP
  volume** `t_bar * h_bar * w_bar` (NOT per-frame). So the video cap is ~T larger
  than an image cap. `train_lora.py` (lines ~2714-2733) writes this from the
  separate `video_min_pixels` / `video_max_pixels` YAML keys onto the video
  processor's `size.shortest_edge` / `size.longest_edge` (= min / max in
  smart_resize). When those keys are absent the video processor is left untouched.

On Qwen2.5-VL there was no dedicated video_processor, so the old config's video path
silently passed native frames; on Qwen3-VL the same branch fires, which is why the
inherited image cap leaked into (and shrank) the video.

## Chosen value: `video_min_pixels = video_max_pixels = 524288`

- `min == max` → deterministic resolution (AutoVLA style).
- 524288 = 2^19, sits mid-plateau (520000–576000 all give 240 tok/cam), robust to
  rounding.
- DERIVED EMPIRICALLY by sweeping the real `smart_resize` and the full processor,
  not from the naive 240×32² formula (the video processor's longest/shortest-edge
  semantics are total-clip, not per-edge, so the formula does not transfer directly).

## E2E verification (val split, 10 samples, exact train-recipe processor setup)

Processor: `AutoProcessor` from base Qwen3-VL-4B snapshot
`/workspace/.hf_home/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17`
with `image_processor.min/max_pixels=109760` and video caps applied via the verbatim
train_lora.py branch. Built `MultiModalPlanningDataset(val)` (len=5119).

```
post-merge video tok/cam : min=240 med=240 max=240   (all 3 cams, every sample)
HD-map image tok          : 121  (unchanged from B.5')
prompt_len (mask boundary): min=1467  med=1474  max=1512
total input_len           : min=1483  med=1490  max=1528   (cap=12288)
truncated samples (>= cap): 0/10  (0.0%)
max_length headroom       : 10760 tokens (87.6%)
```

Before fix: 36 tok/cam. After fix: 240 tok/cam. Clean cross-backbone parity with the
AutoVLA-faithful B.5' Qwen2.5-VL-3B budget.

## Memory / offload expectation

3 cam × 240 = 720 video tokens + 121 HD-map + ~600 text ≈ **~1500-token prompt**
(measured max 1528). That is ~6x SMALLER than the previous native-video plan
(~9100 tokens) the launcher's OOM notes assumed. At this prompt size the per-token
activation budget is tiny → **no CPU offload should be needed**; the existing
`FSDP_CPU_OFFLOAD=0` default in the launcher is correct and LBS=1 GA=4 is very
comfortable (likely room to raise LBS later if desired, but keep GBS=32 / LBS=1 GA=4
for paper parity unless re-audited).

## Ready-to-run launch command (NOT launched)

```bash
cd /workspace/DriveLM_VLM_Project
MODE=full bash scripts/launch_b5ppp_3cam_qwen3vl.sh
```

The launcher already:
- points at `configs/nuscenes_planning_3cam_qwen3vl_multimodal.yaml` (now fixed),
- uses model_id = local Qwen3-VL-4B snapshot (set in the config),
- runs `accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml` → world=8,
- config GBS = LBS=1 × GA=4 × world=8 = **32** (AutoVLA),
- `FSDP_CPU_OFFLOAD=0`, `save_every=50`, `val_every=50`, `keep_latest_k=2`, 3 epochs.

Recommend a `MODE=smoke` 5-step run first (`MODE=smoke bash scripts/launch_b5ppp_3cam_qwen3vl.sh`)
to confirm the new video shape + M-RoPE before the overnight full run.

## Notes / caveats found

- The config's backbone comment block still says "Switched to 2B" but `model_id`
  is the **4B** snapshot and GBS math uses world=8 → this is the 4B run; the 2B
  prose is stale. Did not change model_id (per task: 4B is correct). Flagging only.
- The OLD `max_length` token-audit comment in the config (9139 max, native 2800
  tok/cam video) is now obsolete — real max prompt is ~1528, headroom 87.6%.
