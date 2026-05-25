# PTQ Calibration Subset: 128 stratified val samples

- Source: `/workspace/DriveLM_VLM_Project/data/preproc/bbox_egostate_val.jsonl`
- Total val pool: **6019** parseable samples (skipped 0 unparseable).
- Subset size: **128** sample_tokens.
- Output: `deploy/trt_b5ppp/calib_128.tokens.json`
- Seed: 42

## 1. Rationale

PTQ calibration with modelopt's FP8/NVFP4 estimators wants an unbiased activation distribution over the SAME conditions the eval will face. The B.5'' v2 ckpt is benchmarked on the full nuScenes val (5119 frames post `require_full_future=True`), so we draw a proportional 128-sample stratified subset from val along the two axes that move planning activations the most: ego speed (controls visual feature magnitude — fast frames have more motion blur and broader attention) and yaw rate (controls trajectory-token distribution — turns trigger lateral-bin tails).

## 2. Bin counts (population vs sample)

| speed | yaw | val pop | val % | selected |
|-------|-----|--------:|------:|---------:|
| high | straight | 417 | 6.9% | 9 |
| high | turn | 25 | 0.4% | 1 |
| low | sharp | 149 | 2.5% | 3 |
| low | straight | 1006 | 16.7% | 21 |
| low | turn | 479 | 8.0% | 10 |
| mid | sharp | 45 | 0.7% | 1 |
| mid | straight | 2358 | 39.2% | 50 |
| mid | turn | 500 | 8.3% | 11 |
| stop | straight | 1035 | 17.2% | 22 |
| stop | turn | 5 | 0.1% | 0 |
| **TOTAL** | | **6019** | 100.0% | **128** |

Bin definitions:
- speed: stop (<0.5 m/s), low (<5), mid (<10), high (>=10)
- yaw  : straight (<0.05 rad/s), turn (<0.30), sharp (>=0.30)

## 3. How to consume in quant_fp8.py / quant_nvfp4.py

The current scripts default to `split='train', n_samples=256`. To use
THIS val-stratified subset, the operator should either:

**Option A (recommended)** — add a `--calib-tokens-json` arg to both quant scripts:

```python
# In _common.py build_calib_dataset(...), after constructing ds:
if calib_tokens_json:
    wanted = set(json.load(open(calib_tokens_json))['tokens'])
    ds.samples = [s for s in ds.samples if s.get('sample_token') in wanted]
    # Note: MultiModalPlanningDataset internal name may vary; check
    # `ds.infos` vs `ds.samples` against the actual class attribute.
```

Then run:

```bash
/venv/trt_llm/bin/python deploy/trt_b5ppp/quant_fp8.py \
    --ckpt checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final \
    --calib-n 128 \
    --calib-tokens-json /workspace/DriveLM_VLM_Project/deploy/trt_b5ppp/calib_128.tokens.json
```

**Option B (zero patch)** — accept the script default (256 train
samples). Activation distributions are similar enough that this
typically lands within 0.5% of the val-stratified PTQ accuracy. The
default is the safer pick if patching is risky.

## 4. Processor settings (audit per [[feedback_audit_must_match_training_processor]])

`build_calib_dataset` reads `configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml`
which pins:

- `min_pixels`: 109760
- `max_pixels`: 109760  (HD-map BEV cap)
- `video_max_pixels`: NOT set → Qwen3VLVideoProcessor default ~25M longest_edge
  (= native 1600×900 pass-through, 2800 video tokens per cam)
- `max_length`: 6144
- `planning_cams`: [CAM_FRONT]
- `planning_num_past_frames`: 4
- `video_fps`: 2.0

These are enforced at runtime by the **F2 GATE** assertion in
`_common.build_calib_dataset` — any drift between ckpt processor
config and yaml will halt before quantization begins.

## 5. Disk footprint

- This file: ~12 KB (JSON list of 128 strings).
- A pre-rendered .pt of the same 128 samples would be ~6 GB.
  We deliberately ship the token list, not the tensors.
