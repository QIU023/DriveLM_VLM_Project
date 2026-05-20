# 004 — Vision-CP open problem (research note, not a PR)

**Type**: research note + design rationale for PR 003
**Linked PR**: 003 (qwen3_vl CP text-only wiring)
**Doc**: `/workspace/DriveLM_VLM_Project/docs/2026-05-20_vision_cp_open_problem.md` (1000 words, primary writeup)

## Why this is here

This isn't a PR. It's the research/engineering rationale we'd point reviewers
at if they ask "why didn't you make CP work end-to-end including vision?"

The 1000-word writeup documents 2 specific gaps Agent C hit when trying to
shard a vision-attached forward under CP:

1. **`masked_scatter` shard mismatch** — vision_embeds is unsharded; LM tokens
   are sharded; the mask indexes into the wrong layout.
2. **MRoPE per-shard index advancement is wrong** — multimodal RoPE
   advancement is stateful; each CP shard can't independently compute its
   absolute positions without seeing all earlier image markers.

## Industry parallel

| Framework | Vision-CP posture |
|---|---|
| Megatron-LM | `assert not (cp_size > 1 and pp_size > 1)` for VLM; "future work" |
| NeMo | open issues report CP+VLM instability; recommended cp_size=1 for VLM |
| DeepSpeed | no documented multimodal SP/CP |
| HF Accelerate | no CP at all |
| torchtitan (ours, after PR 003) | text-only CP works; vision-CP raises explicitly |

## SOTA AD VLA workaround

No production AD VLA uses vision-CP. Instead: pre-LM token compression
(Q-Former / spatial-merge / cross-frame compression) keeps the vision token
budget small enough that CP is unnecessary. See `docs/2026-05-20_vision_cp_open_problem.md`
section 5 for the survey table (AutoVLA / DriveVLM / Apollo / LongVU / Wayve).

## What we do

- **Track A**: F+T+P+SP for AD VLA (NO CP, vision present). Aligns with
  SOTA pattern.
- **PR 003**: text-only CP wiring (small focused upstream contribution).
- **This doc**: cite as the reasoning behind the `NotImplementedError` in
  PR 003, so reviewers don't ask "why not just fix it".
