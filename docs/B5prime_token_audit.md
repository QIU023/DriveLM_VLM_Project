# B.5' Token Audit (3-cam Qwen2.5-VL × multimodal)

**Threshold**: max_length = **12288**

**Method**: tokenize chat-template text + add fixed visual expansion (3 cams × 3648 + 1 HD-map × 64 = 11008 visual tokens). Text varies per sample (bbox count + ego speed). No model load, no pixel I/O.

## train

- n = 28130
- min/p50/p90/p99/max = 11103 / 11570 / 11634 / 11674 / 11690
- **samples exceeding 12288: 0 (0.00%)**

## val

- n = 6019
- min/p50/p90/p99/max = 11103 / 11571 / 11632 / 11672 / 11681
- **samples exceeding 12288: 0 (0.00%)**


## Decision rule
- 0% over → B.5' results valid
- <1% over → essentially valid, add caveat
- 1-10% over → noisy, trend valid, absolute may be optimistic
- >10% over → MUST RETRAIN at higher max_length
