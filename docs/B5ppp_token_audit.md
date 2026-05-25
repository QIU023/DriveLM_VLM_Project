# B.5''' Token Audit (3-cam Qwen3-VL-4B × multimodal)

**Threshold**: max_length = **12288**

**Visual fixed**: 3 cams × 2112 + HD-map 121 = **6457** tokens

**Method**: real Qwen3-VL-4B processor for visual expansion + tokenize chat-template text per sample. No model load, no real image I/O (dummy frames for grid probe only).

## train

- n = 100
- min/p50/p90/p99/max = 6761 / 7014 / 7058 / 7086 / 7086
- visual fixed = 6457
- **samples exceeding 12288: 0 (0.00%)**

## val

- n = 100
- min/p50/p90/p99/max = 6545 / 6733 / 7097 / 7113 / 7113
- visual fixed = 6457
- **samples exceeding 12288: 0 (0.00%)**


## Recommended max_length
Set max_length = max(over both splits) + ≥10% safety buffer.
