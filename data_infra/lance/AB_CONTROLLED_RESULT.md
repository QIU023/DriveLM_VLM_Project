# Cached-vs-Live 受控 A/B 最终结果(同配置,500 步)

**日期** 2026-05-27 · **脚本** `run_ab_controlled.sh`

## 配置(两臂完全一致 — 这是 apples-to-apples 的关键)
| 项 | 值 |
|---|---|
| config | `nuscenes_planning_3cam_qwen3vl_NATIVE.yaml`(native 2800 tok/cam,无 video cap) |
| grad checkpointing | **ON 两臂都开**(此前 80-step 跑只一臂开,被用户正确指出"测的内容都是错的") |
| activation checkpointing | ON |
| LBS | 2 · 8-GPU FSDP |
| steps | 500 · `--no-final-save` · `--save-every 999999`(全程零 ckpt 写入,验证空目录 0 字节) |
| 唯一变量 | LIVE=`--compress-method fastervlm --compress-ratio 4`(每步跑 ViT) vs CACHED=`--cached-vision-lance`(跳 ViT,注入缓存 token) |

## 结果
| 臂 | 中位 s/it | wall(500 步) |
|---|---|---|
| LIVE(每步 ViT) | **3.54** | 1937s |
| CACHED(跳 ViT) | **3.39** | — |

**同配置 speedup = 3.54 / 3.39 ≈ 1.04×**

## 诚实结论
- 在**真实训练配置(grad-ckpt ON)**下,跳 ViT 的每步加速只有 ~1.04×。原因:grad-ckpt 在 backward 重算 LM forward → LM 计算翻倍 → ViT 占比被稀释。grad-ckpt OFF 时同样代码是 ~1.49×。
- 缓存方案的真正价值**不在单步训练加速**,而在:(1) 跨 epoch / 跨实验的 ViT 前向**摊销**(只算一次,多次复用);(2) lakehouse / IO 故事(Lance 列存 + 向量检索 + Iceberg lineage)。
- **0 段错误**(500 步规模):`num_workers=0` 修复(Lance fork-unsafe tokio runtime 根因)在完整规模验证通过。此前误判的 "NCCL hang @ 436" 实为 DataLoader worker fork 后读 Lance 段错误。

## 磁盘
全程 37G free 稳定;A/B 结束后空 DELETEME 目录已清理;无训练进程残留写盘 → 可安全重启实例回收 vastai 占盘。
