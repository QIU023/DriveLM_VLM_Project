# 1-cam cached-vs-live A/B,grad-ckpt OFF(no AC),500 步

**日期** 2026-05-27 · **脚本** `run_ab_1cam_noAC.sh` · cache `nusc_1cam_native.lance`(450 行,700 video + 121 image tok)

## 配置(两臂完全一致)
| 项 | 值 |
|---|---|
| config | `nuscenes_planning_1cam_qwen3vl_NATIVE.yaml`(native ~2800 tok/cam → FasterVLM×4 → 700) |
| **AC / grad-ckpt** | **两套都 OFF**(`--no-grad-ckpt`:HF-side GC + FSDP2 plugin AC) |
| LBS | 1 · GA 1 · 8-GPU FSDP(no-AC 在 LBS=2 OOM,降到 1) |
| steps | 500 · 零 ckpt 写入(空目录 0 字节验证) |
| 唯一变量 | LIVE=每步 ViT(fastervlm×4) vs CACHED=读 int8 缓存 token、跳 ViT |

## 结果
| 臂 | 中位 s/it | wall(500 步) |
|---|---|---|
| LIVE | **1.79** | 956s |
| CACHED | **1.72** | 908s |

**speedup ≈ 1.04×(中位)/ 1.05×(wall)**。0 段错误、0 stall(`num_workers=0` 在 no-AC 下同样稳)。

## 与之前数据点合并 — 核心结论
| config | AC | #cam | per-step speedup |
|---|---|---|---|
| 3-cam | ON | 3 | **1.04×**(受控 A/B) |
| 3-cam | OFF | 3 | **1.49×**(Tier-2 fwd+bwd) |
| 1-cam | OFF | 1 | **1.04×**(本次) |

**为什么 1-cam 关了 AC 也只有 1.04×(不是预期的 ~1.5×):**

per-step speedup ≈ **ViT 占整步的比例**。关键:**ViT 是冻结的 → 只有 forward,没有 backward**。一个完整训练步 = `ViT_fwd + LM_fwd + LM_bwd + optimizer`,而 LM 的 backward+optimizer 才是大头。

- **cam 数**↑ → ViT_fwd 占比↑ → speedup↑(3-cam 有 3 份 ViT,1-cam 只有 1 份)
- **AC on** → LM forward 在 backward 里重算 → LM 部分翻倍 → ViT 占比被稀释 → speedup↓
- **1-cam + no-AC**:只有 1 份 ViT forward,相对 ~3500-token 的 LM fwd+bwd 太小 → 即使关了 AC 也只有 1.04×

所以用户"关 AC 会更快"的假设**对、但只在 ViT 占比大时(多 cam)才显著**;1-cam 下 ViT 本身太小,关不关 AC 都接近 1.04×。**1.49× 那个数是 3-cam + no-AC 特例**(多 cam 放大了 ViT 占比 + 不被 AC 稀释)。

## 这对"dataloader 加速"意味着什么
- **训练单步**层面:跳冻结-ViT 的收益高度依赖 ViT 占比,典型 1.04×(1-cam)~1.49×(3-cam no-AC);不是普适大加速。
- 真正普适的价值仍在:**纯 dataloader 管道 31×**(P0 micro-bench,IO+预处理+ViT 全省)+ **跨 epoch/跨实验 ViT 前向摊销** + lakehouse/远程存储 IO —— 这些在 24M/对象存储规模才主导,与 JD 对齐。

## 顺带修的 3 个 train_lora.py 真 bug(GC-off 这条从没被走过的路径)
1. `--no-grad-ckpt` flag:HF-side GC 在 FSDP+full_sft 下被无条件强开(line 2844),config 的 false 无效
2. GC 关后 `use_cache` 没自动关 → KV cache 让 attention key 翻倍
3. **FSDP2 plugin 独立的 `activation_checkpointing=True`**(line 2641)= 第二套 checkpointing,是 `2S must match S` 崩溃真凶;`--no-grad-ckpt` 现在一并关掉
