# GRPO VLA 环境搭建 — 完整可复现记录

veRL-based GRPO RL fine-tuning of the nuScenes-planning VLA (B.5' Qwen2.5-VL-3B 3-cam).
从零搭建的**完整步骤 + 确切版本 + 所有 dep-hell fix + fork 改动**。配套精确版本见
`grpo_venv_freeze.txt`(215 包,实测可跑)。

> 结论先行:**系统 venv(torch 2.10 + AttnRes 栈)与 rollout 后端(sglang/vLLM)的 torch/flash_attn
> pin 不可调和**。唯一可行路径 = **独立 venv `/venv/grpo_vla`**,用 veRL 测过的版本组合。

---

## 0. 组件与位置
| 组件 | 位置 | 分支/版本 |
|---|---|---|
| veRL | `/workspace/verl` | branch `qwen25_vla_grpo`,335920a(在 v0.7.1 之上)|
| SGLang fork | `/sgl-workspace/sglang` | branch `qwen25_vla_inference`(serving 路径)|
| 独立 venv | `/venv/grpo_vla` | vLLM rollout 路径(最终采用)|
| GRPO 代码 | `DriveLM_VLM_Project/grpo_vla/` | reward/dataset_adapter/build_parquet/launcher |
| 数据 | `grpo_vla/data/*.parquet` | 12K train + 500 val(JPEG q75@448 + traj)|

## 1. 独立 venv 配方(最终可跑组合)
```bash
python -m venv /venv/grpo_vla
source /venv/grpo_vla/bin/activate
pip install -e /workspace/verl            # veRL 0.7.1(qwen25_vla_grpo 分支)
pip install vllm==0.12.0                   # rollout 后端(非 sglang——sglang 路径见 §4 卡点)
```
实测关键版本(`grpo_venv_freeze.txt` 完整):
```
torch==2.9.0          vllm==0.12.0         transformers==4.57.6
flash_attn==2.8.3     flashinfer-python==0.5.3   ray==2.55.1
tensordict==0.10.0    accelerate==1.13.0   numpy==1.26.4
```
> 要点:**flash_attn 必须 ≥2.6**(系统 venv 的是 broken stub,正是卡点 §4.8);torch 2.9 +
> vllm 0.12 是 veRL 0.7.1 验证过的组合,**别用系统 venv 的 torch 2.10**。

## 2. 数据 parquet 构建(可复用,无需重建)
```bash
python grpo_vla/build_parquet.py        # v2: JPEG q75 + 448px + 8-worker 并行,37 rows/s
# 产出 grpo_vla/data/{train,val}.parquet(1.35G),幂等
```
schema 坑(卡点 §4.2):veRL 要 **list-of-dicts**(非 str),build_parquet v2 已修。

## 3. veRL 侧改动(2 个 patch)
1. **architectures override**(commit 335920a):honor YAML `actor_rollout_ref.model.architectures`,
   否则 veRL 认不出 Qwen2.5-VL 的自定义 arch。
2. **`_get_input_embeds` 容忍补丁 + vLLM `bad_words_ids`**(任务 #145):veRL 的 embed 注入路径
   对 VLM 多模态 embedding 不兼容,打成容忍式;并给 vLLM 传 bad_words 屏蔽非法 token。

## 4. dep-hell 全部 9 个卡点 + 解法(按遇到顺序)
| # | 卡点 | 解法 |
|---|---|---|
| 1 | Hydra config search path 找不到 | launcher 里加 `--config-path`/search dir |
| 2 | veRL parquet schema(要 list-of-dicts 非 str)| 重写 build_parquet v2 |
| 3 | SGLang `_launch_subprocesses` 缺失 | fork commit 9b760bd:在 http_server 模块级暴露该别名 |
| 4 | sgl-kernel dist-info 缺失 | 写假 METADATA:`/usr/local/lib/python3.12/dist-packages/sgl_kernel-0.4.2.dist-info/`(local-only shim)|
| 5 | **SGLang scheduler CUDA `release_block` 初始化崩** | actor↔rollout GPU context 冲突;未在系统 venv 解决 → 改用 vLLM 路径(本 venv)|
| 6 | veRL `rollout.mode=sync` 被拒 | 改 `async` |
| 7 | vLLM 缺一堆包 | 装 cbor2/cpuinfo/ijson/watchfiles/depyf/model-hosting-container-standards 等 |
| 8 | **vLLM Qwen2.5-VL 需 flash_attn≥2.6**(`flash_attn.ops.triton.rotary`),系统 stub 不行 | 独立 venv 装 flash_attn 2.8.3 |
| 9 | reward 恒 -2.0(输出不可解析,任务 #146)| 模型输出解析/格式对齐 reward.py |

## 5. SGLang fork shim(serving 路径,`/sgl-workspace/sglang` branch `qwen25_vla_inference`)
```
9b760bd  shim: expose _launch_subprocesses at module level for veRL
80dea815 trunk-drift shims for B.5' Qwen2.5-VL 3cam serving (3 patches)
dc154e78 SGLANG_DISABLE_SHM_MM env + fp32 MLA fallback + base64 data-url + NaN trace
a6c4616  [Blackwell] KDA fp16 type-join + MoE shmem + AttnRes cuBLAS bypass
d6fb3bb  [VLM] AttnRes VLM: set_forward_context + MLA wiring + eps
```
> 注:fork 走 sglang serving;GRPO 训练最终用 vLLM(§1)。两条 rollout 后端都试过,vLLM 更快通关。

## 6. 验证(从零搭好后按序跑)
```bash
source /venv/grpo_vla/bin/activate
python grpo_vla/test_reward.py            # reward 5维 sanity(无需模型)→ 10/10 PASS
python grpo_vla/smoke_rollout_step.py     # veRL+vLLM 真跑 1 个 rollout step(验证整栈)
bash   grpo_vla/launch_grpo_b5prime.sh    # 正式 GRPO(指向 b5prime_3cam final/)
```

## 7. 当前状态(2026-05-27)
- 脚手架(reward/dataset/parquet/launcher/smoke)全部就绪、验证过。
- 独立 venv `/venv/grpo_vla` 曾搭成(本文件版本来自它的实测 freeze)。
- **本次为腾磁盘已删除该 venv**;按本文 §1 可 ~30min 重建(parquet/fork/veRL 分支都还在)。
- GRPO 正式训练**尚未跑到收敛**(脚手架完成,等重建 venv 后 smoke→正式)。

复现入口:本文 §1 + `grpo_venv_freeze.txt`。
