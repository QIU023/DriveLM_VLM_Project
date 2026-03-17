# Session Context — 2026-03-17 云端服务器初始化

> 本文件记录了这次 Claude Code 会话的所有关键决策和上下文，供后续会话快速接续。

---

## 1. 完成的工作

### 环境搭建
- 在 Lambda Cloud GH200 (96GB) 上搭建了 Qwen2.5-VL 微调环境
- Conda env `qwen25vl`: Python 3.10, torch 2.10.0+cu128, transformers 5.3.0, peft 0.18.1, bitsandbytes 0.49.2
- flash-attn 当时正在编译，训练脚本不依赖它（自动 fallback 到 SDPA）
- HuggingFace 已登录（token 已配置），可访问 gated repo OpenDriveLab/DriveLM

### 数据下载
- 从 HuggingFace 下载了 DriveLM v1.1 数据（注意：repo 里只有 v1.1，没有 v1.0）
- QA JSON: `data/QA_dataset_nus/v1_1_train_nus.json` — 696 scenes, 377,955 QA pairs
- nuScenes 图片: `data/nuscenes/samples/` — 24,432 张图片，6 个摄像头
- 解压后删除了 zip 文件节省空间
- 注意：解压时有嵌套层 `nuscenes/nuscenes/samples`，已手动修正为 `nuscenes/samples`

### 项目结构整理
- 把 `scripts/`、`data_processed/`、`CLAUDE.md`、`README.md` 从 YQ 根目录移入 DriveLM/
- 所有脚本中的 Windows 路径 `F:/learning/...` 替换为基于 `__file__` 的相对路径

### 训练脚本重构
- `train_lora.py` 重写为 YAML 配置驱动，不再硬编码任何超参
- 新增 `--config` 必选参数，`--bs`/`--lr`/`--epochs` 可选 override
- 创建了两个配置文件：
  - `configs/gh200.yaml`: bf16, bs=8, no quant, max_pixels=512*28*28
  - `configs/4070ti.yaml`: 4-bit quant, bs=1, grad_accum=8
- 进度条从手写替换为 tqdm，显示 batch_loss / avg_loss / lr / opt_step / GPU memory
- `demo_inference.py` 也改为 YAML 配置 + `--lora` 参数支持任意 checkpoint

### 数据转换
- 运行 `convert_data.py` 生成了新的 data_processed/：
  - train.json: 359,057 samples
  - val.json: 18,898 samples
  - train_mini.json: 500 samples
- 类别分布：perception 43%, prediction 32.7%, planning 23.3%, behavior 1.1%

### Git 配置
- Remote: `git@github.com:QIU023/DriveLM_VLM_Project.git` (SSH)
- SSH key: `~/.ssh/id_ed25519`，已添加到 GitHub
- Git user: QIU023 / QIU023@users.noreply.github.com（repo-level config）
- 已 push 一次 commit 到 main

### 全量训练启动
- 用 `nohup` 启动了全量训练：bs=8, bf16, no quant
- 日志输出到 `logs/train_full.log`
- 每 500 opt step 保存一个 LoRA checkpoint
- 总计 ~44,882 opt steps（359,057 samples / bs=8）

---

## 2. 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| GH200 上是否量化 | 否（bf16） | bf16 计算速度更快，96GB 显存充裕，dequant 开销不值得 |
| Batch size | 8 | bs=16 OOM（图片 token 太多），bs=8 稳定 |
| 图片分辨率 | min=256×28², max=512×28² | 降低 max_pixels 是省显存最有效的手段 |
| DriveLM 版本 | v1.1 | HF repo 只有 v1.1，没有 v1.0 |
| 单视角 vs 多视角 | 单视角 (CAM_FRONT) | 简化起步，多视角是未来方向 3 |
| LoRA targets | q/k/v/o_proj + gate/up/down_proj | 覆盖 attention + MLP 全部投影层 |

---

## 3. 遇到的问题与解决

| 问题 | 解决 |
|------|------|
| HF gated repo 401 | 用 `huggingface_hub.login(token=...)` 登录 |
| QA JSON 文件名 404 | v1.0 不存在，改为下载 v1.1 |
| 空 JSON 文件（0 bytes） | 之前未登录时缓存的空文件，删除后重新下载 |
| git push 403 | remote 指向原始 OpenDriveLab repo，改为自己的 fork |
| SSH host key verification failed | `ssh-keyscan github.com >> ~/.ssh/known_hosts` |
| git author identity unknown | `git config user.name/email` 设置 repo-level |
| YAML `2e-4` 被读为字符串 | 加 `float()` 转换 |
| bs=16 OOM | 降到 bs=8 + 降低 max_pixels |
| `import yaml` ModuleNotFoundError | `pip install pyyaml` |

---

## 4. 用户背景与偏好

- **求职方向**: 自动驾驶 + 通用多模态 CV，Efficient VLM + 蒸馏方向
- **学术背景**: Pattern Recognition 2023 论文，feature-level + logit-level KD 用于持续学习
- **技术定位**: 小模型 + 快速微调，不做通用预训练
- **覆盖场景**: 自动驾驶、多模态推荐、视频理解
- **偏好**: 不要直接跑长时间命令，给出完整命令让用户自己执行；代码修改前先检查再改

---

## 5. 待完成事项

- [ ] 全量训练完成后，用 `demo_inference.py` 对比 base model vs LoRA
- [ ] 挑几个中间 checkpoint（step 5000/10000/20000）做收敛分析
- [ ] `train_lora_qwen35.py` 和 `demo_inference_qwen35.py` 尚未更新为 YAML 配置
- [ ] flash-attn 编译完成后可启用以加速 attention
- [ ] 开始方向 10（VLM 持续学习）— 需要准备推荐域和视频域的数据
- [ ] git push 最新的脚本修改（YAML 重构 + tqdm + 新 configs）

---

## 6. 重要文件路径

```
/lambda/nfs/YQ/DriveLM/                     # 项目根目录
/lambda/nfs/YQ/DriveLM/configs/gh200.yaml   # 当前使用的训练配置
/lambda/nfs/YQ/DriveLM/logs/train_full.log  # 全量训练日志
/lambda/nfs/YQ/DriveLM/checkpoints_qwen25/  # LoRA checkpoints
/lambda/nfs/YQ/DriveLM/docs/exploration_directions.md  # 10 个探索方向规划
/home/ubuntu/miniconda3/envs/qwen25vl/      # Conda 环境
/home/ubuntu/.ssh/id_ed25519                # GitHub SSH key
```
