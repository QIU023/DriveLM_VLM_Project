# SATS → Efficient VLM 实验方案 (v2)

## 环境假设

```
模型:     Qwen2.5-VL-3B (student), Qwen2.5-VL-7B (teacher, 方向2.5)
数据集:   DriveLM-nuScenes
已有:     LoRA fine-tune pipeline, FasterVLM/PruMerge/PyramidDrop
评测:     DriveLM accuracy (baseline ~56.7%)
ViT:      depth=32, hidden=1280, heads=16, patch=14
          fullatt layers: [7, 15, 23, 31]
          3B/7B **共享同一 ViT**, 差异只在 LLM 和 merger
LLM:      3B = 36层/2048dim/16heads, 7B = 28层/3584dim/28heads
```

---

## 方向 2: Attention-Guided Token Compression

### 目标

用 ViT self-attention 的 **region-aware importance** 指导 token pruning/merging,
替代 FasterVLM/PruMerge/PyramidDrop 的启发式规则。

### Step 1: 离线提取 ViT Attention Map

**关键问题**: Qwen2.5-VL 默认 flash_attention, 不输出 attn weights。

**解决方案**: 加载时用 `attn_implementation="eager"`, 离线对 DriveLM 全量数据
提取 fullatt 层 [7,15,23,31] 的 attention map, 存为 `.pt` 文件。
训练时直接读取, 不影响速度。

```python
# scripts/extract_vit_attention.py

from transformers import Qwen2_5_VLForConditionalGeneration
import torch, json, os
from tqdm import tqdm

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    attn_implementation="eager",  # 关键: 不用 flash attn
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

FULLATT_LAYERS = [7, 15, 23, 31]
hooks, attn_store = [], {}

for idx in FULLATT_LAYERS:
    block = model.visual.blocks[idx]
    def make_hook(layer_idx):
        def hook_fn(mod, inp, out):
            # 需要确认 Qwen2.5-VL ViT 的 attn forward 返回格式
            # 可能需要修改源码让 attn 返回 weights
            if isinstance(out, tuple) and len(out) > 1:
                attn_store[layer_idx] = out[1].cpu()  # (B, H, N, N)
        return hook_fn
    hooks.append(block.attn.register_forward_hook(make_hook(idx)))

# 遍历 DriveLM keyframes, 提取并保存
# ...

for h in hooks:
    h.remove()
```

**注**: 如果修改 ViT forward 太麻烦, 备选方案是手动算:
```python
# 在 hook 里手动算 attention weights
Q, K = ...  # 从 input/output 推断
attn_weights = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
attn_weights = F.softmax(attn_weights, dim=-1)
```

### Step 2: nuScenes bbox → Patch Label

```python
# scripts/precompute_patch_labels.py

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import numpy as np, pickle

nusc = NuScenes(version='v1.0-trainval', dataroot='/path/to/nuscenes')

def get_patch_labels(sample_token, cam='CAM_FRONT', patch_size=14):
    """3D bbox → 2D bbox → patch-level label"""
    sample = nusc.get('sample', sample_token)
    sd = nusc.get('sample_data', sample['data'][cam])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    intrinsic = np.array(cs['camera_intrinsic'])
    
    # 图片尺寸 (需要和 processor resize 后一致)
    # Qwen2.5-VL 会 resize 到 28 的倍数
    img_h, img_w = ...  # 从 processor 获取
    
    H_p, W_p = img_h // patch_size, img_w // patch_size
    labels = np.zeros((H_p, W_p), dtype=np.int32)
    class_map = {}
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = nusc.get_box(ann_token)
        # 3D→2D 投影 (标准 nuScenes 流程, 需要坐标系变换)
        corners_3d = box.corners()  # (3, 8)
        # ... ego → cam 坐标变换 ...
        corners_2d = view_points(corners_cam, intrinsic, normalize=True)[:2]
        
        x1, y1 = corners_2d.min(axis=1).clip(0)
        x2, y2 = corners_2d.max(axis=1)
        
        cls = ann['category_name'].split('.')[0]
        if cls not in class_map:
            class_map[cls] = len(class_map) + 1
        
        # patch 坐标
        py1 = max(0, int(y1 // patch_size))
        py2 = min(H_p, int(np.ceil(y2 / patch_size)))
        px1 = max(0, int(x1 // patch_size))
        px2 = min(W_p, int(np.ceil(x2 / patch_size)))
        labels[py1:py2, px1:px2] = class_map[cls]
    
    return labels.flatten(), class_map  # (N_patches,)

# 对 DriveLM 全量 keyframe 预计算
results = {}
for item in drivelm_data:
    labels, cmap = get_patch_labels(item['sample_token'])
    results[item['sample_token']] = {'labels': labels, 'classes': cmap}

pickle.dump(results, open('patch_labels.pkl', 'wb'))
```

### Step 3: CRP → Token Importance

```python
# models/crp_token_importance.py

import torch

def crp_importance(attn_maps, patch_labels):
    """
    输入:
        attn_maps: dict {layer: (H, N, N)} — 多层 fullatt attention
        patch_labels: (N,) int — 0=bg, 1..C=foreground
    输出:
        importance: (N,) float — 每个 token 的重要性
    """
    N = patch_labels.shape[0]
    fg_mask = patch_labels > 0
    classes = patch_labels[fg_mask].unique()
    
    importance = torch.zeros(N)
    
    for attn in attn_maps.values():  # (H, N, N)
        attn_avg = attn.mean(dim=0)  # (N, N) 平均 heads
        
        for c in classes:
            c_mask = (patch_labels == c)
            # 类内 pooled attention vector (SATS 公式 1)
            pooled = attn_avg[c_mask].mean(dim=0)  # (N,)
            # 每个 token 被该类关注的程度 → importance
            importance += pooled
    
    # normalize
    importance = importance / (importance.max() + 1e-8)
    return importance


def select_tokens(importance, keep_ratio):
    """Top-k selection"""
    k = max(1, int(len(importance) * keep_ratio))
    return importance.topk(k).indices.sort().values


def merge_tokens(visual_tokens, patch_labels, attn_map):
    """
    同类 token 中 attention 最相似的做平均合并。
    类似 ToMe 但用 class label 限制合并范围。
    """
    # 只在同一 class 内部做 merge
    merged = visual_tokens.clone()
    classes = patch_labels.unique()
    
    for c in classes:
        if c == 0: continue  # skip bg
        idx = (patch_labels == c).nonzero(as_tuple=True)[0]
        if len(idx) < 2: continue
        
        # 同类 token 间的 attention similarity
        sub_attn = attn_map.mean(0)[idx][:, idx]  # (|c|, |c|)
        
        # 贪心合并: 最高 attention 的 pair 做平均
        # (简化版, 可以更精细)
        while len(idx) > max(1, len(idx) // 2):
            flat = sub_attn.triu(diagonal=1)
            if flat.max() < 0.1: break
            i, j = divmod(flat.argmax().item(), flat.shape[1])
            merged[0, idx[i]] = (merged[0, idx[i]] + merged[0, idx[j]]) / 2
            # 标记 j 为已合并 (后续 prune 掉)
            # ...
            break  # 简化: 只合并一轮
    
    return merged
```

### Step 4: 集成到训练 Pipeline

在现有 DriveLM fine-tune pipeline 中:

```python
# 训练时, 在 ViT 输出后插入 compression
visual_tokens = model.visual(pixel_values)  # (B, N, 1280)

# 读取预计算的 attention importance
importance = precomputed_importance[sample_token]  # (N,)

# 选择保留的 token
keep_idx = select_tokens(importance, keep_ratio=0.25)
compressed_tokens = visual_tokens[:, keep_idx, :]

# 送入 merger 和 LLM
merged = model.merger(compressed_tokens)
# ... 后续正常 forward
```

### 实验矩阵

| 方法 | 4× (120 tok) | 8× (60 tok) | 16× (30 tok) |
|------|-------------|------------|--------------|
| FasterVLM (baseline) | 57.4% | ? | 54.3% |
| PruMerge | 56.9% | ? | ? |
| PyramidDrop | 57.0% | ? | ? |
| **Attn-CRP prune** | ? | ? | ? |
| **Attn-CRP merge+prune** | ? | ? | ? |
| FasterVLM + Attn-CRP | ? | ? | ? |

消融:
- bbox label vs attention 自聚类 vs no region info (naive attn importance)
- 只用 layer [31] vs [7,15,23,31] 全部

---

## 方向 2.5: Region-Aware Relation Distillation (7B→3B)

### 目标

蒸馏 7B LLM decoder 中 visual token 的 attention pattern 到 3B,
用 CRP 把 O(N²) 的 token-pair relation 压缩到 O(C²) 的 region relation。

### 蒸馏位置

3B/7B 共享同一 ViT → ViT 层蒸馏无意义。
蒸馏在 **LLM decoder self-attention** 中 visual token 的子矩阵。

### Step 1: 提取 LLM 中 Visual Token Attention

```python
# models/llm_visual_attn.py

def get_visual_token_attention(model, inputs, visual_mask, target_layers):
    """
    Args:
        visual_mask: (B, seq_len) bool, True = visual token position
        target_layers: list of int
    Returns:
        {layer: (B, heads, N_vis, N_vis)}
    """
    out = model(**inputs, output_attentions=True)
    result = {}
    for l in target_layers:
        attn = out.attentions[l]  # (B, heads, seq, seq)
        B = attn.shape[0]
        vis_attns = []
        for b in range(B):
            vis_idx = visual_mask[b].nonzero(as_tuple=True)[0]
            sub = attn[b][:, vis_idx][:, :, vis_idx]  # (heads, Nv, Nv)
            vis_attns.append(sub)
        result[l] = torch.stack(vis_attns)
    return result
```

### Step 2: CRP on LLM Attention + Distillation Loss

```python
# losses/region_relation_loss.py

def region_pooled_attention(vis_attn, patch_labels, num_classes):
    """
    vis_attn: (heads, Nv, Nv)
    patch_labels: (Nv,)
    → (heads, C, C) region-level relation matrix
    """
    H = vis_attn.shape[0]
    C = num_classes
    R = torch.zeros(H, C, C, device=vis_attn.device)
    for ci in range(C):
        mi = (patch_labels == ci + 1)
        if mi.sum() == 0: continue
        for cj in range(C):
            mj = (patch_labels == cj + 1)
            if mj.sum() == 0: continue
            R[:, ci, cj] = vis_attn[:, mi][:, :, mj].mean(dim=(1, 2))
    return R


def region_relation_distill_loss(teacher_attns, student_attns,
                                  patch_labels, num_classes, 
                                  layer_map):
    """
    layer_map: {teacher_layer: student_layer}
    
    7B: 28 LLM layers, heads=28, dim=3584
    3B: 36 LLM layers, heads=16, dim=2048
    
    heads 不匹配 → 对 heads 维度做 mean 后再对齐 (scalar relation)
    或者只对齐 relation matrix (C, C), 不看 per-head
    """
    loss = 0.0
    for t_l, s_l in layer_map.items():
        R_t = region_pooled_attention(
            teacher_attns[t_l].mean(dim=1),  # avg over heads → (Nv, Nv)
            patch_labels, num_classes
        ).unsqueeze(0)  # (1, C, C)
        
        R_s = region_pooled_attention(
            student_attns[s_l].mean(dim=1),
            patch_labels, num_classes
        ).unsqueeze(0)
        
        loss += F.mse_loss(R_s, R_t)
    
    return loss / len(layer_map)
```

### Step 3: 训练流程

```python
# train_distill.py (伪代码)

teacher = load_7b_with_lora(freeze=True)
student = load_3b()
student_lora = apply_lora(student, rank=16)

# Layer mapping: 均匀采样
# 7B: 28 layers → 取 [6, 13, 20, 27]
# 3B: 36 layers → 取 [8, 17, 26, 35]
layer_map = {6: 8, 13: 17, 20: 26, 27: 35}

for batch in drivelm_loader:
    # Teacher forward (no grad)
    with torch.no_grad():
        t_out = teacher(**batch, output_attentions=True)
        t_vis_attn = extract_visual_attns(t_out, batch.visual_mask, 
                                           list(layer_map.keys()))
    
    # Student forward
    s_out = student(**batch, output_attentions=True)
    s_vis_attn = extract_visual_attns(s_out, batch.visual_mask,
                                       list(layer_map.values()))
    
    # Losses
    L_ce = s_out.loss  # autoregressive CE
    L_rrd = region_relation_distill_loss(
        t_vis_attn, s_vis_attn, 
        batch.patch_labels, batch.num_classes,
        layer_map
    )
    L_kd = kl_div(s_out.logits, t_out.logits)  # output KD
    
    loss = L_ce + lambda_a * L_rrd + lambda_d * L_kd
    loss.backward()
    optimizer.step()
```

### 实验矩阵

| 方法 | DriveLM Acc |
|------|-------------|
| 3B LoRA baseline | 56.7% |
| 3B + output KD from 7B | ? |
| 3B + RDist (全量 token-pair, LLaVA-KD style) | ? |
| **3B + RRD (ours, CRP region-level)** | ? |
| **3B + output KD + RRD** | ? |

消融:
- CRP (bbox label) vs Global Pooling vs No Pooling → 对标 SATS Table 4
- 不同 layer_map 策略
- λ_a 和 λ_d 敏感性

---

## 方向 3: Continual Learning (大纲, 暂缓)

### 相关工作

- **ϕ-DPO** (arXiv 2602.22601): DPO-based CL for LMMs, SOTA on CoIN/MLLM-CL
  - 证明 DPO loss 上下界约束 KL divergence → 替代传统 KD
  - Fairness DPO 解决 imbalanced data gradient bias
  - 引用了你的 SATS ([74])
  - 需要 16x A100, LLaVA v1.5 + Vicuna 7B

### 可能方向

- 在 ϕ-DPO 框架中加入 attention distillation 作为 DPO 的补充
- 简化版: DriveLM 上 2-stage CL (perception → planning)
- 需要构建 DPO preference data

### 暂缓原因

计算资源不足, benchmark 复杂度高, 留待后续有 multi-GPU 资源时实施。

---

## 执行优先级

```
Week 1: 方向 2
  - 离线提取 ViT attention map
  - 预计算 nuScenes bbox → patch labels
  - 实现 CRP importance + token selection
  - 跑 4×/8×/16× 压缩实验, 对比 FasterVLM baseline

Week 2: 方向 2.5
  - 实现 LLM visual attention 提取
  - 实现 region relation distillation loss
  - 7B→3B 蒸馏训练
  - 对比 output-only KD vs RRD vs 组合

Week 3: 整合 + 简历
  - 汇总实验数字
  - 更新 DriveLM 项目 bullet
  - 更新 SATS 项目 bullet
  - 面试叙事线排练
```
