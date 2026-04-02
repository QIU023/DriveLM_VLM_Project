"""LoRA fine-tuning of Qwen2.5-VL on DriveLM data.

All hyperparameters are loaded from a YAML config file.
Supports visual token compression experiments via compress_method / compress_ratio.

Usage:
  python train_lora.py --config configs/gh200.yaml --mini
  python train_lora.py --config configs/baseline.yaml
  python train_lora.py --config configs/avg_pool_c4.yaml
  python train_lora.py --config configs/gh200.yaml --bs 4 --epochs 1
"""
import argparse
import json
import math
import os
import sys
import time
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path):
    """Load YAML config with optional base_config inheritance."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base_config" in cfg:
        base_path = cfg.pop("base_config")
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_cfg = load_config(base_path)
        base_cfg.update(cfg)
        cfg = base_cfg
    return cfg


class DriveLMDataset(Dataset):
    """Dataset for DriveLM QA fine-tuning with Qwen2.5-VL."""

    def __init__(self, data_path, processor, max_length=512):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        # Build messages (skip system message for simplicity)
        proc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            proc_messages.append(msg)

        # Extract images and build clean messages
        images = []
        image_name = ""
        clean_messages = []
        for msg in proc_messages:
            if isinstance(msg["content"], list):
                new_content = []
                for part in msg["content"]:
                    if part.get("type") == "image":
                        image_path = part["image"]
                        if image_path.startswith("file://"):
                            image_path = image_path[7:]
                        images.append(Image.open(image_path).convert("RGB"))
                        image_name = os.path.basename(image_path)
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(part)
                clean_messages.append({"role": msg["role"], "content": new_content})
            else:
                clean_messages.append(msg)

        # Apply chat template
        text = self.processor.apply_chat_template(
            clean_messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize with processor
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
        )

        # Squeeze batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Truncate if too long
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        # Create labels: mask everything before the assistant's response
        labels = input_ids.clone()
        assistant_token_str = "<|im_start|>assistant\n"
        assistant_tokens = self.processor.tokenizer.encode(
            assistant_token_str, add_special_tokens=False
        )
        input_list = input_ids.tolist()
        assistant_start = -1
        for i in range(len(input_list) - len(assistant_tokens) + 1):
            if input_list[i : i + len(assistant_tokens)] == assistant_tokens:
                assistant_start = i + len(assistant_tokens)
                break

        if assistant_start > 0:
            labels[:assistant_start] = -100
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0) if pixel_values.dim() > 3 else pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw

        # Store image name for CRP importance lookup
        result["image_name"] = image_name if 'image_name' in dir() else ""

        return result


def collate_fn(batch):
    """Custom collate that handles variable-size sequences and pixel_values."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_token_id = 0  # Qwen uses 0 as pad

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        padded_input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=item["input_ids"].dtype)])
        )
        padded_attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=item["attention_mask"].dtype)])
        )
        padded_labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=item["labels"].dtype)])
        )

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }

    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.cat(
            [item["pixel_values"].unsqueeze(0) if item["pixel_values"].dim() == 3 else item["pixel_values"] for item in batch],
            dim=0,
        )
    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = torch.cat(
            [item["image_grid_thw"].unsqueeze(0) if item["image_grid_thw"].dim() == 1 else item["image_grid_thw"] for item in batch],
            dim=0,
        )
    if "image_name" in batch[0]:
        result["image_names"] = [item["image_name"] for item in batch]
    return result


# --------------- Visual token compression ---------------

def get_base_model(model):
    """Unwrap PEFT to get the original Qwen2.5-VL model."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


_CRP_IMPORTANCE = {}  # Global cache for CRP precomputed importance


def forward_with_compression(model, batch, compress_method, compress_ratio, image_token_id):
    """Forward pass with optional visual token compression.

    For compress_method == "none", falls through to the normal model forward.
    Otherwise:
      1. Run vision encoder on the base model
      2. Compress visual tokens
      3. Adjust input_ids (remove excess image placeholders)
      4. Build inputs_embeds with compressed visual tokens
      5. Forward through LoRA-wrapped LLM with proper 3D RoPE positions
    """
    # Strip non-tensor keys before model forward
    image_names = batch.pop("image_names", [])
    if compress_method == "none" or "pixel_values" not in batch:
        return model(**batch)

    from visual_compress import compress_visual_tokens

    base = get_base_model(model)
    device = batch["input_ids"].device

    # 1. Vision encoder (no grad — LoRA is only on LLM layers)
    vis_dtype = next(base.model.visual.parameters()).dtype
    with torch.no_grad():
        vis_out = base.model.visual(batch["pixel_values"].to(vis_dtype), grid_thw=batch["image_grid_thw"])
        image_embeds = vis_out.pooler_output if hasattr(vis_out, "pooler_output") else vis_out
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]
        image_embeds = image_embeds.detach()
    del vis_out  # free last_hidden_state

    # 2. Compress — use post-merger grid (the merger does 2x2 spatial merge,
    #    so actual token grid is grid_thw with h/2, w/2)
    raw_grid = batch["image_grid_thw"]
    merge_size = getattr(base.model.visual, "spatial_merge_size", 2)
    post_grid = raw_grid.clone()
    post_grid[:, 1] = raw_grid[:, 1] // merge_size
    post_grid[:, 2] = raw_grid[:, 2] // merge_size
    # Build importance list for CRP methods
    importance_list = None
    if compress_method in ("crp", "crp_merge") and _CRP_IMPORTANCE:
        importance_list = [_CRP_IMPORTANCE.get(n) for n in image_names]

    compressed, new_grid_thw = compress_visual_tokens(image_embeds, post_grid, compress_method, compress_ratio, importance_list=importance_list)
    del image_embeds  # free pre-compression tokens

    # per-image token counts
    orig_counts = (post_grid[:, 0] * post_grid[:, 1] * post_grid[:, 2]).tolist()
    new_counts = (new_grid_thw[:, 0] * new_grid_thw[:, 1] * new_grid_thw[:, 2]).tolist()

    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]
    labels = batch["labels"]
    B = input_ids.shape[0]

    # 3. Remove excess image-placeholder tokens from each sample
    new_ids_list, new_mask_list, new_lab_list = [], [], []
    img_idx = 0  # pointer into the per-image counts

    for b in range(B):
        ids = input_ids[b]
        msk = attn_mask[b]
        lab = labels[b]

        img_pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
        n_img = len(img_pos)

        if n_img == 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            new_lab_list.append(lab)
            continue

        n_keep = int(new_counts[img_idx])
        img_idx += 1
        n_remove = n_img - n_keep

        if n_remove <= 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            new_lab_list.append(lab)
            continue

        # remove from the END of the image-placeholder block
        remove_pos = img_pos[n_keep:]
        keep = torch.ones(len(ids), dtype=torch.bool, device=device)
        keep[remove_pos] = False
        new_ids_list.append(ids[keep])
        new_mask_list.append(msk[keep])
        new_lab_list.append(lab[keep])

    # 4. Pad to max length
    max_len = max(t.shape[0] for t in new_ids_list)
    for i in range(B):
        pad = max_len - new_ids_list[i].shape[0]
        if pad > 0:
            new_ids_list[i] = torch.cat([new_ids_list[i], torch.zeros(pad, dtype=new_ids_list[i].dtype, device=device)])
            new_mask_list[i] = torch.cat([new_mask_list[i], torch.zeros(pad, dtype=new_mask_list[i].dtype, device=device)])
            new_lab_list[i] = torch.cat([new_lab_list[i], torch.full((pad,), -100, dtype=new_lab_list[i].dtype, device=device)])

    new_input_ids = torch.stack(new_ids_list)
    new_attn_mask = torch.stack(new_mask_list)
    new_labels = torch.stack(new_lab_list)

    # 5. Build inputs_embeds
    inputs_embeds = base.model.language_model.embed_tokens(new_input_ids).clone()
    img_mask = new_input_ids == image_token_id
    inputs_embeds[img_mask] = compressed.to(inputs_embeds.dtype)

    # 6. Forward — pass input_ids for 3D RoPE position computation,
    #    inputs_embeds for actual content, new_grid_thw for spatial dims
    outputs = model(
        input_ids=new_input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=new_attn_mask,
        image_grid_thw=new_grid_thw,
        labels=new_labels,
    )
    return outputs


@torch.no_grad()
def validate(model, val_loader, compress_method, compress_ratio, image_token_id, val_batches, device):
    """Run validation for val_batches batches and return avg loss + token accuracy."""
    model.eval()
    total_loss, count = 0.0, 0
    correct_tokens, total_tokens = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= val_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            try:
                outputs = forward_with_compression(model, batch, compress_method, compress_ratio, image_token_id)
                total_loss += outputs.loss.item()
                count += 1
                logits = outputs.logits[:, :-1, :]
                labels = batch["labels"][:, 1:]
                mask = labels != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    correct_tokens += (preds[mask] == labels[mask]).sum().item()
                    total_tokens += mask.sum().item()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise
    model.train()
    val_loss = total_loss / max(count, 1)
    val_acc = correct_tokens / max(total_tokens, 1)
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset for testing")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate from config")
    parser.add_argument("--bs", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--compress-method", type=str, default=None, help="Override compress_method")
    parser.add_argument("--compress-ratio", type=int, default=None, help="Override compress_ratio")
    parser.add_argument("--experiment", type=str, default=None, help="Override experiment name")
    parser.add_argument("--val-every", type=int, default=None, help="Validate every N opt steps")
    parser.add_argument("--val-batches", type=int, default=None, help="Number of val batches")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir (e.g. checkpoints_qwen25/crp_c8/checkpoint-66000)")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop training after N optimizer steps")
    args = parser.parse_args()

    # ============ Load config ============
    cfg = load_config(args.config)
    print(f"Config: {args.config}")

    model_id = cfg["model_id"]
    lora_r = cfg["lora_r"]
    lora_alpha = cfg["lora_alpha"]
    lora_dropout = cfg["lora_dropout"]
    lora_targets = cfg["lora_target_modules"]
    quantize = cfg.get("quantize", False)
    dtype_str = cfg.get("dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_str)
    lr = args.lr if args.lr is not None else float(cfg.get("learning_rate", cfg.get("lr", 2e-4)))
    batch_size = args.bs if args.bs is not None else cfg["batch_size"]
    grad_accum = cfg.get("grad_accum_steps", 1)
    num_epochs = args.epochs if args.epochs is not None else cfg.get("num_epochs", cfg.get("epochs", 1))
    max_length = cfg.get("max_length", 512)
    num_workers = cfg.get("num_workers", 0)
    save_every = cfg.get("save_every", 500)
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    # Compression & experiment settings
    compress_method = args.compress_method or cfg.get("compress_method", "none")
    compress_ratio = args.compress_ratio or cfg.get("compress_ratio", 1)
    experiment = args.experiment or cfg.get("experiment", "default")
    val_every = args.val_every or cfg.get("val_every", 0)
    val_batches = args.val_batches or cfg.get("val_batches", 50)

    data_dir = os.path.join(_BASE_DIR, "data_processed")
    output_dir = os.path.join(_BASE_DIR, "checkpoints_qwen25", experiment)
    os.makedirs(output_dir, exist_ok=True)

    eff_bs = batch_size * grad_accum
    print(f"Experiment: {experiment}")
    print(f"Model: {model_id} | dtype: {dtype_str} | quantize: {quantize}")
    print(f"LoRA: r={lora_r} alpha={lora_alpha} dropout={lora_dropout}")
    print(f"BS={batch_size} x accum={grad_accum} = eff_bs={eff_bs} | LR={lr} | max_len={max_length}")
    print(f"Image pixels: {min_pixels} ~ {max_pixels} | workers={num_workers}")
    print(f"Compression: {compress_method} ratio={compress_ratio}")
    if val_every > 0:
        print(f"Validation: every {val_every} opt steps, {val_batches} batches")

    # ============ Model setup ============
    load_kwargs = {"device_map": "auto"}

    if quantize:
        print(f"Loading with {cfg.get('quant_bits', 4)}-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(cfg.get("quant_bits", 4) == 4),
            load_in_8bit=(cfg.get("quant_bits", 4) == 8),
            bnb_4bit_use_double_quant=cfg.get("double_quant", True),
            bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        print(f"Loading in {dtype_str} (no quantization)...")
        load_kwargs["torch_dtype"] = compute_dtype

    load_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels

    # Prepare for training
    if quantize:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()

    # Gradient checkpointing: trade compute for memory (critical for large models)
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.resume:
        # Resume: load LoRA weights from checkpoint instead of init new
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(_BASE_DIR, args.resume)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
        print(f"Resumed LoRA from {resume_path}")
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Image token id for compression
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"Image token id: {image_token_id}")

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after model load: {gpu_mem:.2f} GB")

    # ============ Load CRP importance (if needed) ============
    global _CRP_IMPORTANCE
    if compress_method in ("crp", "crp_merge"):
        crp_path = cfg.get("crp_importance_path", os.path.join(_BASE_DIR, "precomputed", "crp_importance.pt"))
        if not os.path.isabs(crp_path):
            crp_path = os.path.join(_BASE_DIR, crp_path)
        if os.path.exists(crp_path):
            _CRP_IMPORTANCE = torch.load(crp_path, weights_only=True)
            print(f"Loaded CRP importance for {len(_CRP_IMPORTANCE)} images")
        else:
            print(f"[WARN] CRP importance not found at {crp_path}, falling back to L2 norm")

    # ============ Data setup ============
    default_train = "train_mini.json" if args.mini else "train.json"
    train_file = os.path.join(data_dir, cfg.get("train_file", default_train))
    val_file = os.path.join(data_dir, "val.json")
    print(f"Loading dataset: {train_file}")

    train_dataset = DriveLMDataset(train_file, processor, max_length=max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if val_every > 0 and os.path.exists(val_file):
        val_dataset = DriveLMDataset(val_file, processor, max_length=max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"Validation samples: {len(val_dataset)}")

    num_batches = len(train_loader)
    total_steps = num_batches * num_epochs // grad_accum
    print(f"Training samples: {len(train_dataset)}")
    print(f"Steps per epoch: {num_batches} | Total opt steps: {total_steps}")

    # ============ Optimizer ============
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
    )

    # ============ Resume training state ============
    resume_step = 0
    resume_epoch = 0
    if args.resume:
        state_path = os.path.join(
            args.resume if os.path.isabs(args.resume) else os.path.join(_BASE_DIR, args.resume),
            "training_state.pt"
        )
        if os.path.exists(state_path):
            state = torch.load(state_path, weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            resume_step = state["global_step"]
            resume_epoch = state.get("epoch", 0)
            print(f"Resumed optimizer/scheduler from step {resume_step}, epoch {resume_epoch}")
        else:
            print(f"[WARN] No training_state.pt found, resuming LoRA weights only (optimizer reset)")

    # ============ Optional wandb ============
    if args.wandb:
        import wandb
        wandb.init(project="drivelm-qwen25vl", name=experiment, config={
            **cfg, "mini": args.mini, "lr": lr, "batch_size": batch_size,
            "compress_method": compress_method, "compress_ratio": compress_ratio,
        })

    # ============ Training loop ============
    print(f"\n{'='*60}")
    print(f"  Starting training | {experiment} | {num_epochs} epoch(s) | {total_steps} opt steps")
    print(f"  Compression: {compress_method} ratio={compress_ratio}")
    print(f"{'='*60}\n")
    model.train()
    global_step = resume_step
    accum_loss = 0.0
    skip_batches = resume_step * grad_accum if resume_step > 0 else 0

    for epoch in range(resume_epoch, num_epochs):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        pbar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )

        for step, batch in pbar:
            # Skip already-trained batches on resume
            if skip_batches > 0:
                skip_batches -= 1
                if skip_batches % 1000 == 0:
                    pbar.set_postfix_str(f"skipping... {skip_batches} left")
                continue

            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            try:
                outputs = forward_with_compression(
                    model, batch, compress_method, compress_ratio, image_token_id
                )
                loss = outputs.loss / grad_accum
                batch_loss = outputs.loss.item()

                # NaN guard: skip bad batches before they poison the model
                if not math.isfinite(batch_loss):
                    tqdm.write(f"[NaN] batch {step+1}/{num_batches}, loss={batch_loss}, skipping")
                    optimizer.zero_grad()
                    accum_loss = 0.0
                    continue

                loss.backward()
                accum_loss += loss.item()
                epoch_loss_sum += batch_loss
                epoch_loss_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    tqdm.write(f"[OOM] batch {step+1}/{num_batches}, skipping")
                    optimizer.zero_grad()
                    accum_loss = 0.0
                    continue
                raise

            # Update tqdm postfix every batch
            avg_loss = epoch_loss_sum / epoch_loss_count
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            cur_lr = scheduler.get_last_lr()[0] if global_step > 0 else lr
            pbar.set_postfix_str(
                f"batch_loss={batch_loss:.4f} | avg_loss={avg_loss:.4f} | "
                f"lr={cur_lr:.2e} | opt_step={global_step}/{total_steps} | "
                f"GPU={gpu_mem:.1f}GB"
            )

            if (step + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Skip optimizer step if gradients are NaN/Inf
                if not math.isfinite(grad_norm.item()):
                    tqdm.write(f"[NaN grad] step {global_step}, grad_norm={grad_norm.item()}, skipping update")
                    optimizer.zero_grad()
                    accum_loss = 0.0
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.wandb:
                    import wandb
                    cur_lr = scheduler.get_last_lr()[0]
                    wandb.log({
                        "loss": accum_loss, "batch_loss": batch_loss,
                        "avg_loss": avg_loss, "lr": cur_lr, "gpu_mem": gpu_mem,
                    }, step=global_step)
                accum_loss = 0.0

                if global_step % save_every == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    # Save training state for resume
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_idx": step,
                    }, os.path.join(save_path, "training_state.pt"))
                    tqdm.write(f"  [SAVE] checkpoint-{global_step}")

                # Validation
                if val_every > 0 and val_loader is not None and global_step % val_every == 0:
                    val_loss, val_acc = validate(
                        model, val_loader, compress_method, compress_ratio,
                        image_token_id, val_batches, model.device,
                    )
                    tqdm.write(f"  [VAL] step={global_step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
                    if args.wandb:
                        import wandb
                        wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)

                if args.max_steps and global_step >= args.max_steps:
                    tqdm.write(f"  [STOP] Reached max_steps={args.max_steps}")
                    pbar.close()
                    break

        pbar.close()
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"\n  Epoch {epoch+1} done | "
            f"avg_loss={avg_loss:.4f} | LR={cur_lr:.2e} | "
            f"opt_steps={global_step}/{total_steps}\n"
        )
        if args.max_steps and global_step >= args.max_steps:
            break

    # ============ Save final model ============
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
