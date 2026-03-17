"""LoRA fine-tuning of Qwen2.5-VL on DriveLM data.

All hyperparameters are loaded from a YAML config file.
See configs/gh200.yaml and configs/4070ti.yaml for examples.

Usage:
  python train_lora.py --config configs/gh200.yaml --mini
  python train_lora.py --config configs/4070ti.yaml
  python train_lora.py --config configs/gh200.yaml --wandb
"""
import argparse
import json
import os
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


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
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
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset for testing")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate from config")
    parser.add_argument("--bs", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
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
    lr = args.lr if args.lr is not None else float(cfg["learning_rate"])
    batch_size = args.bs if args.bs is not None else cfg["batch_size"]
    grad_accum = cfg.get("grad_accum_steps", 1)
    num_epochs = args.epochs if args.epochs is not None else cfg["num_epochs"]
    max_length = cfg.get("max_length", 512)
    num_workers = cfg.get("num_workers", 0)
    save_every = cfg.get("save_every", 500)
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    data_dir = os.path.join(_BASE_DIR, "data_processed")
    output_dir = os.path.join(_BASE_DIR, "checkpoints_qwen25")
    os.makedirs(output_dir, exist_ok=True)

    eff_bs = batch_size * grad_accum
    print(f"Model: {model_id} | dtype: {dtype_str} | quantize: {quantize}")
    print(f"LoRA: r={lora_r} alpha={lora_alpha} dropout={lora_dropout}")
    print(f"BS={batch_size} x accum={grad_accum} = eff_bs={eff_bs} | LR={lr} | max_len={max_length}")
    print(f"Image pixels: {min_pixels} ~ {max_pixels} | workers={num_workers}")

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

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after model load: {gpu_mem:.2f} GB")

    # ============ Data setup ============
    train_file = os.path.join(data_dir, "train_mini.json" if args.mini else "train.json")
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

    num_batches = len(train_loader)
    total_steps = num_batches * num_epochs // grad_accum
    print(f"Training samples: {len(train_dataset)}")
    print(f"Steps per epoch: {num_batches} | Total opt steps: {total_steps}")

    # ============ Optimizer ============
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ============ Optional wandb ============
    if args.wandb:
        import wandb
        wandb.init(project="drivelm-qwen25vl", config={
            **cfg, "mini": args.mini, "lr": lr, "batch_size": batch_size,
        })

    # ============ Training loop ============
    print(f"\n{'='*60}")
    print(f"  Starting training | {num_epochs} epoch(s) | {total_steps} opt steps")
    print(f"{'='*60}\n")
    model.train()
    global_step = 0
    accum_loss = 0.0

    for epoch in range(num_epochs):
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
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            try:
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
                loss.backward()
                batch_loss = outputs.loss.item()
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    tqdm.write(f"  [SAVE] checkpoint-{global_step}")

        pbar.close()
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"\n  Epoch {epoch+1} done | "
            f"avg_loss={avg_loss:.4f} | LR={cur_lr:.2e} | "
            f"opt_steps={global_step}/{total_steps}\n"
        )

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
