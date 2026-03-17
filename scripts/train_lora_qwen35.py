"""QLoRA fine-tuning of Qwen3.5-4B on DriveLM data.

Qwen3.5 is natively multimodal (no separate VL variant needed).
Architecture: Gated DeltaNet (linear attention) + standard attention hybrid,
with an integrated vision encoder.

Usage:
  # Quick test with mini dataset:
  python train_lora.py --mini

  # Full training:
  python train_lora.py
"""
import argparse
import json
import os
import time
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

# ============ Config ============
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, "data_processed")
OUTPUT_DIR = os.path.join(_BASE_DIR, "checkpoints")
MODEL_ID = "Qwen/Qwen3.5-4B"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 1  # 4070 Ti 12GB, use gradient accumulation instead
GRAD_ACCUM_STEPS = 8  # effective batch size = 8
NUM_EPOCHS = 1
MAX_LENGTH = 1024
LOG_EVERY = 10
SAVE_EVERY = 500


class DriveLMDataset(Dataset):
    """Dataset for DriveLM QA fine-tuning with Qwen3.5."""

    def __init__(self, data_path, processor, max_length=1024):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        # Build messages for the processor (skip system message for simplicity)
        proc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            proc_messages.append(msg)

        # Extract images from messages and replace file:// URIs with PIL images
        # so the processor can handle them directly (no qwen_vl_utils needed)
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

        # Apply chat template to get text with image placeholders
        # Disable thinking mode: DriveLM answers are short factual responses
        text = self.processor.apply_chat_template(
            clean_messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

        # Tokenize with processor (handles image processing + text tokenization)
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
        mm_token_type_ids = inputs.get("mm_token_type_ids")

        # Truncate if too long
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.squeeze(0)[: self.max_length]

        # Create labels: mask everything before the assistant's response
        labels = input_ids.clone()
        # Qwen3.5 uses <|im_start|>assistant\n to mark assistant response
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
        if mm_token_type_ids is not None:
            if mm_token_type_ids.dim() > 1:
                mm_token_type_ids = mm_token_type_ids.squeeze(0)
            result["mm_token_type_ids"] = mm_token_type_ids

        return result


def collate_fn(batch):
    """Custom collate that handles variable-size sequences, pixel_values, and mm_token_type_ids."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_token_id = 0  # Qwen uses 0 as pad

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    padded_mm_token_type_ids = []
    has_mm_token_type_ids = "mm_token_type_ids" in batch[0]

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
        if has_mm_token_type_ids:
            # mm_token_type_ids: 0=text, 1=image, 2=video; pad with 0 (text)
            padded_mm_token_type_ids.append(
                torch.cat([item["mm_token_type_ids"], torch.zeros(pad_len, dtype=item["mm_token_type_ids"].dtype)])
            )

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }
    if has_mm_token_type_ids:
        result["mm_token_type_ids"] = torch.stack(padded_mm_token_type_ids)

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
    parser.add_argument("--mini", action="store_true", help="Use mini dataset for testing")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============ Model setup ============
    print("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Limit image resolution to save memory
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = 256 * 28 * 28  # ~200k pixels
        processor.image_processor.max_pixels = 512 * 28 * 28  # ~400k pixels

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config — Qwen3.5 has hybrid layers:
    #   full_attention layers: q_proj, k_proj, v_proj, o_proj (standard attention)
    #   linear_attention layers: GatedDeltaNet with out_proj
    #   All layers have MLP: gate_proj, up_proj, down_proj
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # full attention layers
            "out_proj",                                # GatedDeltaNet output
            "gate_proj", "up_proj", "down_proj",       # MLP (all layers)
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after model load: {gpu_mem:.2f} GB")

    # ============ Data setup ============
    train_file = os.path.join(DATA_DIR, "train_mini.json" if args.mini else "train.json")
    print(f"Loading dataset: {train_file}")

    train_dataset = DriveLMDataset(train_file, processor, max_length=MAX_LENGTH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Steps per epoch: {len(train_loader)}")
    total_steps = len(train_loader) * args.epochs // GRAD_ACCUM_STEPS
    print(f"Total optimization steps: {total_steps}")

    # ============ Optimizer ============
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ============ Optional wandb ============
    if args.wandb:
        import wandb
        wandb.init(project="drivelm-qwen3.5", config={
            "model": MODEL_ID, "lora_r": LORA_R, "lr": args.lr,
            "batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "mini": args.mini,
        })

    # ============ Training loop ============
    print("\n=== Starting training ===")
    num_batches = len(train_loader)
    print(f"  Batches per epoch: {num_batches} | Grad accum: {GRAD_ACCUM_STEPS} | Opt steps: {total_steps}\n")
    model.train()
    global_step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            try:
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()
                batch_loss = outputs.loss.item()
                accum_loss += loss.item()
                epoch_loss_sum += batch_loss
                epoch_loss_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"\n  [OOM] batch {step+1}/{num_batches}, skipping")
                    optimizer.zero_grad()
                    accum_loss = 0.0
                    continue
                raise

            # Progress bar: every batch
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            pct = (step + 1) / num_batches * 100
            bar_len = 30
            filled = int(bar_len * (step + 1) // num_batches)
            bar = "=" * filled + ">" + "." * (bar_len - filled - 1)
            elapsed = time.time() - epoch_start
            eta = elapsed / (step + 1) * (num_batches - step - 1)
            print(
                f"\r  Epoch {epoch+1}/{args.epochs} [{bar}] "
                f"{step+1}/{num_batches} ({pct:4.1f}%) | "
                f"batch_loss: {batch_loss:.4f} | "
                f"GPU: {gpu_mem:.1f}GB | "
                f"ETA: {int(eta//60)}m{int(eta%60):02d}s",
                end="", flush=True,
            )

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.wandb:
                    import wandb
                    lr = scheduler.get_last_lr()[0]
                    wandb.log({
                        "loss": accum_loss, "batch_loss": batch_loss,
                        "lr": lr, "gpu_mem": gpu_mem,
                    }, step=global_step)
                accum_loss = 0.0

                if global_step % SAVE_EVERY == 0:
                    save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    print(f"\n  [SAVE] checkpoint-{global_step}")

        # Epoch summary
        epoch_elapsed = time.time() - epoch_start
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        lr = scheduler.get_last_lr()[0]
        print(
            f"\n  Epoch {epoch+1} done | "
            f"avg_loss: {avg_loss:.4f} | LR: {lr:.2e} | "
            f"time: {int(epoch_elapsed//60)}m{int(epoch_elapsed%60):02d}s\n"
        )

    # ============ Save final model ============
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
