"""Knowledge distillation: Qwen2.5-VL 7B → 3B with Region-Aware Relation Distillation.

Loads teacher (7B, frozen) and student (3B + LoRA) simultaneously on GB200.
Loss = L_ce + λ_kd * L_kd + λ_rrd * L_rrd

Usage:
  python scripts/train_distill.py --config configs/distill_7b_3b.yaml
  python scripts/train_distill.py --config configs/distill_7b_3b.yaml --mini
"""

import argparse
import json
import math
import os
import sys
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_lora import load_config, DriveLMDataset, collate_fn
from region_relation_loss import region_relation_distill_loss


# ===================== Helpers =====================

def get_base_model(model):
    """Unwrap PEFT to get the original model."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def get_llm_layers(model):
    """Get LLM decoder layers from a (possibly PEFT-wrapped) model."""
    base = get_base_model(model)
    # transformers 5.x: model.language_model.layers
    # transformers 4.x: model.layers
    m = base.model
    if hasattr(m, "language_model"):
        return m.language_model.layers
    return m.layers


def kl_div_loss(student_logits, teacher_logits, temperature=2.0, labels=None):
    """Output-level KL divergence loss with temperature scaling.

    Only computes KD loss on positions where labels != -100 (valid tokens).
    Handles vocab size mismatch by truncating to the smaller vocab.
    """
    # Align vocab sizes (3B and 7B may differ slightly)
    v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_logits = student_logits[..., :v]
    teacher_logits = teacher_logits[..., :v]

    if labels is not None:
        mask = labels != -100
        if not mask.any():
            return torch.tensor(0.0, device=student_logits.device)
        s_log = F.log_softmax(student_logits[mask] / temperature, dim=-1)
        t_prob = F.softmax(teacher_logits[mask] / temperature, dim=-1)
    else:
        s_log = F.log_softmax(student_logits / temperature, dim=-1)
        t_prob = F.softmax(teacher_logits / temperature, dim=-1)

    return F.kl_div(s_log, t_prob, reduction="batchmean") * (temperature ** 2)


def register_qk_hooks(model, layer_indices, store, detach=True):
    """Register forward hooks on Q, K projections to capture outputs."""
    hooks = []
    layers = get_llm_layers(model)
    for idx in layer_indices:
        attn = layers[idx].self_attn

        def make_hook(key, do_detach):
            def hook_fn(mod, inp, out):
                store[key] = out.detach() if do_detach else out
            return hook_fn

        hooks.append(attn.q_proj.register_forward_hook(make_hook(f"q_{idx}", detach)))
        hooks.append(attn.k_proj.register_forward_hook(make_hook(f"k_{idx}", detach)))
    return hooks


def validate_distill(student, val_loader, val_batches, device):
    """Run validation and return avg CE loss and token accuracy on student."""
    student.eval()
    total_loss, count = 0.0, 0
    correct_tokens, total_tokens = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= val_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            _ = batch.pop("image_names", [])
            try:
                out = student(**batch)
                total_loss += out.loss.item()
                count += 1
                # Token-level accuracy: compare argmax predictions with labels
                logits = out.logits[:, :-1, :]  # shift: predict next token
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
    student.train()
    val_loss = total_loss / max(count, 1)
    val_acc = correct_tokens / max(total_tokens, 1)
    return val_loss, val_acc


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint dir (e.g. checkpoints_qwen25/distill_32b_3b/checkpoint-5000)")
    parser.add_argument("--bs", type=int, default=None, help="Override batch_size")
    parser.add_argument("--val-every", type=int, default=None, help="Override val_every")
    parser.add_argument("--val-batches", type=int, default=None, help="Override val_batches")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config}")

    teacher_model_id = cfg["teacher_model_id"]
    student_model_id = cfg["student_model_id"]
    teacher_lora = cfg.get("teacher_lora")
    dtype_str = cfg.get("dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_str)

    lr = float(cfg.get("learning_rate", 2e-4))
    batch_size = args.bs or cfg["batch_size"]
    grad_accum = cfg.get("grad_accum_steps", 1)
    num_epochs = cfg.get("num_epochs", 1)
    max_length = cfg.get("max_length", 2048)
    num_workers = cfg.get("num_workers", 0)
    save_every = cfg.get("save_every", 1000)
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    lambda_kd = cfg.get("lambda_kd", 1.0)
    lambda_crp = cfg.get("lambda_crp", cfg.get("lambda_rrd", 0.5))   # CRP O(C²)
    lambda_rdist = cfg.get("lambda_rdist", 0.0)                     # full O(N²)
    crp_adaptive = cfg.get("crp_adaptive", cfg.get("lambda_adaptive", False))
    crp_adaptive_target = cfg.get("crp_adaptive_target", cfg.get("adaptive_target", 0.15))
    rrd_loss_type = cfg.get("rrd_loss_type", "cosine")
    rrd_combined = cfg.get("rrd_combined", False)
    kd_temp = cfg.get("kd_temperature", 2.0)
    teacher_layers = cfg.get("teacher_layers", [6, 13, 20, 27])
    student_layers = cfg.get("student_layers", [8, 17, 26, 35])
    layer_map = dict(zip(teacher_layers, student_layers))

    experiment = cfg.get("experiment", "distill")
    val_every = args.val_every if args.val_every is not None else cfg.get("val_every", 0)
    val_batches = args.val_batches if args.val_batches is not None else cfg.get("val_batches", 50)

    output_dir = os.path.join(_BASE_DIR, "checkpoints_qwen25", experiment)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Teacher: {teacher_model_id} | Student: {student_model_id}")
    print(f"λ_kd={lambda_kd} λ_crp={lambda_crp} λ_rdist={lambda_rdist} crp_adaptive={crp_adaptive} crp_target={crp_adaptive_target} T={kd_temp}")
    print(f"Layer map: {layer_map}")

    # ============ Load teacher ============
    teacher_quant = cfg.get("teacher_quantize", False)
    teacher_load_kwargs = {"device_map": "auto"}
    if teacher_quant:
        print(f"\nLoading teacher with 4-bit quantization...")
        teacher_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        print(f"\nLoading teacher in {dtype_str}...")
        teacher_load_kwargs["torch_dtype"] = compute_dtype
    teacher = AutoModelForImageTextToText.from_pretrained(
        teacher_model_id, **teacher_load_kwargs
    )
    if teacher_lora:
        print(f"  Loading teacher LoRA: {teacher_lora}")
        teacher = PeftModel.from_pretrained(teacher, teacher_lora)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    t_cfg = getattr(teacher.config, "text_config", teacher.config)
    teacher_attn_cfg = {
        "num_attention_heads": t_cfg.num_attention_heads,
        "num_key_value_heads": t_cfg.num_key_value_heads,
        "hidden_size": t_cfg.hidden_size,
    }
    print(f"  Teacher: {t_cfg.num_attention_heads} heads, {t_cfg.hidden_size} hidden")

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  Teacher GPU: {gpu_mem:.1f} GB")

    # ============ Load student ============
    print("\nLoading student (3B)...")
    student = AutoModelForImageTextToText.from_pretrained(
        student_model_id, torch_dtype=compute_dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(student_model_id)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels

    student.enable_input_require_grads()
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(_BASE_DIR, args.resume)
        student = PeftModel.from_pretrained(student, resume_path, is_trainable=True)
        print(f"Resumed LoRA from {resume_path}")
    else:
        student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    if cfg.get("gradient_checkpointing", False):
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Student gradient checkpointing enabled")

    s_cfg = getattr(get_base_model(student).config, "text_config", get_base_model(student).config)
    student_attn_cfg = {
        "num_attention_heads": s_cfg.num_attention_heads,
        "num_key_value_heads": s_cfg.num_key_value_heads,
        "hidden_size": s_cfg.hidden_size,
    }

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  Total GPU after both models: {gpu_mem:.1f} GB")

    # ============ Register hooks ============
    teacher_store, student_store = {}, {}
    t_hooks = register_qk_hooks(teacher, teacher_layers, teacher_store, detach=True)
    s_hooks = register_qk_hooks(student, student_layers, student_store, detach=False)

    # ============ Load CRP patch labels (for region-aware distillation) ============
    crp_patch_labels = None
    crp_path = cfg.get("crp_importance_path")
    if crp_path:
        # Derive patch_labels path from importance path
        base_dir = crp_path if os.path.isabs(crp_path) else os.path.join(_BASE_DIR, crp_path)
        labels_path = os.path.join(os.path.dirname(base_dir), "crp_patch_labels.pt")
        if os.path.exists(labels_path):
            crp_patch_labels = torch.load(labels_path, weights_only=True)
            print(f"Loaded CRP patch labels for {len(crp_patch_labels)} images")

    # ============ Data ============
    data_dir = os.path.join(_BASE_DIR, "data_processed")
    default_train = "train_mini.json" if args.mini else "train.json"
    train_file = os.path.join(data_dir, cfg.get("train_file", default_train))
    val_file = os.path.join(data_dir, "val.json")

    train_dataset = DriveLMDataset(train_file, processor, max_length=max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    val_loader = None
    if val_every > 0 and os.path.exists(val_file):
        val_dataset = DriveLMDataset(val_file, processor, max_length=max_length)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
        )
        print(f"Validation: every {val_every} opt steps, {val_batches} batches, {len(val_dataset)} samples")

    num_batches = len(train_loader)
    total_steps = num_batches * num_epochs // grad_accum
    print(f"\nTraining: {len(train_dataset)} samples, {total_steps} opt steps")

    # ============ Optimizer ============
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps
    )

    # Resume optimizer/scheduler state
    resume_step = 0
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
            print(f"Resumed optimizer/scheduler from step {resume_step}")
        else:
            # Try to infer step from checkpoint name
            ckpt_name = os.path.basename(args.resume.rstrip("/"))
            if ckpt_name.startswith("checkpoint-"):
                resume_step = int(ckpt_name.split("-")[1])
            print(f"[WARN] No training_state.pt, resuming LoRA only from step {resume_step}")

    if args.wandb:
        import wandb
        wandb.init(project="drivelm-distill", name=experiment, config=cfg)

    # ============ Training ============
    print(f"\n{'='*60}")
    print(f"  Distillation | {experiment} | {num_epochs} epoch(s)")
    print(f"{'='*60}\n")

    student.train()
    global_step = resume_step
    # With shuffle=True, skipping exact batches is not reproducible anyway.
    # Instead, compute remaining batches and train from the start of a fresh shuffle.
    batches_done = resume_step * grad_accum if resume_step > 0 else 0
    remaining_batches = num_batches - batches_done
    if resume_step > 0:
        print(f"Resuming from step {resume_step}, skipping first {batches_done} batches (fresh shuffle, no slow iteration)")

    for epoch in range(num_epochs):
        epoch_losses = {"ce": 0, "kd": 0, "crp": 0, "rdist": 0, "total": 0}
        epoch_count = 0

        effective_total = remaining_batches if (epoch == 0 and resume_step > 0) else num_batches
        pbar = tqdm(enumerate(train_loader), total=effective_total,
                     desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)

        for step, batch in pbar:
            # Stop after remaining batches for resumed epoch
            if epoch == 0 and resume_step > 0 and step >= remaining_batches:
                break
            device = next(student.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            image_names_step = batch.pop("image_names", [])

            try:
                # Teacher forward (no grad)
                with torch.no_grad():
                    t_out = teacher(**batch)

                # Student forward
                s_out = student(**batch)

                # L_ce: student autoregressive loss
                L_ce = s_out.loss

                # L_kd: output KL divergence
                L_kd = kl_div_loss(s_out.logits, t_out.logits, kd_temp, batch.get("labels"))

                # L_rrd: region relation distillation
                # Build patch_labels_batch from precomputed labels
                image_names = image_names_step
                if crp_patch_labels is not None and image_names:
                    patch_labels_batch = [
                        crp_patch_labels.get(n).to(device) if crp_patch_labels.get(n) is not None else None
                        for n in image_names
                    ]
                else:
                    patch_labels_batch = [None] * batch["input_ids"].shape[0]

                L_crp, L_rdist_full = region_relation_distill_loss(
                    teacher_store, student_store,
                    layer_map, batch["input_ids"], image_token_id,
                    teacher_attn_cfg, student_attn_cfg,
                    patch_labels_batch,
                    rrd_loss_type=rrd_loss_type,
                    combined=rrd_combined,
                )

                _crp_val = L_crp.item() if torch.is_tensor(L_crp) else L_crp
                if crp_adaptive and _crp_val > 0:
                    effective_crp = crp_adaptive_target * (lambda_kd * L_kd.item()) / _crp_val
                else:
                    effective_crp = lambda_crp

                loss = (L_ce + lambda_kd * L_kd + effective_crp * L_crp + lambda_rdist * L_rdist_full) / grad_accum
                batch_total_raw = loss.item() * grad_accum

                # NaN guard: skip bad batches before they poison the model
                if not math.isfinite(batch_total_raw):
                    tqdm.write(f"[NaN] batch {step+1}, loss={batch_total_raw}, skipping")
                    del t_out, s_out, L_ce, L_kd, L_crp, L_rdist_full, loss
                    teacher_store.clear()
                    student_store.clear()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                loss.backward()

                batch_ce = L_ce.item()
                batch_kd = lambda_kd * L_kd.item()
                batch_crp = effective_crp * (L_crp.item() if torch.is_tensor(L_crp) else L_crp)
                batch_rdist = lambda_rdist * (L_rdist_full.item() if torch.is_tensor(L_rdist_full) else L_rdist_full)
                batch_total = batch_ce + batch_kd + batch_crp + batch_rdist
                epoch_losses["ce"] += batch_ce
                epoch_losses["kd"] += batch_kd
                epoch_losses["crp"] += batch_crp
                epoch_losses["rdist"] += batch_rdist
                epoch_losses["total"] += batch_total
                epoch_count += 1

                # Drop refs so activations / forward graph / hook stores get
                # freed immediately. NOTE: do NOT call empty_cache() on the
                # normal path — it returns cached blocks to the driver and
                # forces re-allocation next step, defeating the caching
                # allocator and costing ~50% throughput. empty_cache() is only
                # worth it in OOM/NaN recovery paths.
                # teacher_store / student_store MUST be cleared so hook-captured
                # QK tensors from this step don't accumulate across steps.
                del t_out, s_out, L_ce, L_kd, L_crp, L_rdist_full, loss
                teacher_store.clear()
                student_store.clear()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    tqdm.write(f"[OOM] batch {step+1}, skipping")
                    try:
                        del t_out, s_out, L_ce, L_kd, L_crp, L_rdist_full, loss
                    except NameError:
                        pass
                    teacher_store.clear()
                    student_store.clear()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            avg = {k: v / epoch_count for k, v in epoch_losses.items()}
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            pbar.set_postfix_str(
                f"ce={batch_ce:.3f}/{avg['ce']:.3f} "
                f"kd={batch_kd:.3f}/{avg['kd']:.3f} "
                f"crp={batch_crp:.3f}/{avg['crp']:.3f} "
                f"rdist={batch_rdist:.3f}/{avg['rdist']:.3f} "
                f"total={batch_total:.3f}/{avg['total']:.3f} "
                f"GPU={gpu_mem:.1f}GB"
            )

            if (step + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                # Skip optimizer step if gradients are NaN/Inf
                if not math.isfinite(grad_norm.item()):
                    tqdm.write(f"[NaN grad] step {global_step}, grad_norm={grad_norm.item()}, skipping update")
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.wandb:
                    import wandb
                    wandb.log({
                        "loss_ce": avg["ce"], "loss_kd": avg["kd"],
                        "loss_crp": avg["crp"], "loss_rdist": avg["rdist"],
                        "loss_total": avg["total"],
                        "lr": scheduler.get_last_lr()[0], "gpu_mem": gpu_mem,
                    }, step=global_step)

                if global_step % save_every == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    student.save_pretrained(save_path)
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }, os.path.join(save_path, "training_state.pt"))
                    tqdm.write(f"  [SAVE] checkpoint-{global_step}")

                if val_every > 0 and val_loader is not None and global_step % val_every == 0:
                    val_loss, val_acc = validate_distill(student, val_loader, val_batches, device)
                    tqdm.write(f"  [VAL] step={global_step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
                    if args.wandb:
                        import wandb
                        wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)

        pbar.close()
        avg = {k: v / max(epoch_count, 1) for k, v in epoch_losses.items()}
        print(f"\n  Epoch {epoch+1} | ce={avg['ce']:.4f} kd={avg['kd']:.4f} "
              f"crp={avg['crp']:.4f} rdist={avg['rdist']:.4f} total={avg['total']:.4f}\n")

    # Save final
    final_path = os.path.join(output_dir, "final")
    student.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Training complete! Final model: {final_path}")

    # Cleanup hooks
    for h in t_hooks + s_hooks:
        h.remove()

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
