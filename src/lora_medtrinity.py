"""
LoRA Fine-tuning on MedTrinity (Instruction Format)
=====================================================
Trains LoRA adapters on the LLM + full projector update, using the
diffusion-pretrained encoder (epoch 3). Uses instruction-style formatting
so the model learns to give structured medical answers, not verbose captions.

This is still zero-shot on VQA-RAD — we only train on MedTrinity data.

Trainable:
  - LoRA adapters on LLM (q_proj, v_proj, k_proj, o_proj) 
  - Full projector MLP (~21M params)
Frozen:
  - Vision encoder (diffusion-pretrained, epoch 3)

Usage:
    python lora_medtrinity.py
    python lora_medtrinity.py --max_samples 20000 --epochs 1
    python lora_medtrinity.py --encoder_ckpt none  # baseline LLaVA + LoRA (for comparison)
"""

import os
import re
import sys
import time
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from datasets import load_from_disk
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


# =========================================================================
# Args
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning on MedTrinity")
    # Model
    p.add_argument("--model_path", type=str, default="../models/llava-1.5-7b-hf")
    p.add_argument("--encoder_ckpt", type=str,
                    default="../checkpoints/diffusion_pretrain/checkpoint_epoch_3.pt",
                    help="Encoder checkpoint. Use 'none' for baseline LLaVA")
    # Data
    p.add_argument("--dataset_path", type=str,
                    default="../data/medtrinity-demo/hf_dataset")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_caption_words", type=int, default=90,
                    help="Truncate captions to this many words")
    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8,
                    help="Effective batch = batch_size * grad_accum = 32")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_seq_len", type=int, default=256,
                    help="Max tokens for caption (shorter = faster)")
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Output
    p.add_argument("--output_dir", type=str, default="../checkpoints/lora_medtrinity")
    p.add_argument("--tag", type=str, default=None,
                    help="Tag for output dir (auto-set based on encoder)")
    # Misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =========================================================================
# Logging
# =========================================================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "train.log")),
        ],
    )
    return logging.getLogger(__name__)


# =========================================================================
# Encoder loading (same as before)
# =========================================================================
def extract_and_load_encoder(model, ckpt_path, logger):
    """Load encoder from diffusion checkpoint into model's vision tower."""
    logger.info(f"Loading encoder from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract encoder keys (strip denoising head + timestep embed)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        loss = ckpt.get("avg_loss", "N/A")
    elif "vision_encoder_state_dict" in ckpt:
        state_dict = ckpt["vision_encoder_state_dict"]
        loss = ckpt.get("final_loss", "N/A")
    else:
        state_dict = ckpt
        loss = "N/A"

    encoder_weights = {
        k: v for k, v in state_dict.items()
        if not k.startswith("denoising_head.") and not k.startswith("timestep_embed.")
    }

    vision_tower = model.model.vision_tower.vision_model
    vt_state = vision_tower.state_dict()

    loaded = 0
    for key, val in encoder_weights.items():
        if key in vt_state and vt_state[key].shape == val.shape:
            vt_state[key] = val
            loaded += 1

    vision_tower.load_state_dict(vt_state)
    logger.info(f"  Loaded {loaded} encoder keys (train loss: {loss})")
    return model


# =========================================================================
# Dataset with instruction format
# =========================================================================
class MedTrinityInstructDataset(torch.utils.data.Dataset):
    """
    Formats MedTrinity samples as instruction-following examples:
    
    USER: <image>
    Describe this medical image briefly.
    ASSISTANT: {truncated caption}
    
    Captions are truncated to max_caption_words to encourage concise outputs.
    """

    PROMPTS = [
        "Describe this medical image briefly.",
        "What do you see in this medical image?",
        "Provide a brief description of this medical scan.",
        "What does this medical image show?",
        "Briefly describe the findings in this image.",
    ]

    def __init__(self, hf_dataset, processor, max_seq_len=256, max_caption_words=50):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.max_caption_words = max_caption_words
        self.num_image_tokens = 576

    def __len__(self):
        return len(self.dataset)

    def _truncate_caption(self, caption):
        """Truncate to max_caption_words."""
        words = caption.split()
        if len(words) > self.max_caption_words:
            words = words[:self.max_caption_words]
        return ' '.join(words)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        caption = sample["caption"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Truncate caption by word count (50 words ≈ 65-70 tokens)
        caption = self._truncate_caption(caption)

        # Pick a prompt variant (deterministic based on idx)
        prompt_template = self.PROMPTS[idx % len(self.PROMPTS)]

        # Format as instruction — processor will expand <image> to 576 tokens
        # Total seq: 576 (image) + ~10 (prompt) + ~70 (caption) ≈ 656 tokens
        text = f"USER: <image>\n{prompt_template}\nASSISTANT: {caption}"

        # Process image + text together
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=False,
        )

        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class CollateFn:
    """Collator that masks prompt tokens. Uses tokenizer to find ASSISTANT: boundary."""

    def __init__(self, tokenizer):
        # Tokenize "ASSISTANT:" to get the exact token IDs
        self.assistant_ids = tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False
        )
        self.IMAGE_TOKEN_ID = 32000

    def _find_subsequence(self, seq, subseq):
        """Find start index of subseq in seq, or -1."""
        sublen = len(subseq)
        for i in range(len(seq) - sublen + 1):
            if seq[i:i + sublen].tolist() == subseq:
                return i
        return -1

    def __call__(self, batch):
        import torch.nn.functional as F

        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        max_len = max(b["input_ids"].size(0) for b in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for b in batch:
            ids = b["input_ids"]
            mask = b["attention_mask"]
            pad_len = max_len - ids.size(0)

            ids_pad = F.pad(ids, (0, pad_len), value=0)
            mask_pad = F.pad(mask, (0, pad_len), value=0)

            labels = ids_pad.clone()
            labels[mask_pad == 0] = -100

            # Find "ASSISTANT:" in token sequence
            ast_start = self._find_subsequence(ids_pad, self.assistant_ids)
            if ast_start >= 0:
                # Mask everything up to and including "ASSISTANT:"
                response_start = ast_start + len(self.assistant_ids)
                labels[:response_start] = -100
            else:
                # Fallback: mask image tokens + some overhead
                img_positions = (ids_pad == self.IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(img_positions) > 0:
                    labels[:img_positions[-1].item() + 15] = -100
                else:
                    labels[:2] = -100

            input_ids_list.append(ids_pad)
            attention_mask_list.append(mask_pad)
            labels_list.append(labels)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }

# =========================================================================
# Main training
# =========================================================================
def train(args):
    # Auto-set tag
    if args.tag is None:
        if args.encoder_ckpt == "none":
            args.tag = "baseline_lora"
        else:
            epoch = args.encoder_ckpt.split("epoch_")[-1].split(".")[0]
            args.tag = f"enc{epoch}_lora"

    output_dir = os.path.join(args.output_dir, args.tag)
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info(f"  LoRA FINE-TUNING — {args.tag}")
    logger.info("=" * 60)
    logger.info(f"Args: {vars(args)}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------
    logger.info(f"Loading model from {args.model_path}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Swap encoder if specified
    if args.encoder_ckpt != "none":
        model = extract_and_load_encoder(model, args.encoder_ckpt, logger)
    else:
        logger.info("Using original LLaVA encoder (baseline)")

    model = model.to(device)

    # -----------------------------------------------------------------
    # Freeze vision encoder, set up LoRA on LLM
    # -----------------------------------------------------------------
    # Freeze vision encoder entirely
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False

    # Unfreeze projector (full fine-tune, not LoRA)
    for param in model.model.multi_modal_projector.parameters():
        param.requires_grad = True

    # Apply LoRA to LLM attention layers
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA — this wraps the language model layers
    model = get_peft_model(model, lora_config)

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params:     {total / 1e6:.1f}M")
    logger.info(f"Trainable params: {trainable / 1e6:.1f}M")
    logger.info(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    logger.info(f"  Target modules: q_proj, k_proj, v_proj, o_proj")
    logger.info(f"  Projector: fully trainable")
    logger.info(f"  Vision encoder: frozen")

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    logger.info(f"Loading dataset from {args.dataset_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "right"

    raw_dataset = load_from_disk(args.dataset_path)
    if args.max_samples and args.max_samples < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(args.max_samples))
        logger.info(f"Subsetted to {len(raw_dataset)} samples")

    dataset = MedTrinityInstructDataset(
        raw_dataset, processor, args.max_seq_len, args.max_caption_words
    )
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CollateFn(processor.tokenizer),
        pin_memory=True,
        drop_last=True,
    )

    # -----------------------------------------------------------------
    # Optimizer & scheduler
    # -----------------------------------------------------------------
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info(f"Effective batch: {args.batch_size} x {args.grad_accum} = {effective_batch}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"LR: {args.lr}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STARTING LoRA TRAINING")
    logger.info("=" * 60)

    model.train()
    # Set vision tower to eval (frozen, no dropout)
    # After PEFT wrapping, access via base_model
    for name, module in model.named_modules():
        if "vision_tower" in name and hasattr(module, 'eval'):
            module.eval()
            break

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        batch_loss_accum = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader, 1):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum

            if torch.isnan(loss):
                logger.warning(f"NaN at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                batch_loss_accum = 0.0
                continue

            loss.backward()
            batch_loss_accum += loss.item()

            if batch_idx % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_steps += 1
                epoch_loss += batch_loss_accum

                if global_step % args.log_every == 0:
                    avg = epoch_loss / epoch_steps
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    throughput = (batch_idx * args.batch_size) / elapsed
                    logger.info(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Step {epoch_steps}/{steps_per_epoch} | "
                        f"Loss: {batch_loss_accum:.4f} (avg: {avg:.4f}) | "
                        f"LR: {lr_now:.2e} | "
                        f"{throughput:.1f} img/s"
                    )
                batch_loss_accum = 0.0

        epoch_time = time.time() - epoch_start
        epoch_avg = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        logger.info(
            f"Epoch {epoch} complete | Avg loss: {epoch_avg:.4f} | "
            f"Time: {epoch_time:.0f}s ({epoch_time/60:.1f}min)"
        )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)

    # Merge LoRA weights into base model for easy inference
    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save full merged model
    save_path = os.path.join(output_dir, "merged_model")
    logger.info(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    logger.info(f"Done! Model saved to {save_path}")
    logger.info(f"Evaluate with: python eval_aligned.py --model_path {save_path} --tag {args.tag}")


if __name__ == "__main__":
    args = parse_args()
    train(args)