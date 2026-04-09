"""
Projection MLP Realignment
===========================
After diffusion SSL pretraining, the ViT-L/14 encoder produces features in a
shifted distribution. This script realigns the 2-layer projection MLP
(1024→4096→4096) so the LLM (Vicuna-7B) can interpret the new visual features.

Trainable:  multi_modal_projector only (~21M params)
Frozen:     vision_tower (diffusion-adapted) + language_model (Vicuna-7B)
Objective:  next-token prediction (autoregressive LM loss) on image-caption pairs
Data:       MedTrinity demo subset (161,630 samples) — image + caption columns

Usage:
    python projection_align.py [--epochs 3] [--lr 1e-4] [--batch_size 8] [--grad_accum 4]

SLURM example:
    srun --partition=gpu --gres=gpu:1 --mem=64G --time=06:00:00 \
         python projection_align.py --epochs 3
"""

import os
import sys
import json
import math
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from PIL import Image

# ===========================================================================
# Config
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Projection MLP realignment")
    p.add_argument("--model_path", type=str, default="../models/llava-1.5-7b-hf")
    p.add_argument("--encoder_ckpt", type=str,
                    default="../checkpoints/diffusion_pretrain/vision_encoder_adapted.pt")
    p.add_argument("--dataset_path", type=str,
                    default="../data/medtrinity-demo/hf_dataset")
    p.add_argument("--output_dir", type=str,
                    default="../checkpoints/projection_align")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_seq_len", type=int, default=768)
    p.add_argument("--max_samples", type=int, default=None,
                    help="Use only first N samples (default: use all)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every_epoch", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ===========================================================================
# Logging
# ===========================================================================
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


# ===========================================================================
# Dataset & Collator
# ===========================================================================
class MedTrinityAlignDataset(torch.utils.data.Dataset):
    """
    Wraps MedTrinity HF dataset for projection alignment.
    Each sample: image + caption → processor produces pixel_values + input_ids.

    Prompt template (matches LLaVA-1.5 pretraining format):
        <image>\n{caption}
    
    The model learns to predict the caption tokens given the image.
    
    The processor expands <image> into 576 placeholder tokens in input_ids.
    We truncate only the caption to fit within max_seq_len (which must be
    >= 576 + caption tokens).
    """

    IMAGE_TOKEN_ID = 32000

    def __init__(self, hf_dataset, processor, max_seq_len=768):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_seq_len = max_seq_len
        # Number of image tokens the processor inserts (576 for ViT-L/14 @ 336px)
        self.num_image_tokens = 576

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        caption = sample["caption"]

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Truncate caption text BEFORE tokenization to leave room for
        # image tokens (576) + BOS + <image>\n overhead
        # We tokenize the caption alone first to truncate it
        max_caption_tokens = self.max_seq_len - self.num_image_tokens - 10  # safety margin
        caption_ids = self.processor.tokenizer(
            caption, add_special_tokens=False
        )["input_ids"]
        if len(caption_ids) > max_caption_tokens:
            # Decode the truncated tokens back to text
            caption = self.processor.tokenizer.decode(
                caption_ids[:max_caption_tokens],
                skip_special_tokens=True,
            )

        # Now process image + text together (processor handles expansion)
        prompt = f"<image>\n{caption}"
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=False,
        )

        pixel_values = inputs["pixel_values"].squeeze(0)       # [3, 336, 336]
        input_ids = inputs["input_ids"].squeeze(0)             # [seq_len]
        attention_mask = inputs["attention_mask"].squeeze(0)    # [seq_len]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def collate_fn(batch):
    """
    Pad input_ids and attention_mask to the max length in the batch.
    Labels = input_ids shifted (handled by the model), with padding set to -100.
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    # Find max seq len in batch
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids_padded = []
    attention_mask_padded = []
    labels_padded = []

    for b in batch:
        ids = b["input_ids"]
        mask = b["attention_mask"]
        pad_len = max_len - ids.size(0)

        # Pad on the right
        ids_pad = F.pad(ids, (0, pad_len), value=0)
        mask_pad = F.pad(mask, (0, pad_len), value=0)

        # Labels: same as input_ids but padding tokens → -100
        labels = ids_pad.clone()
        labels[mask_pad == 0] = -100

        # Mask the prompt prefix so loss is only on caption tokens.
        # The processor expands <image> into 576 image placeholder tokens (all ID 32000).
        # Sequence looks like: [BOS, 32000, 32000, ...(576 times)..., 32000, \n, caption...]
        # We mask everything up to and including the \n after the last image token.
        IMAGE_TOKEN_ID = 32000
        img_positions = (ids_pad == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(img_positions) > 0:
            # Last image token position + 1 for the \n that follows
            last_img_pos = img_positions[-1].item()
            mask_until = last_img_pos + 1  # +1 for the \n
            mask_until = min(mask_until, ids_pad.size(0) - 1)
            labels[:mask_until + 1] = -100
        else:
            # Fallback: if no image token found, mask first 2 tokens (BOS + \n)
            labels[:2] = -100

        input_ids_padded.append(ids_pad)
        attention_mask_padded.append(mask_pad)
        labels_padded.append(labels)

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_mask_padded),
        "labels": torch.stack(labels_padded),
    }


# ===========================================================================
# Load encoder weights into model
# ===========================================================================
def load_adapted_encoder(model, encoder_ckpt_path, logger):
    """
    Load diffusion-pretrained encoder weights into the model's vision tower.
    The checkpoint contains only vision encoder keys (389 keys).
    """
    logger.info(f"Loading adapted encoder from {encoder_ckpt_path}")
    ckpt = torch.load(encoder_ckpt_path, map_location="cpu", weights_only=True)

    # The diffusion pretraining script saves as:
    #   {"vision_encoder_state_dict": {...}, "args": ..., "final_loss": ...}
    # Handle both nested and flat formats
    if "vision_encoder_state_dict" in ckpt:
        encoder_weights = ckpt["vision_encoder_state_dict"]
        logger.info(f"  Loaded from nested dict (final_loss={ckpt.get('final_loss', 'N/A')})")
    elif isinstance(ckpt, dict) and any("encoder.layers" in k for k in ckpt.keys()):
        encoder_weights = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format. Keys: {list(ckpt.keys())[:5]}")

    # The vision tower in LLaVA is at model.model.vision_tower.vision_model
    # (LlavaForConditionalGeneration → LlavaModel → vision_tower → CLIPVisionModel)
    vision_tower = model.model.vision_tower.vision_model
    vt_state = vision_tower.state_dict()

    # Match keys — checkpoint keys are like "embeddings.patch_embedding.weight",
    # "encoder.layers.0.self_attn.q_proj.weight" etc.
    loaded, skipped = 0, 0
    for key, val in encoder_weights.items():
        if key in vt_state:
            if vt_state[key].shape == val.shape:
                vt_state[key] = val
                loaded += 1
            else:
                logger.warning(f"  Shape mismatch for {key}: "
                             f"model={vt_state[key].shape}, ckpt={val.shape}")
                skipped += 1
        else:
            logger.warning(f"  Key not found in model: {key}")
            skipped += 1

    vision_tower.load_state_dict(vt_state)
    logger.info(f"  Loaded {loaded} keys, skipped {skipped}")
    return model


# ===========================================================================
# Freeze / unfreeze
# ===========================================================================
def setup_trainable_params(model, logger):
    """
    Freeze everything except multi_modal_projector.
    """
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze projector
    projector_params = 0
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True
            projector_params += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total params:     {total_params / 1e6:.1f}M")
    logger.info(f"Trainable params: {trainable_params / 1e6:.1f}M (projector only)")
    logger.info(f"Frozen params:    {(total_params - trainable_params) / 1e6:.1f}M")

    return model


# ===========================================================================
# Training loop
# ===========================================================================
def train(args):
    logger = setup_logging(args.output_dir)
    logger.info("=" * 60)
    logger.info("PROJECTION MLP REALIGNMENT")
    logger.info("=" * 60)
    logger.info(f"Args: {vars(args)}")

    # Seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    logger.info(f"Loading LLaVA model from {args.model_path}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Load adapted encoder weights
    model = load_adapted_encoder(model, args.encoder_ckpt, logger)

    # Freeze everything except projector
    model = setup_trainable_params(model, logger)

    # Enable gradient checkpointing to reduce VRAM
    # This recomputes activations during backward instead of storing them
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled (saves VRAM, ~30% slower)")

    # Move to GPU
    model = model.to(device)

    # Projector stays in float32 for training stability
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    logger.info("Model loaded and configured.")

    # -----------------------------------------------------------------------
    # Load processor & dataset
    # -----------------------------------------------------------------------
    logger.info(f"Loading processor from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    # Set padding side to right for causal LM
    processor.tokenizer.padding_side = "right"

    logger.info(f"Loading dataset from {args.dataset_path}...")
    raw_dataset = load_from_disk(args.dataset_path)
    if args.max_samples and args.max_samples < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(args.max_samples))
        logger.info(f"Subsetted to {len(raw_dataset)} samples")
    dataset = MedTrinityAlignDataset(raw_dataset, processor, args.max_seq_len)
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # -----------------------------------------------------------------------
    # Optimizer & scheduler
    # -----------------------------------------------------------------------
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info(f"Batch size: {args.batch_size} x {args.grad_accum} accum = {effective_batch} effective")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {args.lr}")

    # Only optimize projector params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,  # LLaVA pretraining uses no weight decay for projector
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STARTING PROJECTION ALIGNMENT")
    logger.info("=" * 60)

    model.train()
    # But keep frozen modules in eval mode to disable dropout etc.
    model.model.vision_tower.eval()
    model.model.language_model.eval()

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_steps = 0
        batch_loss_accum = 0.0
        epoch_start = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader, 1):
            # Move to device
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass — use the model's native forward which handles
            # image token expansion, embedding merging, etc. correctly.
            # Pass labels=None so it doesn't compute loss internally in fp16.
            # Use bfloat16 — same memory as fp16 but fp32 dynamic range,
            # prevents NaN from softmax overflow over 32K vocab
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum

            # Check for NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                batch_loss_accum = 0.0
                continue

            # Backward
            loss.backward()
            batch_loss_accum += loss.item()

            # Count non-masked tokens
            n_tokens = (labels != -100).sum().item()
            epoch_tokens += n_tokens

            # Optimizer step
            if batch_idx % args.grad_accum == 0:
                # Gradient clipping
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

                # Log
                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    throughput = (batch_idx * args.batch_size) / elapsed

                    logger.info(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Step {epoch_steps}/{steps_per_epoch} | "
                        f"Loss: {batch_loss_accum:.4f} (avg: {avg_loss:.4f}) | "
                        f"LR: {lr_now:.2e} | "
                        f"{throughput:.1f} img/s"
                    )

                batch_loss_accum = 0.0

        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg = epoch_loss / epoch_steps if epoch_steps > 0 else 0

        logger.info(
            f"Epoch {epoch} complete | "
            f"Avg loss: {epoch_avg:.4f} | "
            f"Time: {epoch_time:.0f}s ({epoch_time/60:.1f}min) | "
            f"Tokens: {epoch_tokens:,}"
        )

        # Save checkpoint
        if args.save_every_epoch:
            ckpt_path = os.path.join(args.output_dir, f"projector_epoch_{epoch}.pt")
            # Save only projector weights
            projector_state = {
                k: v.cpu()
                for k, v in model.state_dict().items()
                if "multi_modal_projector" in k
            }
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "projector_state_dict": projector_state,
                    "loss": epoch_avg,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # Track best
        if epoch_avg < best_loss:
            best_loss = epoch_avg
            best_path = os.path.join(args.output_dir, "projector_best.pt")
            projector_state = {
                k: v.cpu()
                for k, v in model.state_dict().items()
                if "multi_modal_projector" in k
            }
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "projector_state_dict": projector_state,
                    "loss": best_loss,
                },
                best_path,
            )
            logger.info(f"New best loss: {best_loss:.4f} → saved {best_path}")

    # -----------------------------------------------------------------------
    # Save final full model for inference
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SAVING FINAL ALIGNED MODEL")
    logger.info("=" * 60)

    final_dir = os.path.join(args.output_dir, "llava_medvlm_aligned")
    logger.info(f"Saving full model to {final_dir}...")

    # Convert projector back to float16 for saving
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.data = param.data.to(torch.bfloat16)

    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    logger.info("Done! Full aligned model saved.")
    logger.info(f"Next step: evaluate on VQA-RAD using {final_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)