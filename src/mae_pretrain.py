"""
MAE Pretraining of CLIP ViT-L/14 on Medical Images
====================================================
Mirrors diffusion_pretrain.py structure exactly.
Same hyperparameters: epochs=15, batch=64, lr=5e-5, conv_lr=5e-6, img=336

Usage:
  python mae_pretrain.py --num_samples -1 --epochs 15 --batch_size 64 \
      --lr 5e-5 --conv_lr_scale 0.1 --img_size 336 \
      --model_path /path/to/llava-1.5-7b-hf \
      --dataset_path /path/to/hf_dataset \
      --output_dir /path/to/output
"""

import os
import math
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import GradScaler, autocast

from transformers import LlavaForConditionalGeneration, AutoProcessor
from datasets import load_from_disk
import numpy as np


# =========================================================================
# 1. ARGS
# =========================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    type=str,   default="/ix/cs2770_2026s/abn80/cs2770_project/models/llava-1.5-7b-hf")
    p.add_argument("--dataset_path",  type=str,   default="/ix/cs2770_2026s/feg48/data/medtrinity-demo/hf_dataset")
    p.add_argument("--output_dir",    type=str,   default="/ix/cs2770_2026s/feg48/checkpoints/mae_pretrain_v2")
    p.add_argument("--resume",        type=str,   default=None)
    p.add_argument("--num_samples",   type=int,   default=-1,
                   help="Number of images to use. -1 = full dataset")
    p.add_argument("--epochs",        type=int,   default=15)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr",            type=float, default=5e-5,
                   help="LR for ViT transformer layers")
    p.add_argument("--conv_lr_scale", type=float, default=0.1,
                   help="LR multiplier for Conv2d patch embedding → 5e-6")
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--warmup_steps",  type=int,   default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mask_ratio",    type=float, default=0.75)
    p.add_argument("--img_size",      type=int,   default=336)
    p.add_argument("--decoder_dim",   type=int,   default=512)
    p.add_argument("--decoder_depth", type=int,   default=8)
    p.add_argument("--decoder_heads", type=int,   default=16)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--log_every",     type=int,   default=50)
    p.add_argument("--save_every_epoch", type=int, default=1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--fp16",          action="store_true", default=True)
    return p.parse_args()


# =========================================================================
# 2. DATASET
# =========================================================================

class MedTrinityMAEDataset(Dataset):
    """Image-only dataset for MAE pretraining."""
    def __init__(self, hf_dataset, processor):
        self.dataset   = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image  = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, text="dummy", return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)   # [3, 336, 336]


# =========================================================================
# 3. POSITIONAL EMBEDDING
# =========================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    h = w = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid   = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)
    grid   = grid.reshape(2, -1).T

    assert embed_dim % 4 == 0
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000 ** omega)
    emb_h = torch.einsum("n,d->nd", grid[:, 1], omega)
    emb_w = torch.einsum("n,d->nd", grid[:, 0], omega)
    emb   = torch.cat([emb_h.sin(), emb_h.cos(),
                       emb_w.sin(), emb_w.cos()], dim=-1)
    if cls_token:
        emb = torch.cat([torch.zeros(1, embed_dim), emb], dim=0)
    return emb


# =========================================================================
# 4. MAE DECODER  (discarded after pretraining)
# =========================================================================

class MAEDecoder(nn.Module):
    def __init__(self, encoder_dim=1024, decoder_dim=512,
                 num_patches=576, patch_size=14,
                 depth=8, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.decoder_embed   = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token      = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_emb = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.decoder_norm   = nn.LayerNorm(decoder_dim)
        self.decoder_pred   = nn.Linear(decoder_dim, patch_size * patch_size * 3, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        grid = int(math.sqrt(self.decoder_pos_emb.shape[1] - 1))
        pos  = get_2d_sincos_pos_embed(self.decoder_pos_emb.shape[-1],
                                       grid, cls_token=True)
        self.decoder_pos_emb.data.copy_(pos.unsqueeze(0))

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, L, D = x.shape
        num_patches = ids_restore.shape[1]
        num_mask    = num_patches - (L - 1)

        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        x_          = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_          = torch.gather(x_, 1,
                                   ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_emb
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x[:, 1:, :])
        return x


# =========================================================================
# 5. MAE ViT WRAPPER  (mirrors DiffusionViT structure)
# =========================================================================

class MAEViT(nn.Module):
    def __init__(self, vision_model, mask_ratio=0.75, patch_size=14, img_size=336):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.patch_size  = patch_size
        self.img_size    = img_size
        self.num_patches = (img_size // patch_size) ** 2   # 576

        # Original CLIP ViT components (all trainable)
        self.embeddings   = vision_model.embeddings
        self.encoder      = vision_model.encoder
        self.pre_layrnorm = vision_model.pre_layrnorm

        enc_dim = vision_model.config.hidden_size  # 1024

        # MAE decoder (discarded after pretraining)
        self.decoder = MAEDecoder(
            encoder_dim=enc_dim, decoder_dim=512,
            num_patches=self.num_patches, patch_size=patch_size,
            depth=8, num_heads=16,
        )

    def get_patch_embeddings(self, pixel_values):
        """Conv2d only — mirrors DiffusionViT.get_patch_embeddings()"""
        patch_embeds = self.embeddings.patch_embedding(pixel_values)  # [B,1024,24,24]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)        # [B,576,1024]
        return patch_embeds

    def random_masking(self, x):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))

        noise       = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :keep]
        x_vis    = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_vis, mask, ids_restore

    def patchify(self, imgs):
        p = self.patch_size
        h = w = self.img_size // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(imgs.shape[0], h * w, p * p * 3)
        return x

    def forward(self, pixel_values):
        target_dtype = next(self.embeddings.patch_embedding.parameters()).dtype
        B = pixel_values.shape[0]

        # 1. Patch embeddings
        patch_tokens = self.get_patch_embeddings(pixel_values)

        # 2. Random masking (75%)
        x_vis, mask, ids_restore = self.random_masking(patch_tokens)

        # 3. Prepend CLS + position embeddings on VISIBLE tokens only
        cls_tokens    = self.embeddings.class_embedding.expand(B, 1, -1).to(target_dtype)
        hidden_states = torch.cat([cls_tokens, x_vis], dim=1)

        num_vis       = hidden_states.shape[1]
        position_ids  = self.embeddings.position_ids[:, :num_vis]
        hidden_states = hidden_states + \
                        self.embeddings.position_embedding(position_ids).to(target_dtype)

        # 4. Pre-layernorm + transformer encoder
        hidden_states = self.pre_layrnorm(hidden_states)
        enc_out       = self.encoder(inputs_embeds=hidden_states,
                                     output_hidden_states=False, return_dict=True)
        hidden_states = enc_out.last_hidden_state

        # 5. MAE decoder
        pred = self.decoder(hidden_states, ids_restore)

        # 6. MSE loss on masked patches (per-patch normalised)
        target = self.patchify(pixel_values.float())
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        loss = ((pred.float() - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss, pred, mask


# =========================================================================
# 6. LR SCHEDULER  (identical to diffusion_pretrain.py)
# =========================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================================
# 7. TRAINING LOOP
# =========================================================================

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
        ],
    )
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading LLaVA from {args.model_path} ...")
    full_model   = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    vision_model = full_model.model.vision_tower.vision_model
    mae_vit      = MAEViT(vision_model, mask_ratio=args.mask_ratio,
                          patch_size=14, img_size=args.img_size).to(device)
    del full_model, vision_model
    torch.cuda.empty_cache()

    total_params   = sum(p.numel() for p in mae_vit.parameters())
    decoder_params = sum(p.numel() for p in mae_vit.decoder.parameters())
    logger.info(f"Total params:   {total_params/1e6:.1f}M")
    logger.info(f"Decoder params: {decoder_params/1e6:.1f}M  (discarded after training)")

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from {args.dataset_path} ...")
    processor  = AutoProcessor.from_pretrained(args.model_path)
    hf_dataset = load_from_disk(args.dataset_path)
    full_ds    = MedTrinityMAEDataset(hf_dataset, processor)

    if args.num_samples > 0 and args.num_samples < len(full_ds):
        rng     = np.random.default_rng(args.seed)
        indices = rng.choice(len(full_ds), size=args.num_samples, replace=False).tolist()
        dataset = Subset(full_ds, indices)
        logger.info(f"Using {args.num_samples:,} / {len(full_ds):,} images")
    else:
        dataset = full_ds
        logger.info(f"Using full dataset: {len(full_ds):,} images")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    eff_batch = args.batch_size * args.grad_accum_steps
    logger.info(f"Batch {args.batch_size} × {args.grad_accum_steps} accum = {eff_batch} effective")
    logger.info(f"Steps per epoch: {len(loader):,}")

    # ── Optimizer with differential LR (same as diffusion_pretrain.py) ───
    conv_params    = list(mae_vit.embeddings.patch_embedding.parameters())
    conv_param_ids = {id(p) for p in conv_params}
    other_params   = [p for p in mae_vit.parameters()
                      if p.requires_grad and id(p) not in conv_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": conv_params,  "lr": args.lr * args.conv_lr_scale, "name": "conv2d"},
            {"params": other_params, "lr": args.lr,                      "name": "vit+decoder"},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = len(loader) * args.epochs // args.grad_accum_steps
    scheduler   = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler      = GradScaler(enabled=args.fp16)

    logger.info(f"Total opt steps: {total_steps:,} | Warmup: {args.warmup_steps}")
    logger.info(f"Conv2d LR: {args.lr * args.conv_lr_scale:.1e} | ViT LR: {args.lr:.1e}")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        mae_vit.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        logger.info(f"Resumed at epoch {start_epoch}, step {global_step}")

    # ── Training ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STARTING MAE PRETRAINING")
    logger.info(f"Mask ratio: {args.mask_ratio} | img_size: {args.img_size}")
    logger.info("=" * 60)

    mae_vit.train()
    avg_epoch_loss = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_loss  = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for step, pixel_values in enumerate(loader):
            pixel_values = pixel_values.to(device)

            with autocast(device_type="cuda", enabled=args.fp16):
                loss, _, _ = mae_vit(pixel_values)
                loss       = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mae_vit.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss  += loss.item() * args.grad_accum_steps
            epoch_steps += 1

            if (step + 1) % args.log_every == 0:
                avg     = epoch_loss / epoch_steps
                lr_conv = optimizer.param_groups[0]["lr"]
                lr_vit  = optimizer.param_groups[1]["lr"]
                elapsed = time.time() - epoch_start
                ips     = (step + 1) * args.batch_size / elapsed
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step+1}/{len(loader)} | "
                    f"Loss: {loss.item()*args.grad_accum_steps:.4f} (avg: {avg:.4f}) | "
                    f"LR: conv={lr_conv:.2e} vit={lr_vit:.2e} | "
                    f"{ips:.1f} img/s"
                )

        avg_epoch_loss = epoch_loss / epoch_steps
        epoch_time     = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1} complete | "
            f"Avg loss: {avg_epoch_loss:.4f} | "
            f"Time: {epoch_time:.0f}s ({epoch_time/60:.1f}min)"
        )

        if (epoch + 1) % args.save_every_epoch == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch":            epoch,
                "global_step":      global_step,
                "model_state_dict": mae_vit.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "args":             vars(args),
                "avg_loss":         avg_epoch_loss,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # ── Save final encoder (discard decoder) ─────────────────────────────
    logger.info("=" * 60)
    logger.info("SAVING FINAL ENCODER WEIGHTS")
    logger.info("=" * 60)

    final_weights = {
        k: v for k, v in mae_vit.state_dict().items()
        if not k.startswith("decoder.")
    }
    final_path = os.path.join(args.output_dir, "vision_encoder_adapted.pt")
    torch.save({
        "vision_encoder_state_dict": final_weights,
        "args":       vars(args),
        "final_loss": avg_epoch_loss,
    }, final_path)
    logger.info(f"Saved: {final_path}  ({len(final_weights)} keys)")
    logger.info("Done! Next: checkpoint sweep + residual blending on VQA-RAD.")


if __name__ == "__main__":
    args = get_args()
    train(args)