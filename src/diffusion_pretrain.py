"""
Diffusion Pretraining of CLIP ViT-L/14 on Medical Images
==========================================================
Trains the LLaVA vision encoder with a denoising diffusion objective
on MedTrinity-25M images to adapt it for medical image understanding.

Key design choices:
  - Cosine noise schedule (Nichol & Dhariwal, ICML 2021)
  - v-prediction parameterization (Salimans & Ho, ICLR 2022)
  - Stratified timestep sampling (Kingma et al., NeurIPS 2021)
  - Differential learning rates: Conv2d at 0.1x, ViT at 1x

After pretraining:
  - Discard the denoising head and timestep embedding
  - Keep the adapted ViT weights (Conv2d + transformer + embeddings)
  - Plug back into LLaVA and realign the projection MLP

Usage:
  # Single GPU (A100 40GB)
  python diffusion_pretrain.py

  # Override defaults
  python diffusion_pretrain.py --epochs 20 --batch_size 16 --lr 1e-4

  # Resume from checkpoint
  python diffusion_pretrain.py --resume ../checkpoints/diffusion_pretrain/checkpoint_epoch_5.pt
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
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

from transformers import LlavaForConditionalGeneration, AutoProcessor
from datasets import load_from_disk
from PIL import Image
import numpy as np


# =========================================================================
# 1. ARGUMENT PARSING
# =========================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Diffusion pretraining of ViT encoder")

    # Paths
    parser.add_argument("--model_path", type=str, default="../models/llava-1.5-7b-hf")
    parser.add_argument("--dataset_path", type=str, default="../data/medtrinity-demo/hf_dataset")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/diffusion_pretrain")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Effective batch = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for ViT")
    parser.add_argument("--conv_lr_scale", type=float, default=0.1, help="LR multiplier for Conv2d patch embedding")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Diffusion
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--prediction_type", type=str, default="v_prediction",
                        choices=["v_prediction", "epsilon"], help="What the model predicts")

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save_every_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)

    return parser.parse_args()


# =========================================================================
# 2. COSINE NOISE SCHEDULE (Nichol & Dhariwal, ICML 2021)
# =========================================================================

class CosineNoiseSchedule:
    """
    Cosine noise schedule from "Improved DDPM".
    
    Produces alpha_bar values that follow a cosine curve,
    ensuring noise is distributed more evenly across timesteps
    compared to the linear schedule.
    
    alpha_bar(t) = f(t) / f(0)
    where f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
    and s = 0.008 is a small offset to prevent singularity at t=0.
    """

    def __init__(self, num_timesteps=1000, s=0.008, device="cuda"):
        self.T = num_timesteps
        self.device = device

        # Compute alpha_bar for each timestep
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alpha_bar = f_t / f_t[0]

        # Clip to prevent numerical issues
        alpha_bar = torch.clamp(alpha_bar, min=1e-5, max=0.9999)

        # Compute beta from alpha_bar
        # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alpha_bar[:-1]])
        betas = 1.0 - (alpha_bar[1:] / alpha_bar_prev[1:])
        betas = torch.clamp(betas, min=1e-5, max=0.999)

        # Store everything as float32 on device
        self.alpha_bar = alpha_bar[1:].float().to(device)  # shape: [T]
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.betas = betas.float().to(device)

    def add_noise(self, x_0, noise, timesteps):
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: clean patch tokens [B, 576, 1024]
            noise: sampled Gaussian noise, same shape as x_0
            timesteps: [B] tensor of timestep indices (0 to T-1)
            
        Returns:
            x_t: noisy patch tokens [B, 576, 1024]
        """
        sqrt_ab = self.sqrt_alpha_bar[timesteps][:, None, None]        # [B, 1, 1]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[timesteps][:, None, None]  # [B, 1, 1]

        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
        return x_t

    def get_v_target(self, x_0, noise, timesteps):
        """
        v-prediction target: v = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x_0
        (Salimans & Ho, ICLR 2022)
        
        This parameterization gives more balanced gradients across timesteps.
        """
        sqrt_ab = self.sqrt_alpha_bar[timesteps][:, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[timesteps][:, None, None]

        v = sqrt_ab * noise - sqrt_one_minus_ab * x_0
        return v


# =========================================================================
# 3. STRATIFIED TIMESTEP SAMPLER (Kingma et al., NeurIPS 2021)
# =========================================================================

class StratifiedTimestepSampler:
    """
    Divides [0, T) into B equal strata, then samples one timestep
    from each stratum. Ensures each batch has even coverage of
    all noise levels instead of purely random sampling.
    
    For a batch of size B with T=1000:
      - Stratum 0: sample from [0, 1000/B)
      - Stratum 1: sample from [1000/B, 2*1000/B)
      - ...
      - Stratum B-1: sample from [(B-1)*1000/B, 1000)
    """

    def __init__(self, num_timesteps=1000):
        self.T = num_timesteps

    def sample(self, batch_size, device="cuda"):
        """Returns [B] tensor of stratified timestep indices."""
        strata_size = self.T / batch_size
        # One random offset per stratum
        offsets = torch.rand(batch_size, device=device)
        timesteps = (torch.arange(batch_size, device=device).float() + offsets) * strata_size
        timesteps = timesteps.long().clamp(0, self.T - 1)
        # Shuffle so strata order doesn't correlate with sample order
        perm = torch.randperm(batch_size, device=device)
        return timesteps[perm]


# =========================================================================
# 4. SINUSOIDAL TIMESTEP EMBEDDING
# =========================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Encodes timestep t into a 1024-dim vector using sinusoidal positional
    encoding (same idea as the original Transformer, but for diffusion timesteps).
    
    Followed by a 2-layer MLP to project into the token space.
    This embedding gets added to every patch token before the transformer.
    """

    def __init__(self, dim=1024, max_timesteps=1000):
        super().__init__()
        self.dim = dim

        # Precompute sinusoidal frequencies
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

        # MLP to project sinusoidal encoding to token space
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps):
        """
        Args:
            timesteps: [B] tensor of timestep indices
        Returns:
            [B, 1024] timestep embeddings
        """
        # Sinusoidal encoding
        t = timesteps.float()[:, None]  # [B, 1]
        args = t * self.freqs[None, :]  # [B, half_dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]

        # Project through MLP
        return self.mlp(embedding)


# =========================================================================
# 5. DENOISING HEAD
# =========================================================================

class DenoisingHead(nn.Module):
    """
    Lightweight MLP that takes ViT encoder output tokens and predicts
    the noise (or v-prediction target) per token.
    
    Applied identically to each of the 576 patch tokens.
    This module is DISCARDED after pretraining.
    """

    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        """
        Args:
            x: ViT output tokens [B, 576, 1024] (CLS already removed)
        Returns:
            predicted noise/v [B, 576, 1024]
        """
        return self.net(x)


# =========================================================================
# 6. DIFFUSION ViT WRAPPER
# =========================================================================

class DiffusionViT(nn.Module):
    """
    Wraps the CLIP ViT-L/14 vision encoder with diffusion components.
    
    Training flow:
      1. Image → Conv2d → clean patch tokens x_0 (576 x 1024)
      2. Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
      3. Add timestep embedding to noisy tokens
      4. Prepend CLS token + add position embeddings
      5. Forward through 24 transformer layers
      6. Drop CLS, pass through denoising head
      7. Predict noise (or v) → MSE loss
    """

    def __init__(self, vision_model, num_timesteps=1000):
        super().__init__()

        # Original CLIP ViT components (all trainable)
        self.embeddings = vision_model.embeddings
        self.encoder = vision_model.encoder
        self.pre_layrnorm = vision_model.pre_layrnorm

        hidden_dim = vision_model.config.hidden_size  # 1024

        # New components for diffusion (trained from scratch)
        self.timestep_embed = SinusoidalTimestepEmbedding(dim=hidden_dim, max_timesteps=num_timesteps)
        self.denoising_head = DenoisingHead(hidden_dim=hidden_dim)

    def get_patch_embeddings(self, pixel_values):
        """
        Extract patch embeddings from Conv2d WITHOUT adding CLS or position embeddings.
        This gives us the clean tokens x_0 that we add noise to.
        
        Args:
            pixel_values: [B, 3, 336, 336]
        Returns:
            patch_tokens: [B, 576, 1024]
        """
        # Conv2d: [B, 3, 336, 336] -> [B, 1024, 24, 24]
        patch_embeds = self.embeddings.patch_embedding(pixel_values)
        # Flatten spatial dims: [B, 1024, 24, 24] -> [B, 1024, 576] -> [B, 576, 1024]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        return patch_embeds

    def forward_encoder(self, noisy_tokens, timesteps):
        """
        Forward pass through the ViT encoder with timestep conditioning.
        
        Args:
            noisy_tokens: [B, 576, 1024] noisy patch tokens
            timesteps: [B] timestep indices
        Returns:
            predicted: [B, 576, 1024] predicted noise or v
        """
        batch_size = noisy_tokens.shape[0]
        target_dtype = noisy_tokens.dtype

        # 1. Add timestep embedding to every patch token
        t_emb = self.timestep_embed(timesteps)  # [B, 1024]
        t_emb = t_emb.unsqueeze(1)              # [B, 1, 1024]
        noisy_tokens = noisy_tokens + t_emb      # broadcasts to [B, 576, 1024]

        # 2. Prepend CLS token
        cls_tokens = self.embeddings.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        hidden_states = torch.cat([cls_tokens, noisy_tokens], dim=1)  # [B, 577, 1024]

        # 3. Add position embeddings
        position_ids = self.embeddings.position_ids[:, :hidden_states.shape[1]]
        hidden_states = hidden_states + self.embeddings.position_embedding(position_ids).to(target_dtype)

        # 4. Pre-layernorm (CLIP ViT has this)
        hidden_states = self.pre_layrnorm(hidden_states)

        # 5. Forward through all 24 transformer layers
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = encoder_outputs.last_hidden_state  # [B, 577, 1024]

        # 6. Drop CLS token (position 0)
        patch_output = hidden_states[:, 1:, :]  # [B, 576, 1024]

        # 7. Denoising head predicts noise/v
        predicted = self.denoising_head(patch_output)  # [B, 576, 1024]

        return predicted


# =========================================================================
# 7. DATASET
# =========================================================================

class MedTrinityDiffusionDataset(Dataset):
    """
    Wraps the HuggingFace MedTrinity dataset for diffusion training.
    Only loads images (captions not needed for this stage).
    Applies CLIP's image preprocessing (resize 512→336, normalize).
    """

    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # CLIP preprocessing: resize to 336x336, normalize
        inputs = self.processor(images=image, text="dummy", return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3, 336, 336]

        return pixel_values


# =========================================================================
# 8. LEARNING RATE SCHEDULER (cosine with warmup)
# =========================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """
    Linear warmup for warmup_steps, then cosine decay to 0.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================================
# 9. TRAINING LOOP
# =========================================================================

def train(args):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
        ],
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------
    logger.info(f"Loading LLaVA model from {args.model_path}...")
    full_model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,  # load in fp32, cast later for mixed precision
        low_cpu_mem_usage=True,
    )

    # Extract just the vision model
    vision_model = full_model.model.vision_tower.vision_model

    # Build diffusion wrapper
    diffusion_vit = DiffusionViT(vision_model, num_timesteps=args.num_timesteps).to(device)

    # Free the rest of the model (language model, projector) from memory
    del full_model
    torch.cuda.empty_cache()

    # Count parameters
    total_params = sum(p.numel() for p in diffusion_vit.parameters())
    trainable_params = sum(p.numel() for p in diffusion_vit.parameters() if p.requires_grad)
    new_params = (
        sum(p.numel() for p in diffusion_vit.timestep_embed.parameters())
        + sum(p.numel() for p in diffusion_vit.denoising_head.parameters())
    )
    logger.info(f"Total params:     {total_params / 1e6:.1f}M")
    logger.info(f"Trainable params: {trainable_params / 1e6:.1f}M")
    logger.info(f"New params:       {new_params / 1e6:.1f}M (timestep embed + denoising head)")

    # -----------------------------------------------------------------
    # Load processor and dataset
    # -----------------------------------------------------------------
    logger.info(f"Loading dataset from {args.dataset_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    hf_dataset = load_from_disk(args.dataset_path)
    train_dataset = MedTrinityDiffusionDataset(hf_dataset, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"Batch size: {args.batch_size} x {args.grad_accum_steps} accum = {args.batch_size * args.grad_accum_steps} effective")
    logger.info(f"Steps per epoch: {len(train_loader)}")

    # -----------------------------------------------------------------
    # Optimizer with differential learning rates
    # -----------------------------------------------------------------
    # Group 1: Conv2d patch embedding (lower LR)
    conv_params = list(diffusion_vit.embeddings.patch_embedding.parameters())

    # Group 2: Everything else in the ViT (normal LR)
    conv_param_ids = {id(p) for p in conv_params}
    vit_params = [
        p for p in diffusion_vit.parameters()
        if p.requires_grad and id(p) not in conv_param_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": conv_params, "lr": args.lr * args.conv_lr_scale, "name": "conv2d"},
            {"params": vit_params, "lr": args.lr, "name": "vit+head"},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    logger.info(f"Total optimization steps: {total_steps}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info(f"Conv2d LR: {args.lr * args.conv_lr_scale:.1e}")
    logger.info(f"ViT LR: {args.lr:.1e}")

    # -----------------------------------------------------------------
    # Diffusion components
    # -----------------------------------------------------------------
    noise_schedule = CosineNoiseSchedule(num_timesteps=args.num_timesteps, device=device)
    timestep_sampler = StratifiedTimestepSampler(num_timesteps=args.num_timesteps)

    # Mixed precision
    scaler = GradScaler(enabled=args.fp16)

    # -----------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------
    start_epoch = 0
    global_step = 0

    if args.resume:
        logger.info(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        diffusion_vit.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        # Fresh optimizer/scheduler with new LR (don't restore old states)
        logger.info(f"Resumed model weights at epoch {start_epoch}, global step {global_step}")
        logger.info(f"Using fresh optimizer with LR={args.lr}")

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STARTING DIFFUSION PRETRAINING")
    logger.info("=" * 60)
    logger.info(f"Schedule: cosine | Prediction: {args.prediction_type} | T={args.num_timesteps}")

    diffusion_vit.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for step, pixel_values in enumerate(train_loader):
            pixel_values = pixel_values.to(device)  # [B, 3, 336, 336]
            batch_size = pixel_values.shape[0]

            with autocast(device_type="cuda", enabled=args.fp16):
                # 1. Get clean patch embeddings from Conv2d
                x_0 = diffusion_vit.get_patch_embeddings(pixel_values)  # [B, 576, 1024]

                # 2. Sample noise
                noise = torch.randn_like(x_0)

                # 3. Sample stratified timesteps
                timesteps = timestep_sampler.sample(batch_size, device=device)

                # 4. Add noise to get x_t
                x_t = noise_schedule.add_noise(x_0, noise, timesteps)

                # 5. Predict noise/v from noisy tokens
                predicted = diffusion_vit.forward_encoder(x_t, timesteps)

                # 6. Compute loss
                if args.prediction_type == "v_prediction":
                    target = noise_schedule.get_v_target(x_0, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(predicted, target)
                loss = loss / args.grad_accum_steps  # normalize for accumulation

            # 7. Backward
            scaler.scale(loss).backward()

            # 8. Optimizer step (with gradient accumulation)
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion_vit.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum_steps
            epoch_steps += 1

            # Logging
            if (step + 1) % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                lr_conv = optimizer.param_groups[0]["lr"]
                lr_vit = optimizer.param_groups[1]["lr"]
                elapsed = time.time() - epoch_start
                imgs_per_sec = (step + 1) * batch_size / elapsed

                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: conv={lr_conv:.2e} vit={lr_vit:.2e} | "
                    f"{imgs_per_sec:.1f} img/s"
                )

        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_steps
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1} complete | "
            f"Avg loss: {avg_epoch_loss:.4f} | "
            f"Time: {epoch_time:.0f}s ({epoch_time/60:.1f}min)"
        )

        # Save checkpoint
        if (epoch + 1) % args.save_every_epoch == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": diffusion_vit.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "args": vars(args),
                    "avg_loss": avg_epoch_loss,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # -----------------------------------------------------------------
    # Save final encoder weights (without denoising head)
    # -----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SAVING FINAL ENCODER WEIGHTS")
    logger.info("=" * 60)

    # Extract only the ViT components we want to keep
    final_weights = {}
    for name, param in diffusion_vit.state_dict().items():
        # Skip denoising head and timestep embedding — we discard these
        if name.startswith("denoising_head.") or name.startswith("timestep_embed."):
            continue
        final_weights[name] = param

    final_path = os.path.join(args.output_dir, "vision_encoder_adapted.pt")
    torch.save(
        {
            "vision_encoder_state_dict": final_weights,
            "args": vars(args),
            "final_loss": avg_epoch_loss,
        },
        final_path,
    )
    logger.info(f"Saved adapted vision encoder: {final_path}")
    logger.info(f"Keys saved: {len(final_weights)}")
    logger.info("Done! Next step: realign the projection MLP using MedTrinity captions.")


# =========================================================================
# 10. ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    args = get_args()
    train(args)