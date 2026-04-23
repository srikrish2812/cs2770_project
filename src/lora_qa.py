"""
LoRA Fine-tuning on MedTrinity QA Pairs
==========================================
Trains LoRA on LLM + full projector using instruction-format QA pairs
generated from MedTrinity captions. The QA format matches VQA-RAD
(short answers to specific medical questions).

Still zero-shot on VQA-RAD — trained only on MedTrinity-derived QA.

Usage:
    # Diffusion encoder + LoRA on QA data
    python lora_qa.py --encoder_ckpt ../checkpoints/diffusion_pretrain/checkpoint_epoch_3.pt

    # Baseline + LoRA on QA data (control)
    python lora_qa.py --encoder_ckpt none
"""

import os
import sys
import time
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


# =========================================================================
# Args
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning on MedTrinity QA")
    p.add_argument("--model_path", type=str, default="../models/llava-1.5-7b-hf")
    p.add_argument("--encoder_ckpt", type=str,
                    default="../checkpoints/diffusion_pretrain/checkpoint_epoch_3.pt",
                    help="Encoder checkpoint. 'none' for baseline")
    p.add_argument("--qa_dataset", type=str, default="../data/medtrinity-qa/hf_dataset")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Limit QA pairs (default: use all)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--output_dir", type=str, default="../checkpoints/lora_qa")
    p.add_argument("--tag", type=str, default=None)
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
# Encoder loading
# =========================================================================
def load_encoder(model, ckpt_path, logger):
    logger.info(f"Loading encoder from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

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

    vt = model.model.vision_tower.vision_model
    vt_state = vt.state_dict()
    loaded = 0
    for k, v in encoder_weights.items():
        if k in vt_state and vt_state[k].shape == v.shape:
            vt_state[k] = v
            loaded += 1
    vt.load_state_dict(vt_state)
    logger.info(f"  Loaded {loaded} encoder keys (train loss: {loss})")
    return model


# =========================================================================
# Dataset — QA format matching VQA-RAD
# =========================================================================
class MedTrinityQADataset(torch.utils.data.Dataset):
    """
    Uses generated QA pairs in VQA-RAD format:
        USER: <image>\n{question} Answer in a few words.\nASSISTANT: {answer}

    Closed-style questions (yes/no answers) use:
        USER: <image>\n{question} Answer with only yes or no.\nASSISTANT: {answer}
    """

    YES_NO_ANSWERS = {'yes', 'no'}

    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Use closed/open prompt based on answer type
        if answer.strip().lower() in self.YES_NO_ANSWERS:
            prompt_suffix = "Answer with only yes or no."
        else:
            prompt_suffix = "Answer in a few words."

        text = f"USER: <image>\n{question} {prompt_suffix}\nASSISTANT: {answer}"

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=False,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


class QACollateFn:
    """Pads batch and masks prompt — only trains on the answer tokens."""

    def __init__(self, tokenizer):
        self.assistant_ids = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        self.IMAGE_TOKEN_ID = 32000

    def _find_subseq(self, seq, subseq):
        sublen = len(subseq)
        for i in range(len(seq) - sublen + 1):
            if seq[i:i + sublen].tolist() == subseq:
                return i
        return -1

    def __call__(self, batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        max_len = max(b["input_ids"].size(0) for b in batch)

        ids_list, mask_list, labels_list = [], [], []

        for b in batch:
            ids = b["input_ids"]
            mask = b["attention_mask"]
            pad_len = max_len - ids.size(0)

            ids_pad = F.pad(ids, (0, pad_len), value=0)
            mask_pad = F.pad(mask, (0, pad_len), value=0)
            labels = ids_pad.clone()
            labels[mask_pad == 0] = -100

            # Mask everything before and including "ASSISTANT:"
            ast_pos = self._find_subseq(ids_pad, self.assistant_ids)
            if ast_pos >= 0:
                labels[:ast_pos + len(self.assistant_ids)] = -100
            else:
                # Fallback: mask image tokens + overhead
                img_pos = (ids_pad == self.IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(img_pos) > 0:
                    labels[:img_pos[-1].item() + 15] = -100

            ids_list.append(ids_pad)
            mask_list.append(mask_pad)
            labels_list.append(labels)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(ids_list),
            "attention_mask": torch.stack(mask_list),
            "labels": torch.stack(labels_list),
        }


# =========================================================================
# Training
# =========================================================================
def train(args):
    if args.tag is None:
        if args.encoder_ckpt == "none":
            args.tag = "baseline_qa_lora"
        else:
            epoch = args.encoder_ckpt.split("epoch_")[-1].split(".")[0]
            args.tag = f"enc{epoch}_qa_lora"

    output_dir = os.path.join(args.output_dir, args.tag)
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info(f"  LoRA QA FINE-TUNING — {args.tag}")
    logger.info("=" * 60)
    logger.info(f"Args: {vars(args)}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if args.encoder_ckpt != "none":
        model = load_encoder(model, args.encoder_ckpt, logger)
    else:
        logger.info("Using original CLIP encoder (baseline)")

    model = model.to(device)

    # Freeze encoder, unfreeze projector
    for p in model.model.vision_tower.parameters():
        p.requires_grad = False
    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad = True

    # Apply LoRA to LLM
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params:     {total / 1e6:.1f}M")
    logger.info(f"Trainable params: {trainable / 1e6:.1f}M")
    logger.info(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, targets=q/k/v/o_proj")
    logger.info(f"  Projector: fully trainable")
    logger.info(f"  Encoder: frozen")

    # Dataset
    logger.info(f"Loading QA dataset from {args.qa_dataset}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "right"

    qa_dataset = load_from_disk(args.qa_dataset)
    if args.max_samples and args.max_samples < len(qa_dataset):
        qa_dataset = qa_dataset.shuffle(seed=42).select(range(args.max_samples))
        logger.info(f"Subsetted to {len(qa_dataset)} QA pairs")

    dataset = MedTrinityQADataset(qa_dataset, processor)
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=QACollateFn(processor.tokenizer),
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info(f"Effective batch: {effective_batch}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup: {warmup_steps}")
    logger.info(f"LR: {args.lr}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
    )

    # Train
    logger.info("=" * 60)
    logger.info("STARTING LoRA QA TRAINING")
    logger.info("=" * 60)

    model.train()
    for name, module in model.named_modules():
        if "vision_tower" in name and hasattr(module, 'eval'):
            module.eval()
            break

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        batch_accum = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader, 1):
            pv = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(pixel_values=pv, input_ids=ids,
                           attention_mask=mask, labels=labels)
                loss = out.loss / args.grad_accum

            if torch.isnan(loss):
                logger.warning(f"NaN at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                batch_accum = 0.0
                continue

            loss.backward()
            batch_accum += loss.item()

            if batch_idx % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_steps += 1
                epoch_loss += batch_accum

                if global_step % args.log_every == 0:
                    avg = epoch_loss / epoch_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    tput = (batch_idx * args.batch_size) / elapsed
                    logger.info(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Step {epoch_steps}/{steps_per_epoch} | "
                        f"Loss: {batch_accum:.4f} (avg: {avg:.4f}) | "
                        f"LR: {lr:.2e} | {tput:.1f} img/s"
                    )
                batch_accum = 0.0

        elapsed = time.time() - t0
        avg = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        logger.info(
            f"Epoch {epoch} complete | Avg loss: {avg:.4f} | "
            f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)"
        )

    # Save
    logger.info("=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)

    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    save_path = os.path.join(output_dir, "merged_model")
    logger.info(f"Saving to {save_path}...")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    logger.info(f"Done! Evaluate with:")
    logger.info(f"  python eval_aligned.py --model_path {save_path} --tag {args.tag}")


if __name__ == "__main__":
    args = parse_args()
    train(args)