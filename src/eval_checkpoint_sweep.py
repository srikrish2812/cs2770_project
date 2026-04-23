"""
Checkpoint Sweep Evaluation
==============================
Extracts vision encoder weights from multiple diffusion pretraining
checkpoints (which contain the full DiffusionViT including denoising head)
and evaluates each on VQA-RAD with the original LLaVA projector + LLM.

This helps find the optimal pretraining epoch — early checkpoints may
generalize better than the final (potentially overfit) checkpoint.

Usage:
    # Quick exact-match sweep across epochs 3,5,8,10,15
    python eval_checkpoint_sweep.py --skip_judge

    # Full eval with LLM judge on specific epochs
    python eval_checkpoint_sweep.py --epochs 3 5 8

    # Custom checkpoint dir
    python eval_checkpoint_sweep.py --ckpt_dir ../checkpoints/diffusion_pretrain --epochs 3 5 8 10 12 15
"""

import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)


# =========================================================================
# Args
# =========================================================================
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Sweep diffusion checkpoints on VQA-RAD")
    p.add_argument("--model_path", type=str, default="../models/llava-1.5-7b-hf")
    p.add_argument("--ckpt_dir", type=str,
                    default="../checkpoints/diffusion_pretrain",
                    help="Directory containing checkpoint_epoch_N.pt files")
    p.add_argument("--epochs", type=int, nargs="+", default=[3, 5, 8, 10, 15],
                    help="Which epoch checkpoints to evaluate")
    p.add_argument("--dataset", type=str, default="abhay2812/vqa-rad")
    p.add_argument("--cache_dir", type=str, default="../data/vqa-rad-cache")
    p.add_argument("--judge_model", type=str,
                    default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--judge_cache", type=str, default="../models/mistral-judge")
    p.add_argument("--output_dir", type=str, default="../data/checkpoint_sweep")
    p.add_argument("--skip_judge", action="store_true",
                    help="Skip LLM judge (exact match only, much faster)")
    p.add_argument("--test_only", action="store_true",
                    help="Evaluate only on test split (faster)")
    return p.parse_args()


# =========================================================================
# Helpers (same as baseline)
# =========================================================================
def get_prompt(question, answer_type):
    if answer_type == 'CLOSED':
        return f"USER: <image>\n{question} Answer with only yes or no.\nASSISTANT:"
    else:
        return f"USER: <image>\n{question} Answer in a few words.\nASSISTANT:"


def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = ' '.join(s.split())
    return s


def contains_match(pred, gt):
    pred_n = normalize_answer(pred)
    gt_n = normalize_answer(gt)
    return gt_n in pred_n or pred_n in gt_n


def judge_vqa(question, gt_answer, pred_answer, judge_model, judge_tokenizer):
    prompt = f"""You are a strict medical evaluator. Given a medical visual question, the ground truth answer, and a predicted answer, determine if the prediction is correct.

A prediction is CORRECT only if it:
- Is semantically equivalent to the ground truth
- Captures the SPECIFIC medical finding, not just a general category
- Example: GT "axial" vs Pred "axial plane" → CORRECT
- Example: GT "liver" vs Pred "hepatic" → CORRECT
- Example: GT "ct scan" vs Pred "ct" → CORRECT

A prediction is INCORRECT if it:
- Is too vague or generic compared to the ground truth (e.g., GT "pulmonary nodules" vs Pred "lung" → INCORRECT)
- Refers to a different structure, condition, or concept
- Describes a general category instead of the specific finding (e.g., GT "ring-enhancing" vs Pred "tumor" → INCORRECT)
- Example: GT "elliptical" vs Pred "round" → INCORRECT
- Example: GT "ct scan" vs Pred "x-ray" → INCORRECT
- Example: GT "shrunken and nodular" vs Pred "large" → INCORRECT

Be STRICT. When in doubt, mark INCORRECT.

Question: {question}
Ground truth answer: {gt_answer}
Predicted answer: {pred_answer}

Respond with ONLY one word: CORRECT or INCORRECT"""

    messages = [{"role": "user", "content": prompt}]
    inputs = judge_tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    ).to("cuda")
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=5, do_sample=False)

    response = judge_tokenizer.decode(
        output[0][input_len:], skip_special_tokens=True
    ).strip().upper()

    if 'CORRECT' in response and 'INCORRECT' not in response:
        return 'CORRECT'
    else:
        return 'INCORRECT'


# =========================================================================
# Extract encoder from diffusion checkpoint
# =========================================================================
def extract_encoder_from_checkpoint(ckpt_path):
    """
    Diffusion checkpoints save the full DiffusionViT state_dict which includes:
      - embeddings.* (encoder)
      - encoder.* (encoder)
      - pre_layrnorm.* (encoder)
      - timestep_embed.* (discard)
      - denoising_head.* (discard)
    
    We extract only the encoder keys, matching what vision_encoder_adapted.pt contains.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        avg_loss = ckpt.get("avg_loss", "N/A")
    elif "vision_encoder_state_dict" in ckpt:
        # Already extracted (like vision_encoder_adapted.pt)
        return ckpt["vision_encoder_state_dict"], ckpt.get("final_loss", "N/A")
    else:
        state_dict = ckpt
        avg_loss = "N/A"

    # Filter out denoising head and timestep embedding
    encoder_weights = {}
    for key, val in state_dict.items():
        if key.startswith("denoising_head.") or key.startswith("timestep_embed."):
            continue
        encoder_weights[key] = val

    return encoder_weights, avg_loss


def swap_encoder(model, encoder_weights):
    """Swap encoder weights into model's vision tower."""
    vision_tower = model.model.vision_tower.vision_model
    vt_state = vision_tower.state_dict()

    loaded = 0
    for key, val in encoder_weights.items():
        if key in vt_state and vt_state[key].shape == val.shape:
            vt_state[key] = val
            loaded += 1

    vision_tower.load_state_dict(vt_state)
    return model, loaded


# =========================================================================
# Run inference
# =========================================================================
def run_inference(model, processor, ds, splits):
    """Run VQA-RAD inference and return results DataFrame."""
    results = []
    for split in splits:
        for i in tqdm(range(len(ds[split])), desc=f"  {split}"):
            sample = ds[split][i]
            img = sample['image'].convert("RGB")
            prompt = get_prompt(sample['question'], sample['answer_type'])
            inputs = processor(
                text=prompt, images=img, return_tensors="pt"
            ).to("cuda", torch.float16)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            pred_raw = processor.decode(
                output[0], skip_special_tokens=True
            ).split("ASSISTANT:")[-1].strip()

            results.append({
                'qid': sample['qid'],
                'image_name': sample['image_name'],
                'image_organ': sample['image_organ'],
                'question': sample['question'],
                'question_type': sample['question_type_primary'],
                'answer_type': sample['answer_type'],
                'phrase_type': sample['phrase_type'],
                'split': split,
                'gt_answer': sample['answer'],
                'gt_normalized': sample['answer_normalized'],
                'pred_raw': pred_raw,
                'pred_normalized': pred_raw.strip().lower(),
            })

    df = pd.DataFrame(results)
    df['exact_match'] = df.apply(
        lambda r: normalize_answer(r['pred_normalized']) == normalize_answer(r['gt_normalized']),
        axis=1
    )
    df['contains'] = df.apply(
        lambda r: contains_match(r['pred_normalized'], r['gt_normalized']),
        axis=1
    )
    return df


def compute_metrics(df, label):
    """Compute and return metrics dict for a results DataFrame."""
    closed = df[df['answer_type'] == 'CLOSED']
    opened = df[df['answer_type'] == 'OPEN']

    metrics = {
        'label': label,
        'n_total': len(df),
        'n_closed': len(closed),
        'n_open': len(opened),
        'closed_em': (closed['gt_normalized'] == closed['pred_normalized']).mean() * 100,
        'open_em': (opened['gt_normalized'] == opened['pred_normalized']).mean() * 100,
        'overall_em': (df['gt_normalized'] == df['pred_normalized']).mean() * 100,
    }

    if 'llm_judge_correct' in df.columns:
        metrics['open_judge'] = opened['llm_judge_correct'].mean() * 100
        metrics['overall_judge'] = df['llm_judge_correct'].mean() * 100
        metrics['closed_judge'] = closed['llm_judge_correct'].mean() * 100

    return metrics


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    splits = ["test"] if args.test_only else ["train", "test"]

    print("=" * 70)
    print("  CHECKPOINT SWEEP — Diffusion Pretraining Epochs vs VQA-RAD")
    print("=" * 70)
    print(f"Base model:   {args.model_path}")
    print(f"Checkpoints:  {args.ckpt_dir}")
    print(f"Epochs:       {args.epochs}")
    print(f"Splits:       {splits}")
    print(f"LLM Judge:    {'Yes' if not args.skip_judge else 'No (exact match only)'}")

    # Load dataset
    print(f"\nLoading dataset...")
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)

    # Load processor (shared across all runs)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Store original model weights for resetting between runs
    print(f"Loading base model (will reset encoder between runs)...")
    base_state = torch.load(
        os.path.join(args.model_path, "model-00001-of-00004.safetensors"),
        map_location="cpu",
    ) if False else None  # we'll just reload the model each time for simplicity

    # ---------------------------------------------------------------
    # Run baseline first
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  BASELINE (original LLaVA-1.5-7b, no encoder swap)")
    print(f"{'='*70}")

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()

    baseline_df = run_inference(model, processor, ds, splits)
    baseline_df.to_csv(os.path.join(args.output_dir, "baseline_results.csv"), index=False)
    baseline_metrics = compute_metrics(baseline_df, "Baseline")

    del model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Run each epoch checkpoint
    # ---------------------------------------------------------------
    all_metrics = [baseline_metrics]
    all_results = {"baseline": baseline_df}

    for epoch in args.epochs:
        ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
        if not os.path.exists(ckpt_path):
            print(f"\n  [WARNING] {ckpt_path} not found, skipping epoch {epoch}")
            continue

        label = f"Epoch {epoch}"
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        # Extract encoder weights
        encoder_weights, train_loss = extract_encoder_from_checkpoint(ckpt_path)
        print(f"  Extracted {len(encoder_weights)} encoder keys (train loss: {train_loss})")

        # Load fresh model and swap encoder
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to("cuda")
        model, n_loaded = swap_encoder(model, encoder_weights)
        model.eval()
        print(f"  Loaded {n_loaded} encoder keys into model")

        # Run inference
        epoch_df = run_inference(model, processor, ds, splits)
        epoch_df.to_csv(
            os.path.join(args.output_dir, f"epoch_{epoch}_results.csv"), index=False
        )
        epoch_metrics = compute_metrics(epoch_df, label)
        epoch_metrics['train_loss'] = train_loss

        all_metrics.append(epoch_metrics)
        all_results[f"epoch_{epoch}"] = epoch_df

        del model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # LLM Judge (if enabled)
    # ---------------------------------------------------------------
    if not args.skip_judge:
        print(f"\n{'='*70}")
        print(f"  Loading LLM Judge for all configurations...")
        print(f"{'='*70}")

        judge_tokenizer = AutoTokenizer.from_pretrained(
            args.judge_model, cache_dir=args.judge_cache
        )
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model, torch_dtype=torch.float16, cache_dir=args.judge_cache
        ).to("cuda")
        judge_model.generation_config.pad_token_id = judge_tokenizer.eos_token_id
        judge_model.eval()

        for key, df in all_results.items():
            label = key.replace("_", " ").title()
            open_all = df[df['answer_type'] == 'OPEN'].copy()
            print(f"\n  Judging {label} ({len(open_all)} open-ended)...")

            verdicts = []
            for i, row in tqdm(open_all.iterrows(), total=len(open_all), desc=f"  {label}"):
                verdict = judge_vqa(
                    row['question'], row['gt_normalized'], row['pred_normalized'],
                    judge_model, judge_tokenizer
                )
                verdicts.append(verdict)

            open_all['llm_judge'] = verdicts
            open_all['llm_judge_correct'] = open_all['llm_judge'] == 'CORRECT'

            df = df.merge(
                open_all[['qid', 'split', 'llm_judge', 'llm_judge_correct']],
                on=['qid', 'split'], how='left'
            )
            df.loc[df['answer_type'] == 'CLOSED', 'llm_judge_correct'] = \
                df.loc[df['answer_type'] == 'CLOSED', 'gt_normalized'] == \
                df.loc[df['answer_type'] == 'CLOSED', 'pred_normalized']

            # Save updated results
            df.to_csv(os.path.join(args.output_dir, f"{key}_results_final.csv"), index=False)
            all_results[key] = df

            # Update metrics
            for m in all_metrics:
                if m['label'] == key.replace("_", " ").title() or \
                   (key == "baseline" and m['label'] == "Baseline") or \
                   (key.startswith("epoch_") and m['label'] == f"Epoch {key.split('_')[1]}"):
                    m.update(compute_metrics(df, m['label']))

        del judge_model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — Checkpoint Sweep Results")
    print(f"{'='*70}")

    # Determine which split we're reporting
    split_label = "TEST" if args.test_only else "ALL"

    if args.skip_judge:
        # Exact match only table
        header = f"{'Config':<16} {'Train Loss':>11} {'Closed EM':>10} {'Open EM':>9} {'Overall EM':>11}"
        print(f"\n  {header}")
        print(f"  {'-'*len(header)}")
        for m in all_metrics:
            loss = f"{m.get('train_loss', '-'):>11}" if isinstance(m.get('train_loss'), float) else f"{'—':>11}"
            print(f"  {m['label']:<16} {loss} {m['closed_em']:>9.1f}% {m['open_em']:>8.1f}% {m['overall_em']:>10.1f}%")
    else:
        # Full table with judge
        header = f"{'Config':<16} {'Loss':>7} {'Cl EM':>6} {'Op EM':>6} {'Op Jdg':>7} {'All EM':>7} {'All Jdg':>8}"
        print(f"\n  {header}")
        print(f"  {'-'*len(header)}")
        for m in all_metrics:
            loss = f"{m.get('train_loss', 0):>7.4f}" if isinstance(m.get('train_loss'), float) else f"{'—':>7}"
            print(
                f"  {m['label']:<16} {loss} "
                f"{m['closed_em']:>5.1f}% "
                f"{m['open_em']:>5.1f}% "
                f"{m.get('open_judge', 0):>6.1f}% "
                f"{m['overall_em']:>6.1f}% "
                f"{m.get('overall_judge', 0):>7.1f}%"
            )

    # Highlight best
    best_em = max(all_metrics, key=lambda m: m['overall_em'])
    print(f"\n  Best overall EM: {best_em['label']} ({best_em['overall_em']:.1f}%)")

    if not args.skip_judge:
        best_judge = max(all_metrics, key=lambda m: m.get('overall_judge', 0))
        print(f"  Best overall Judge: {best_judge['label']} ({best_judge.get('overall_judge', 0):.1f}%)")

    # Save summary
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(args.output_dir, "sweep_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to {summary_path}")
    print(f"  Individual results in {args.output_dir}/")


if __name__ == "__main__":
    main()