"""
Residual Encoder Blending Evaluation
======================================
Blends original CLIP encoder features with diffusion-adapted encoder features:
    final_features = (1 - α) * CLIP_features + α * diffusion_features

Sweeps multiple α values and evaluates on VQA-RAD test split.
No training needed — this is pure inference with feature blending.

Usage:
    python eval_residual_blend.py --alphas 0.05 0.1 0.2 0.3 0.5
    python eval_residual_blend.py --alphas 0.1 --with_judge  # single alpha with LLM judge
"""

import os
import re
import sys
import copy
import time
import argparse
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="../models/llava-1.5-7b-hf")
    p.add_argument("--encoder_ckpt", type=str,
                    default="../checkpoints/diffusion_pretrain/checkpoint_epoch_3.pt")
    p.add_argument("--dataset", type=str, default="abhay2812/vqa-rad")
    p.add_argument("--cache_dir", type=str, default="../data/vqa-rad-cache")
    p.add_argument("--alphas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    p.add_argument("--with_judge", action="store_true")
    p.add_argument("--judge_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--judge_cache", type=str, default="../models/mistral-judge")
    p.add_argument("--output_dir", type=str, default="../data/residual_blend")
    p.add_argument("--test_only", action="store_true", default=True)
    return p.parse_args()


# =========================================================================
# Standard helpers
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


def judge_vqa(question, gt_answer, pred_answer, judge_model, judge_tokenizer):
    prompt = f"""You are a strict medical evaluator. Given a medical visual question, the ground truth answer, and a predicted answer, determine if the prediction is correct.

A prediction is CORRECT only if it:
- Is semantically equivalent to the ground truth
- Captures the SPECIFIC medical finding, not just a general category

A prediction is INCORRECT if it:
- Is too vague or generic compared to the ground truth
- Refers to a different structure, condition, or concept

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
    response = judge_tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip().upper()
    return 'CORRECT' if ('CORRECT' in response and 'INCORRECT' not in response) else 'INCORRECT'


# =========================================================================
# Extract encoder weights from diffusion checkpoint
# =========================================================================
def extract_encoder_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "vision_encoder_state_dict" in ckpt:
        state_dict = ckpt["vision_encoder_state_dict"]
    else:
        state_dict = ckpt

    return {
        k: v for k, v in state_dict.items()
        if not k.startswith("denoising_head.") and not k.startswith("timestep_embed.")
    }


def blend_encoder_weights(original_state, adapted_state, alpha):
    """Blend: (1 - alpha) * original + alpha * adapted"""
    blended = {}
    for key in original_state:
        if key in adapted_state and original_state[key].shape == adapted_state[key].shape:
            blended[key] = (1 - alpha) * original_state[key] + alpha * adapted_state[key]
        else:
            blended[key] = original_state[key]
    return blended


# =========================================================================
# Run inference
# =========================================================================
def run_eval(model, processor, ds, splits):
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
                'question': sample['question'],
                'question_type': sample['question_type_primary'],
                'answer_type': sample['answer_type'],
                'image_organ': sample['image_organ'],
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
    return df


def compute_metrics(df):
    closed = df[df['answer_type'] == 'CLOSED']
    opened = df[df['answer_type'] == 'OPEN']
    return {
        'closed_em': (closed['gt_normalized'] == closed['pred_normalized']).mean() * 100,
        'open_em': (opened['gt_normalized'] == opened['pred_normalized']).mean() * 100,
        'overall_em': (df['gt_normalized'] == df['pred_normalized']).mean() * 100,
        'n_closed': len(closed),
        'n_open': len(opened),
        'n_total': len(df),
    }


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    splits = ["test"] if args.test_only else ["train", "test"]

    print("=" * 70)
    print("  RESIDUAL ENCODER BLENDING — Alpha Sweep")
    print("=" * 70)
    print(f"Alphas: {args.alphas}")
    print(f"Encoder: {args.encoder_ckpt}")
    print(f"Judge: {'Yes' if args.with_judge else 'No'}")

    # Load dataset
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load original model and save original encoder weights
    print("\nLoading original model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()

    original_encoder_state = {
        k: v.cpu().clone()
        for k, v in model.model.vision_tower.vision_model.state_dict().items()
    }

    # Load adapted encoder weights
    print(f"Loading adapted encoder from {args.encoder_ckpt}...")
    adapted_encoder_state = extract_encoder_weights(args.encoder_ckpt)
    # Convert to same dtype
    for k in adapted_encoder_state:
        adapted_encoder_state[k] = adapted_encoder_state[k].float()
    for k in original_encoder_state:
        original_encoder_state[k] = original_encoder_state[k].float()

    print(f"Original keys: {len(original_encoder_state)}")
    print(f"Adapted keys: {len(adapted_encoder_state)}")

    # Run baseline first (alpha = 0)
    print(f"\n{'='*70}")
    print(f"  α = 0.0 (Baseline — original CLIP)")
    print(f"{'='*70}")
    baseline_df = run_eval(model, processor, ds, splits)
    baseline_metrics = compute_metrics(baseline_df)
    baseline_df.to_csv(os.path.join(args.output_dir, "alpha_0.0_results.csv"), index=False)

    all_metrics = [{'alpha': 0.0, **baseline_metrics}]

    # Sweep alphas
    for alpha in args.alphas:
        print(f"\n{'='*70}")
        print(f"  α = {alpha}")
        print(f"{'='*70}")

        # Blend weights
        blended = blend_encoder_weights(original_encoder_state, adapted_encoder_state, alpha)

        # Load blended weights into model
        vt = model.model.vision_tower.vision_model
        vt_state = vt.state_dict()
        for k, v in blended.items():
            if k in vt_state:
                vt_state[k] = v.to(vt_state[k].dtype)
        vt.load_state_dict(vt_state)

        # Run eval
        df = run_eval(model, processor, ds, splits)
        metrics = compute_metrics(df)
        df.to_csv(os.path.join(args.output_dir, f"alpha_{alpha}_results.csv"), index=False)

        all_metrics.append({'alpha': alpha, **metrics})

    # Restore original weights for clean state
    vt = model.model.vision_tower.vision_model
    vt_state = vt.state_dict()
    for k, v in original_encoder_state.items():
        if k in vt_state:
            vt_state[k] = v.to(vt_state[k].dtype)
    vt.load_state_dict(vt_state)

    # LLM Judge (if enabled)
    if args.with_judge:
        del model
        torch.cuda.empty_cache()

        print(f"\nLoading judge...")
        judge_tokenizer = AutoTokenizer.from_pretrained(
            args.judge_model, cache_dir=args.judge_cache
        )
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model, torch_dtype=torch.float16, cache_dir=args.judge_cache
        ).to("cuda")
        judge_model.generation_config.pad_token_id = judge_tokenizer.eos_token_id
        judge_model.eval()

        # Re-read all saved CSVs and run judge
        for m in all_metrics:
            alpha = m['alpha']
            csv_path = os.path.join(args.output_dir, f"alpha_{alpha}_results.csv")
            df = pd.read_csv(csv_path)

            open_df = df[df['answer_type'] == 'OPEN'].copy()
            print(f"\n  Judging α={alpha} ({len(open_df)} open-ended)...")

            verdicts = []
            for _, row in tqdm(open_df.iterrows(), total=len(open_df)):
                v = judge_vqa(row['question'], row['gt_normalized'],
                             row['pred_normalized'], judge_model, judge_tokenizer)
                verdicts.append(v)

            open_df['llm_judge'] = verdicts
            open_df['llm_judge_correct'] = open_df['llm_judge'] == 'CORRECT'

            df = df.merge(
                open_df[['qid', 'split', 'llm_judge', 'llm_judge_correct']],
                on=['qid', 'split'], how='left'
            )
            df.loc[df['answer_type'] == 'CLOSED', 'llm_judge_correct'] = \
                df.loc[df['answer_type'] == 'CLOSED', 'gt_normalized'] == \
                df.loc[df['answer_type'] == 'CLOSED', 'pred_normalized']

            df.to_csv(csv_path, index=False)

            m['open_judge'] = open_df['llm_judge_correct'].mean() * 100
            m['overall_judge'] = df['llm_judge_correct'].mean() * 100

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — Residual Blending Sweep")
    print(f"{'='*70}")

    if args.with_judge:
        header = f"{'Alpha':>7} {'Cl EM':>7} {'Op EM':>7} {'Op Jdg':>8} {'All EM':>8} {'All Jdg':>9}"
        print(f"\n  {header}")
        print(f"  {'-'*len(header)}")
        for m in all_metrics:
            print(f"  {m['alpha']:>7.2f} {m['closed_em']:>6.1f}% {m['open_em']:>6.1f}% "
                  f"{m.get('open_judge', 0):>7.1f}% {m['overall_em']:>7.1f}% "
                  f"{m.get('overall_judge', 0):>8.1f}%")
    else:
        header = f"{'Alpha':>7} {'Closed EM':>10} {'Open EM':>9} {'Overall EM':>11}"
        print(f"\n  {header}")
        print(f"  {'-'*len(header)}")
        for m in all_metrics:
            print(f"  {m['alpha']:>7.2f} {m['closed_em']:>9.1f}% {m['open_em']:>8.1f}% {m['overall_em']:>10.1f}%")

    best = max(all_metrics, key=lambda m: m['overall_em'])
    print(f"\n  Best EM: α={best['alpha']} ({best['overall_em']:.1f}%)")
    if args.with_judge:
        best_j = max(all_metrics, key=lambda m: m.get('overall_judge', 0))
        print(f"  Best Judge: α={best_j['alpha']} ({best_j.get('overall_judge', 0):.1f}%)")

    # Save summary
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(args.output_dir, "blend_summary.csv"), index=False)
    print(f"\n  Summary saved to {args.output_dir}/blend_summary.csv")


if __name__ == "__main__":
    main()