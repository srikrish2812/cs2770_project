"""
Aligned Model Evaluation on VQA-RAD
=====================================
Mirrors the baseline_infer.ipynb pipeline but uses the diffusion-pretrained
+ projection-aligned model.

Pipeline:
  1. Load aligned model from checkpoints/projection_align/llava_medvlm_aligned
  2. Run inference on all VQA-RAD samples (train + test)
  3. Compute exact match metrics
  4. Load Mistral-7B-Instruct as LLM judge for open-ended questions
  5. Compute LLM judge metrics
  6. Print comprehensive breakdowns (split, question type, organ, etc.)
  7. Save results CSV for comparison with baseline

Usage:
    python evaluate_aligned_model.py
    
    # Use baseline model instead (for re-running baseline with same code)
    python evaluate_aligned_model.py --model_path ../models/llava-1.5-7b-hf --tag baseline
"""

import os
import re
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
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
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model on VQA-RAD")
    p.add_argument("--model_path", type=str,
                    default="../checkpoints/projection_align/llava_medvlm_aligned",
                    help="Path to LLaVA model (aligned or baseline)")
    p.add_argument("--dataset", type=str, default="abhay2812/vqa-rad",
                    help="HuggingFace dataset name")
    p.add_argument("--cache_dir", type=str, default="../data/vqa-rad-cache")
    p.add_argument("--judge_model", type=str,
                    default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--judge_cache", type=str, default="../models/mistral-judge")
    p.add_argument("--output_dir", type=str, default="../data")
    p.add_argument("--tag", type=str, default="aligned",
                    help="Tag for output filenames (e.g., 'aligned', 'baseline')")
    p.add_argument("--skip_judge", action="store_true",
                    help="Skip LLM judge evaluation (faster, exact match only)")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    return p.parse_args()


# =========================================================================
# Prompts (same as baseline)
# =========================================================================
def get_prompt(question, answer_type):
    if answer_type == 'CLOSED':
        return f"USER: <image>\n{question} Answer with only yes or no.\nASSISTANT:"
    else:
        return f"USER: <image>\n{question} Answer in a few words.\nASSISTANT:"


# =========================================================================
# Normalization (same as baseline)
# =========================================================================
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


# =========================================================================
# LLM Judge (same prompt as baseline)
# =========================================================================
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
# Print helpers
# =========================================================================
def print_metrics_em(df, label):
    closed = df[df['answer_type'] == 'CLOSED']
    opened = df[df['answer_type'] == 'OPEN']

    closed_acc = (closed['gt_normalized'] == closed['pred_normalized']).mean()
    open_acc = (opened['gt_normalized'] == opened['pred_normalized']).mean()
    overall_acc = (df['gt_normalized'] == df['pred_normalized']).mean()

    print(f"\n{'='*60}")
    print(f"  {label} (Exact Match)")
    print(f"{'='*60}")
    print(f"  Closed-ended: {closed_acc*100:5.1f}% (n={len(closed)})")
    print(f"  Open-ended:   {open_acc*100:5.1f}% (n={len(opened)})")
    print(f"  Overall:      {overall_acc*100:5.1f}% (n={len(df)})")


def print_final(df, label):
    closed = df[df['answer_type'] == 'CLOSED']
    opened = df[df['answer_type'] == 'OPEN']

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  CLOSED (n={len(closed)})")
    print(f"    Accuracy:        {closed['llm_judge_correct'].mean()*100:5.1f}%")
    print(f"  OPEN (n={len(opened)})")
    print(f"    Exact match:     {opened['exact_match'].mean()*100:5.1f}%")
    print(f"    LLM judge:       {opened['llm_judge_correct'].mean()*100:5.1f}%")
    print(f"  OVERALL (n={len(df)})")
    print(f"    Exact match:     {df['exact_match'].mean()*100:5.1f}%")
    print(f"    LLM judge:       {df['llm_judge_correct'].mean()*100:5.1f}%")


def print_breakdown(df, column, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for val in sorted(df[column].unique()):
        subset = df[df[column] == val]
        acc = (subset['gt_normalized'] == subset['pred_normalized']).mean()
        print(f"  {val:12s}: {acc*100:5.1f}% (n={len(subset)})")


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  VQA-RAD EVALUATION — {args.tag.upper()}")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Splits: {args.splits}")
    print(f"Output tag: {args.tag}")

    # -----------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------
    print(f"\nLoading model from {args.model_path}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()

    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"Device: {model.device}")

    # -----------------------------------------------------------------
    # 2. Load dataset
    # -----------------------------------------------------------------
    print(f"\nLoading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)

    # -----------------------------------------------------------------
    # 3. Run inference
    # -----------------------------------------------------------------
    print(f"\nRunning inference...")
    results = []

    for split in args.splits:
        for i in tqdm(range(len(ds[split])), desc=f"{split}"):
            sample = ds[split][i]
            img = sample['image'].convert("RGB")

            prompt = get_prompt(sample['question'], sample['answer_type'])
            inputs = processor(
                text=prompt, images=img, return_tensors="pt"
            ).to("cuda", torch.float16)

            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=50, do_sample=False
                )

            pred_raw = processor.decode(
                output[0], skip_special_tokens=True
            ).split("ASSISTANT:")[-1].strip()
            pred_normalized = pred_raw.strip().lower()

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
                'pred_normalized': pred_normalized,
            })

    results_df = pd.DataFrame(results)

    # Add exact match and contains match columns
    results_df['exact_match'] = results_df.apply(
        lambda r: normalize_answer(r['pred_normalized']) == normalize_answer(r['gt_normalized']),
        axis=1
    )
    results_df['contains'] = results_df.apply(
        lambda r: contains_match(r['pred_normalized'], r['gt_normalized']),
        axis=1
    )

    # Save intermediate results
    intermediate_path = os.path.join(
        args.output_dir, f"{args.tag}_vqa_rad_results_all.csv"
    )
    results_df.to_csv(intermediate_path, index=False)
    print(f"\nSaved {len(results_df)} predictions to {intermediate_path}")

    # -----------------------------------------------------------------
    # 4. Print exact match results
    # -----------------------------------------------------------------
    train_df = results_df[results_df['split'] == 'train']
    test_df = results_df[results_df['split'] == 'test']

    print_metrics_em(train_df, "TRAIN")
    print_metrics_em(test_df, "TEST")
    print_metrics_em(results_df, "ALL")

    # By question type
    print_breakdown(train_df, 'question_type', "TRAIN - By Question Type")
    print_breakdown(test_df, 'question_type', "TEST - By Question Type")
    print_breakdown(results_df, 'question_type', "ALL - By Question Type")

    # By organ
    print_breakdown(train_df, 'image_organ', "TRAIN - By Image Organ")
    print_breakdown(test_df, 'image_organ', "TEST - By Image Organ")
    print_breakdown(results_df, 'image_organ', "ALL - By Image Organ")

    # By organ x answer type
    print(f"\n{'='*60}")
    print(f"  ALL - By Organ x Answer Type")
    print(f"{'='*60}")
    for organ in sorted(results_df['image_organ'].unique()):
        for atype in ['CLOSED', 'OPEN']:
            subset = results_df[
                (results_df['image_organ'] == organ) &
                (results_df['answer_type'] == atype)
            ]
            if len(subset) > 0:
                acc = (subset['gt_normalized'] == subset['pred_normalized']).mean()
                print(f"  {organ:6s} {atype:6s}: {acc*100:5.1f}% (n={len(subset)})")

    # -----------------------------------------------------------------
    # 5. LLM Judge evaluation
    # -----------------------------------------------------------------
    if args.skip_judge:
        print("\n[Skipping LLM judge evaluation]")
        results_df.to_csv(
            os.path.join(args.output_dir, f"{args.tag}_vqa_rad_results_final.csv"),
            index=False
        )
        print("Done!")
        return

    # Free VLM memory before loading judge
    del model
    torch.cuda.empty_cache()
    print(f"\nFreed VLM memory. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\nLoading judge model: {args.judge_model}...")
    judge_tokenizer = AutoTokenizer.from_pretrained(
        args.judge_model, cache_dir=args.judge_cache
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.float16,
        cache_dir=args.judge_cache
    ).to("cuda")
    judge_model.generation_config.pad_token_id = judge_tokenizer.eos_token_id
    judge_model.eval()
    print(f"Judge loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Run judge on all open-ended predictions
    open_all = results_df[results_df['answer_type'] == 'OPEN'].copy()
    print(f"\nRunning judge on {len(open_all)} open-ended predictions...")

    judge_verdicts = []
    for i, row in tqdm(open_all.iterrows(), total=len(open_all)):
        verdict = judge_vqa(
            row['question'], row['gt_normalized'], row['pred_normalized'],
            judge_model, judge_tokenizer
        )
        judge_verdicts.append(verdict)

    open_all['llm_judge'] = judge_verdicts
    open_all['llm_judge_correct'] = open_all['llm_judge'] == 'CORRECT'

    # Merge back into results_df
    results_df = results_df.merge(
        open_all[['qid', 'split', 'llm_judge', 'llm_judge_correct']],
        on=['qid', 'split'],
        how='left'
    )

    # For closed-ended, judge = exact match
    results_df.loc[results_df['answer_type'] == 'CLOSED', 'llm_judge_correct'] = \
        results_df.loc[results_df['answer_type'] == 'CLOSED', 'gt_normalized'] == \
        results_df.loc[results_df['answer_type'] == 'CLOSED', 'pred_normalized']

    # Save final results
    final_path = os.path.join(
        args.output_dir, f"{args.tag}_vqa_rad_results_final.csv"
    )
    results_df.to_csv(final_path, index=False)
    print(f"\nSaved final results to {final_path}")

    # -----------------------------------------------------------------
    # 6. Print comprehensive results with judge
    # -----------------------------------------------------------------
    train_df = results_df[results_df['split'] == 'train']
    test_df = results_df[results_df['split'] == 'test']

    print_final(train_df, "TRAIN")
    print_final(test_df, "TEST")
    print_final(results_df, "ALL")

    # Open-ended by question type
    opened = results_df[results_df['answer_type'] == 'OPEN']
    print(f"\n{'='*60}")
    print(f"  OPEN-ENDED by Question Type (all)")
    print(f"{'='*60}")
    for qtype in sorted(opened['question_type'].unique()):
        subset = opened[opened['question_type'] == qtype]
        em = subset['exact_match'].mean() * 100
        jd = subset['llm_judge_correct'].mean() * 100
        print(f"  {qtype:12s}: EM={em:5.1f}%  Judge={jd:5.1f}%  (n={len(subset)})")

    # All by organ with judge
    print(f"\n{'='*60}")
    print(f"  ALL by Image Organ (LLM Judge)")
    print(f"{'='*60}")
    for organ in sorted(results_df['image_organ'].unique()):
        subset = results_df[results_df['image_organ'] == organ]
        acc = subset['llm_judge_correct'].mean() * 100
        print(f"  {organ:12s}: {acc:5.1f}% (n={len(subset)})")

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Results saved to: {final_path}")
    print(f"  Compare with baseline: ../data/llava_vqa_rad_results_final.csv")


if __name__ == "__main__":
    main()