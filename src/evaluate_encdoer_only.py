"""
Encoder-Only Swap Evaluation on VQA-RAD
=========================================
Loads the original LLaVA-1.5-7b with its original projector,
swaps in ONLY the diffusion-pretrained vision encoder, and evaluates.

This tests whether the encoder adaptation helps even without
projector realignment.

Usage:
    python eval_encoder_only.py
    python eval_encoder_only.py --skip_judge   # exact match only (fast)
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
# Config
# =========================================================================
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="../models/llava-1.5-7b-hf")
    p.add_argument("--encoder_ckpt", type=str,
                    default="../checkpoints/diffusion_pretrain/vision_encoder_adapted.pt")
    p.add_argument("--dataset", type=str, default="abhay2812/vqa-rad")
    p.add_argument("--cache_dir", type=str, default="../data/vqa-rad-cache")
    p.add_argument("--judge_model", type=str,
                    default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--judge_cache", type=str, default="../models/mistral-judge")
    p.add_argument("--output_dir", type=str, default="../data")
    p.add_argument("--tag", type=str, default="encoder_only")
    p.add_argument("--skip_judge", action="store_true")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
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


def print_metrics_em(df, label):
    closed = df[df['answer_type'] == 'CLOSED']
    opened = df[df['answer_type'] == 'OPEN']
    print(f"\n{'='*60}")
    print(f"  {label} (Exact Match)")
    print(f"{'='*60}")
    print(f"  Closed-ended: {(closed['gt_normalized'] == closed['pred_normalized']).mean()*100:5.1f}% (n={len(closed)})")
    print(f"  Open-ended:   {(opened['gt_normalized'] == opened['pred_normalized']).mean()*100:5.1f}% (n={len(opened)})")
    print(f"  Overall:      {(df['gt_normalized'] == df['pred_normalized']).mean()*100:5.1f}% (n={len(df)})")


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


# =========================================================================
# Encoder swap
# =========================================================================
def swap_encoder(model, encoder_ckpt_path):
    """Load diffusion-pretrained encoder into model, keeping projector and LLM intact."""
    print(f"Loading adapted encoder from {encoder_ckpt_path}")
    ckpt = torch.load(encoder_ckpt_path, map_location="cpu", weights_only=True)

    if "vision_encoder_state_dict" in ckpt:
        encoder_weights = ckpt["vision_encoder_state_dict"]
        print(f"  Checkpoint final_loss: {ckpt.get('final_loss', 'N/A')}")
    else:
        encoder_weights = ckpt

    vision_tower = model.model.vision_tower.vision_model
    vt_state = vision_tower.state_dict()

    loaded, skipped = 0, 0
    for key, val in encoder_weights.items():
        if key in vt_state and vt_state[key].shape == val.shape:
            vt_state[key] = val
            loaded += 1
        else:
            skipped += 1

    vision_tower.load_state_dict(vt_state)
    print(f"  Loaded {loaded} keys, skipped {skipped}")
    print(f"  Projector: ORIGINAL (not replaced)")
    print(f"  LLM: ORIGINAL (not replaced)")
    return model


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  VQA-RAD EVALUATION — ENCODER ONLY (no projector realign)")
    print("=" * 60)
    print(f"Base model: {args.model_path}")
    print(f"Encoder: {args.encoder_ckpt}")

    # Load original LLaVA
    print(f"\nLoading original LLaVA model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Swap in encoder only
    model = swap_encoder(model, args.encoder_ckpt)
    model.eval()
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Load dataset
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)

    # Run inference
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

    results_df = pd.DataFrame(results)
    results_df['exact_match'] = results_df.apply(
        lambda r: normalize_answer(r['pred_normalized']) == normalize_answer(r['gt_normalized']),
        axis=1
    )
    results_df['contains'] = results_df.apply(
        lambda r: contains_match(r['pred_normalized'], r['gt_normalized']),
        axis=1
    )

    path = os.path.join(args.output_dir, f"{args.tag}_vqa_rad_results_all.csv")
    results_df.to_csv(path, index=False)
    print(f"\nSaved {len(results_df)} predictions to {path}")

    # Print exact match
    train_df = results_df[results_df['split'] == 'train']
    test_df = results_df[results_df['split'] == 'test']
    print_metrics_em(train_df, "TRAIN")
    print_metrics_em(test_df, "TEST")
    print_metrics_em(results_df, "ALL")

    # Judge
    if args.skip_judge:
        print("\n[Skipping LLM judge]")
        results_df.to_csv(
            os.path.join(args.output_dir, f"{args.tag}_vqa_rad_results_final.csv"),
            index=False
        )
        return

    del model
    torch.cuda.empty_cache()

    print(f"\nLoading judge model...")
    judge_tokenizer = AutoTokenizer.from_pretrained(
        args.judge_model, cache_dir=args.judge_cache
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, torch_dtype=torch.float16, cache_dir=args.judge_cache
    ).to("cuda")
    judge_model.generation_config.pad_token_id = judge_tokenizer.eos_token_id
    judge_model.eval()

    open_all = results_df[results_df['answer_type'] == 'OPEN'].copy()
    print(f"Running judge on {len(open_all)} open-ended predictions...")

    judge_verdicts = []
    for i, row in tqdm(open_all.iterrows(), total=len(open_all)):
        verdict = judge_vqa(
            row['question'], row['gt_normalized'], row['pred_normalized'],
            judge_model, judge_tokenizer
        )
        judge_verdicts.append(verdict)

    open_all['llm_judge'] = judge_verdicts
    open_all['llm_judge_correct'] = open_all['llm_judge'] == 'CORRECT'

    results_df = results_df.merge(
        open_all[['qid', 'split', 'llm_judge', 'llm_judge_correct']],
        on=['qid', 'split'], how='left'
    )
    results_df.loc[results_df['answer_type'] == 'CLOSED', 'llm_judge_correct'] = \
        results_df.loc[results_df['answer_type'] == 'CLOSED', 'gt_normalized'] == \
        results_df.loc[results_df['answer_type'] == 'CLOSED', 'pred_normalized']

    final_path = os.path.join(args.output_dir, f"{args.tag}_vqa_rad_results_final.csv")
    results_df.to_csv(final_path, index=False)

    train_df = results_df[results_df['split'] == 'train']
    test_df = results_df[results_df['split'] == 'test']
    print_final(train_df, "TRAIN")
    print_final(test_df, "TEST")
    print_final(results_df, "ALL")

    opened = results_df[results_df['answer_type'] == 'OPEN']
    print(f"\n{'='*60}")
    print(f"  OPEN-ENDED by Question Type (all)")
    print(f"{'='*60}")
    for qtype in sorted(opened['question_type'].unique()):
        subset = opened[opened['question_type'] == qtype]
        em = subset['exact_match'].mean() * 100
        jd = subset['llm_judge_correct'].mean() * 100
        print(f"  {qtype:12s}: EM={em:5.1f}%  Judge={jd:5.1f}%  (n={len(subset)})")

    print(f"\n{'='*60}")
    print(f"  ALL by Image Organ (LLM Judge)")
    print(f"{'='*60}")
    for organ in sorted(results_df['image_organ'].unique()):
        subset = results_df[results_df['image_organ'] == organ]
        print(f"  {organ:12s}: {subset['llm_judge_correct'].mean()*100:5.1f}% (n={len(subset)})")

    print(f"\nDone! Results: {final_path}")


if __name__ == "__main__":
    main()