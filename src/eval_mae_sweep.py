"""
MAE Checkpoint Sweep Evaluation on VQA-RAD
============================================
Adapted from abn80's eval_checkpoint_sweep.py.
Uses LLM-as-Judge (Mistral-7B) for open-ended questions.

Usage:
    # Full eval with LLM judge
    python eval_mae_sweep.py

    # Skip judge (exact match only, faster)
    python eval_mae_sweep.py --skip_judge
"""

import os
import re
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
import argparse


# =========================================================================
# Config
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  type=str,
                   default="/ix/cs2770_2026s/abn80/cs2770_project/models/llava-1.5-7b-hf")
    p.add_argument("--ckpt_dir",    type=str,
                   default="/ix/cs2770_2026s/feg48/checkpoints/mae_pretrain_v2")
    p.add_argument("--epochs",      type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 15])
    p.add_argument("--judge_model", type=str,
                   default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--judge_cache", type=str,
                   default="/ix/cs2770_2026s/abn80/cs2770_project/models/mistral-judge")
    p.add_argument("--output_dir",  type=str,
                   default="/ix/cs2770_2026s/feg48/results/mae_sweep")
    p.add_argument("--skip_judge",  action="store_true")
    p.add_argument("--vqa_rad_parquet", type=str,
                   default="/ix/cs2770_2026s/abn80/cs2770_project/data/vqa-rad/data/test-00000-of-00001.parquet")
    return p.parse_args()


# =========================================================================
# Helpers  (identical to abn80's script)
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
- Example: GT "axial" vs Pred "axial plane" → CORRECT
- Example: GT "liver" vs Pred "hepatic" → CORRECT
- Example: GT "ct scan" vs Pred "ct" → CORRECT

A prediction is INCORRECT if it:
- Is too vague or generic compared to the ground truth
- Refers to a different structure, condition, or concept
- Example: GT "pulmonary nodules" vs Pred "lung" → INCORRECT
- Example: GT "elliptical" vs Pred "round" → INCORRECT
- Example: GT "ct scan" vs Pred "x-ray" → INCORRECT

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

    return 'CORRECT' if ('CORRECT' in response and 'INCORRECT' not in response) else 'INCORRECT'


# =========================================================================
# MAE encoder extraction  (handles model_state_dict key)
# =========================================================================

def extract_mae_encoder(ckpt_path):
    """Extract encoder weights from MAE checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "vision_encoder_state_dict" in ckpt:
        return ckpt["vision_encoder_state_dict"], ckpt.get("final_loss", "N/A")

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        avg_loss   = ckpt.get("avg_loss", "N/A")
    else:
        state_dict = ckpt
        avg_loss   = "N/A"

    # Discard MAE decoder keys
    encoder_weights = {
        k: v for k, v in state_dict.items()
        if not k.startswith("decoder.")
    }
    return encoder_weights, avg_loss


def swap_encoder(model, encoder_weights):
    """Swap MAE encoder into LLaVA vision tower."""
    vision_model = model.model.vision_tower.vision_model
    vt_state     = vision_model.state_dict()
    loaded = 0

    for key, val in encoder_weights.items():
        # Strip 'vision_model.' prefix if present
        clean_key = key[len("vision_model."):] if key.startswith("vision_model.") else key
        if clean_key in vt_state and vt_state[clean_key].shape == val.shape:
            vt_state[clean_key] = val
            loaded += 1

    vision_model.load_state_dict(vt_state)
    return model, loaded


# =========================================================================
# Inference
# =========================================================================

def run_inference(model, processor, ds):
    results = []
    for sample in tqdm(ds, desc="  Inference"):
        img    = sample['image']
        if img.mode != "RGB":
            img = img.convert("RGB")
        prompt = get_prompt(sample['question'], sample['answer_type'])
        inputs = processor(text=prompt, images=img,
                           return_tensors="pt").to("cuda", torch.float16)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        pred_raw = processor.decode(
            output[0], skip_special_tokens=True
        ).split("ASSISTANT:")[-1].strip()

        results.append({
            'qid':          sample['qid'],
            'question':     sample['question'],
            'answer_type':  sample['answer_type'],
            'gt_answer':    sample['answer'],
            'gt_normalized':sample['answer_normalized'],
            'pred_raw':     pred_raw,
            'pred_normalized': pred_raw.strip().lower(),
        })

    df = pd.DataFrame(results)
    df['exact_match'] = df.apply(
        lambda r: normalize_answer(r['pred_normalized']) == normalize_answer(r['gt_normalized']),
        axis=1
    )
    return df


def compute_metrics(df, label, train_loss=None):
    closed = df[df['answer_type'] == 'CLOSED']
    opened = df[df['answer_type'] == 'OPEN']
    m = {
        'label':      label,
        'train_loss': train_loss,
        'n_total':    len(df),
        'n_closed':   len(closed),
        'n_open':     len(opened),
        'closed_em':  (closed['gt_normalized'] == closed['pred_normalized']).mean() * 100,
        'open_em':    (opened['gt_normalized'] == opened['pred_normalized']).mean() * 100,
        'overall_em': (df['gt_normalized'] == df['pred_normalized']).mean() * 100,
    }
    if 'llm_judge_correct' in df.columns:
        m['closed_judge'] = closed['llm_judge_correct'].mean() * 100
        m['open_judge']   = opened['llm_judge_correct'].mean() * 100
        m['overall_judge']= df['llm_judge_correct'].mean() * 100
    return m


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  MAE CHECKPOINT SWEEP — VQA-RAD Evaluation")
    print("=" * 70)
    print(f"Model:      {args.model_path}")
    print(f"Checkpoints:{args.ckpt_dir}")
    print(f"Epochs:     {args.epochs}")
    print(f"LLM Judge:  {'No (EM only)' if args.skip_judge else 'Yes (Mistral-7B)'}")

    # Load VQA-RAD from local parquet
    print("\nLoading VQA-RAD test set ...")
    ds = load_dataset("parquet",
                      data_files={"test": args.vqa_rad_parquet},
                      split="test")
    print(f"  {len(ds)} samples")

    processor = AutoProcessor.from_pretrained(args.model_path)
    all_results = {}
    all_metrics = []

    # ── Baseline ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  BASELINE\n{'='*70}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16).to("cuda")
    model.eval()
    df = run_inference(model, processor, ds)
    df.to_csv(os.path.join(args.output_dir, "baseline_results.csv"), index=False)
    all_results["baseline"] = df
    all_metrics.append(compute_metrics(df, "Baseline"))
    del model; torch.cuda.empty_cache()

    # ── Epoch sweep ───────────────────────────────────────────────────────
    for epoch in args.epochs:
        ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping epoch {epoch} — not found")
            continue

        print(f"\n{'='*70}\n  MAE Epoch {epoch}\n{'='*70}")
        enc_weights, train_loss = extract_mae_encoder(ckpt_path)
        print(f"  Encoder keys: {len(enc_weights)} | Train loss: {train_loss}")

        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16).to("cuda")
        model, n_loaded = swap_encoder(model, enc_weights)
        model.eval()
        print(f"  Loaded {n_loaded} keys into vision tower")

        df = run_inference(model, processor, ds)
        df.to_csv(os.path.join(args.output_dir, f"epoch_{epoch}_results.csv"), index=False)
        all_results[f"epoch_{epoch}"] = df
        all_metrics.append(compute_metrics(df, f"Epoch {epoch}", train_loss))
        del model; torch.cuda.empty_cache()

    # ── LLM Judge ─────────────────────────────────────────────────────────
    if not args.skip_judge:
        print(f"\n{'='*70}\n  Loading Mistral-7B Judge\n{'='*70}")
        judge_tok = AutoTokenizer.from_pretrained(
            args.judge_model, cache_dir=args.judge_cache)
        judge_mdl = AutoModelForCausalLM.from_pretrained(
            args.judge_model, torch_dtype=torch.float16,
            cache_dir=args.judge_cache).to("cuda")
        judge_mdl.generation_config.pad_token_id = judge_tok.eos_token_id
        judge_mdl.eval()

        for key, df in all_results.items():
            open_df = df[df['answer_type'] == 'OPEN'].copy()
            print(f"\n  Judging {key} ({len(open_df)} open questions)...")
            verdicts = [
                judge_vqa(r['question'], r['gt_normalized'], r['pred_normalized'],
                          judge_mdl, judge_tok)
                for _, r in tqdm(open_df.iterrows(), total=len(open_df))
            ]
            open_df['llm_judge']         = verdicts
            open_df['llm_judge_correct'] = open_df['llm_judge'] == 'CORRECT'
            df = df.merge(open_df[['qid', 'llm_judge', 'llm_judge_correct']],
                          on='qid', how='left')
            df.loc[df['answer_type'] == 'CLOSED', 'llm_judge_correct'] = \
                df.loc[df['answer_type'] == 'CLOSED', 'gt_normalized'] == \
                df.loc[df['answer_type'] == 'CLOSED', 'pred_normalized']

            df.to_csv(os.path.join(args.output_dir, f"{key}_results_final.csv"), index=False)
            all_results[key] = df

            # Update metrics
            label = "Baseline" if key == "baseline" else f"Epoch {key.split('_')[1]}"
            loss  = next((m['train_loss'] for m in all_metrics if m['label'] == label), None)
            for i, m in enumerate(all_metrics):
                if m['label'] == label:
                    all_metrics[i] = compute_metrics(df, label, loss)

        del judge_mdl; torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  MAE CHECKPOINT SWEEP SUMMARY")
    print(f"{'='*70}")

    if args.skip_judge:
        print(f"\n  {'Config':<16} {'Train Loss':>11} {'Closed EM':>10} {'Open EM':>9} {'Overall EM':>11}")
        print(f"  {'-'*57}")
        for m in all_metrics:
            loss = f"{m['train_loss']:>11.4f}" if isinstance(m['train_loss'], float) else f"{'—':>11}"
            print(f"  {m['label']:<16} {loss} {m['closed_em']:>9.1f}% {m['open_em']:>8.1f}% {m['overall_em']:>10.1f}%")
    else:
        print(f"\n  {'Config':<16} {'Loss':>7} {'Cl EM':>6} {'Op EM':>6} {'Op Jdg':>7} {'All EM':>7} {'All Jdg':>8}")
        print(f"  {'-'*60}")
        for m in all_metrics:
            loss = f"{m['train_loss']:>7.4f}" if isinstance(m['train_loss'], float) else f"{'—':>7}"
            print(
                f"  {m['label']:<16} {loss} "
                f"{m['closed_em']:>5.1f}% "
                f"{m['open_em']:>5.1f}% "
                f"{m.get('open_judge', 0):>6.1f}% "
                f"{m['overall_em']:>6.1f}% "
                f"{m.get('overall_judge', 0):>7.1f}%"
            )

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(args.output_dir, "sweep_summary.csv"), index=False)
    print(f"\n  Saved to {args.output_dir}/sweep_summary.csv")


if __name__ == "__main__":
    main()