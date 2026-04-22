"""
MAE Residual Blending Evaluation on VQA-RAD
=============================================
features = (1-alpha) * CLIP_original + alpha * MAE_adapted
Uses LLM-as-Judge (Mistral-7B) for open-ended questions.

Usage:
    python eval_mae_blend.py --mae_epoch 1
    python eval_mae_blend.py --mae_epoch 1 --skip_judge
"""

import os
import re
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
import argparse


# =========================================================================
# Config
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  type=str,
                   default="/ix/cs2770_2026s/abn80/cs2770_project/models/llava-1.5-7b-hf")
    p.add_argument("--mae_epoch",   type=int, default=1)
    p.add_argument("--ckpt_dir",    type=str,
                   default="/ix/cs2770_2026s/feg48/checkpoints/mae_pretrain_v2")
    p.add_argument("--alphas",      type=float, nargs="+",
                   default=[0.00, 0.05, 0.10, 0.15, 0.20, 0.50])
    p.add_argument("--judge_model", type=str,
                   default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--judge_cache", type=str,
                   default="/ix/cs2770_2026s/abn80/cs2770_project/models/mistral-judge")
    p.add_argument("--output_dir",  type=str,
                   default="/ix/cs2770_2026s/feg48/results/mae_blend")
    p.add_argument("--skip_judge",  action="store_true")
    p.add_argument("--vqa_rad_parquet", type=str,
                   default="/ix/cs2770_2026s/abn80/cs2770_project/data/vqa-rad/data/test-00000-of-00001.parquet")
    return p.parse_args()


# =========================================================================
# Helpers
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


def load_mae_encoder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "vision_encoder_state_dict" in ckpt:
        return ckpt["vision_encoder_state_dict"]
    state_dict = ckpt.get("model_state_dict", ckpt)
    return {k: v for k, v in state_dict.items() if not k.startswith("decoder.")}


def blend_and_load(model, original_sd, mae_enc, alpha):
    """Blend (1-alpha)*CLIP + alpha*MAE into vision tower."""
    vision_model = model.model.vision_tower.vision_model
    blended = {}

    for key in original_sd:
        # MAE keys have no prefix, vision tower keys also have no prefix
        if key in mae_enc and original_sd[key].shape == mae_enc[key].shape:
            orig = original_sd[key].cpu().float()
            mae  = mae_enc[key].cpu().float()
            blended[key] = ((1-alpha)*orig + alpha*mae).to(original_sd[key].dtype)
        else:
            blended[key] = original_sd[key]

    vision_model.load_state_dict(blended, strict=False)
    return model


def run_inference(model, processor, ds):
    results = []
    for sample in tqdm(ds, desc="  Inference"):
        img = sample['image']
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
            'qid':           sample['qid'],
            'question':      sample['question'],
            'answer_type':   sample['answer_type'],
            'gt_answer':     sample['answer'],
            'gt_normalized': sample['answer_normalized'],
            'pred_raw':      pred_raw,
            'pred_normalized': pred_raw.strip().lower(),
        })

    df = pd.DataFrame(results)
    df['exact_match'] = df.apply(
        lambda r: normalize_answer(r['pred_normalized']) == normalize_answer(r['gt_normalized']),
        axis=1
    )
    return df


def compute_metrics(df, label, alpha=None):
    closed = df[df['answer_type'] == 'CLOSED']
    opened = df[df['answer_type'] == 'OPEN']
    m = {
        'label':      label,
        'alpha':      alpha,
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
    print(f"  MAE RESIDUAL BLENDING — Epoch {args.mae_epoch}")
    print(f"  features = (1-α)×CLIP + α×MAE")
    print("=" * 70)
    print(f"Alphas:    {args.alphas}")
    print(f"LLM Judge: {'No (EM only)' if args.skip_judge else 'Yes (Mistral-7B)'}")

    # Load VQA-RAD
    print("\nLoading VQA-RAD test set ...")
    ds = load_dataset("parquet",
                      data_files={"test": args.vqa_rad_parquet},
                      split="test")
    print(f"  {len(ds)} samples")

    # Load MAE encoder once
    ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch_{args.mae_epoch}.pt")
    print(f"Loading MAE encoder (epoch {args.mae_epoch}) ...")
    mae_enc = load_mae_encoder(ckpt_path)
    print(f"  {len(mae_enc)} keys loaded")

    processor   = AutoProcessor.from_pretrained(args.model_path)
    all_results = {}
    all_metrics = []

    for alpha in args.alphas:
        label = f"α={alpha} (baseline)" if alpha == 0.0 else f"α={alpha}"
        print(f"\n{'='*70}\n  {label}\n{'='*70}")

        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16).to("cuda")

        if alpha > 0:
            # Save original vision tower state dict
            original_sd = {k: v.clone()
                           for k, v in model.model.vision_tower.vision_model.state_dict().items()}
            model = blend_and_load(model, original_sd, mae_enc, alpha)
            print(f"  Blended: (1-{alpha})×CLIP + {alpha}×MAE")

        model.eval()
        df = run_inference(model, processor, ds)
        df.to_csv(os.path.join(args.output_dir,
                               f"alpha_{str(alpha).replace('.','')}_results.csv"), index=False)
        all_results[str(alpha)] = df
        all_metrics.append(compute_metrics(df, label, alpha))
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
            print(f"\n  Judging α={key} ({len(open_df)} open questions)...")
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

            df.to_csv(os.path.join(args.output_dir,
                                   f"alpha_{key.replace('.','')}_results_final.csv"), index=False)
            all_results[key] = df

            for i, m in enumerate(all_metrics):
                if m['alpha'] == float(key):
                    all_metrics[i] = compute_metrics(df, m['label'], float(key))

        del judge_mdl; torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  MAE RESIDUAL BLENDING SUMMARY (Epoch {args.mae_epoch})")
    print(f"{'='*70}")

    if args.skip_judge:
        print(f"\n  {'Alpha':<20} {'Closed EM':>10} {'Open EM':>9} {'Overall EM':>11}")
        print(f"  {'-'*50}")
        for m in all_metrics:
            print(f"  {m['label']:<20} {m['closed_em']:>9.1f}% {m['open_em']:>8.1f}% {m['overall_em']:>10.1f}%")
    else:
        print(f"\n  {'Alpha':<20} {'Cl EM':>6} {'Op EM':>6} {'Op Jdg':>7} {'All EM':>7} {'All Jdg':>8}")
        print(f"  {'-'*55}")
        for m in all_metrics:
            print(
                f"  {m['label']:<20} "
                f"{m['closed_em']:>5.1f}% "
                f"{m['open_em']:>5.1f}% "
                f"{m.get('open_judge', 0):>6.1f}% "
                f"{m['overall_em']:>6.1f}% "
                f"{m.get('overall_judge', 0):>7.1f}%"
            )

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(args.output_dir, "blend_summary.csv"), index=False)
    print(f"\n  Saved to {args.output_dir}/blend_summary.csv")


if __name__ == "__main__":
    main()