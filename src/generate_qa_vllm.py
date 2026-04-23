"""
Generate QA Pairs from MedTrinity Captions (vLLM batch)
=========================================================
Uses vLLM offline batch inference for ~50x speedup over sequential generation.
Generates short-answer medical VQA pairs matching VQA-RAD question types.

Install: pip install vllm --break-system-packages

Usage:
    python generate_qa_vllm.py --max_samples 20000
"""

import os
import sys
import time
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from vllm import LLM, SamplingParams


# =========================================================================
# Args
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Generate QA pairs using vLLM batch")
    p.add_argument("--dataset_path", type=str,
                    default="../data/medtrinity-demo/hf_dataset")
    p.add_argument("--model_name", type=str,
                    default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--model_cache", type=str, default="../models/mistral-judge")
    p.add_argument("--output_dir", type=str, default="../data/medtrinity-qa")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_caption_words", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=512,
                    help="vLLM handles batching internally, this is for chunked processing")
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
            logging.FileHandler(os.path.join(output_dir, "generate.log")),
        ],
    )
    return logging.getLogger(__name__)


# =========================================================================
# Prompt & Parsing
# =========================================================================
QA_PROMPT_TEMPLATE = """You are a medical VQA dataset creator. Given a medical image caption, generate 3-5 question-answer pairs that could be asked about the image.

Rules:
- Answers must be SHORT (1-5 words maximum)
- Questions should cover different aspects: imaging modality, organ/body part, orientation/plane, abnormalities, specific findings
- Only generate questions that can be CLEARLY answered from the caption
- Do NOT generate questions about information not in the caption
- Format each QA pair as: Q: [question] | A: [short answer] | T: [type]
- Types: MODALITY, ORGAN, PLANE, ABN, PRES, ATTRIB, POS, OTHER

Caption: {caption}

Generate QA pairs (one per line, no numbering):"""

VALID_TYPES = {'MODALITY', 'ORGAN', 'PLANE', 'ABN', 'PRES', 'ATTRIB', 'POS',
               'COLOR', 'COUNT', 'SIZE', 'OTHER'}


def truncate_caption(caption, max_words=150):
    words = caption.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return caption


def build_chat_prompt(caption, max_caption_words=150):
    """Build the chat-formatted prompt for Mistral-Instruct."""
    caption = truncate_caption(caption, max_caption_words)
    content = QA_PROMPT_TEMPLATE.format(caption=caption)
    # Mistral-Instruct chat format
    return f"[INST] {content} [/INST]"


def parse_qa_response(response):
    """Parse LLM response into structured QA pairs."""
    qa_pairs = []
    for line in response.split('\n'):
        line = line.strip()
        if not line or 'Q:' not in line or 'A:' not in line:
            continue
        try:
            parts = line.split('|')
            if len(parts) < 2:
                continue

            question = parts[0].strip()
            if question.startswith('Q:'):
                question = question[2:].strip()

            answer = parts[1].strip()
            if answer.startswith('A:'):
                answer = answer[2:].strip()

            qtype = "OTHER"
            if len(parts) >= 3:
                t = parts[2].strip()
                if t.startswith('T:'):
                    t = t[2:].strip().upper()
                    if t in VALID_TYPES:
                        qtype = t

            # Validate
            if not question or not answer:
                continue
            if len(answer.split()) > 10:
                continue

            qa_pairs.append({
                'question': question,
                'answer': answer,
                'question_type': qtype,
            })
        except Exception:
            continue

    return qa_pairs


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("  GENERATE QA PAIRS — vLLM BATCH")
    logger.info("=" * 60)
    logger.info(f"Args: {vars(args)}")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    raw_dataset = load_from_disk(args.dataset_path)
    if args.max_samples < len(raw_dataset):
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(args.max_samples))
        logger.info(f"Shuffled and selected {len(raw_dataset)} samples")

    # Build all prompts
    logger.info("Building prompts...")
    prompts = []
    indices = []
    for idx in range(len(raw_dataset)):
        caption = raw_dataset[idx]['caption']
        prompt = build_chat_prompt(caption, args.max_caption_words)
        prompts.append(prompt)
        indices.append(idx)
    logger.info(f"Built {len(prompts)} prompts")

    # Initialize vLLM
    logger.info(f"Loading vLLM model: {args.model_name}...")
    # Check if model is cached locally
    model_path = args.model_cache
    if not os.path.exists(os.path.join(model_path, "config.json")):
        # Look for HF cache structure
        import glob
        cache_dirs = glob.glob(os.path.join(model_path, "models--*", "snapshots", "*"))
        if cache_dirs:
            model_path = cache_dirs[0]
            logger.info(f"  Found cached model at {model_path}")
        else:
            model_path = args.model_name
            logger.info(f"  Using HF model name: {model_path}")

    llm = LLM(
        model=model_path,
        dtype="half",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        max_tokens=300,
        temperature=0,
        top_p=1.0,
    )

    # Generate in batches
    logger.info(f"Generating QA pairs for {len(prompts)} captions...")
    start_time = time.time()

    all_outputs = llm.generate(prompts, sampling_params)

    elapsed = time.time() - start_time
    logger.info(f"vLLM generation complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Throughput: {len(prompts)/elapsed:.1f} samples/s")

    # Parse responses
    logger.info("Parsing QA pairs from responses...")
    qa_records = []
    empty_count = 0

    for idx, output in zip(indices, all_outputs):
        response = output.outputs[0].text
        qa_pairs = parse_qa_response(response)

        if len(qa_pairs) == 0:
            empty_count += 1

        sample = raw_dataset[idx]
        for qa in qa_pairs:
            qa_records.append({
                'image_idx': idx,
                'image_id': sample.get('id', str(idx)),
                'question': qa['question'],
                'answer': qa['answer'],
                'question_type': qa['question_type'],
                'source_caption': sample['caption'][:500],
            })

    logger.info(f"Parsed {len(qa_records)} QA pairs from {len(prompts)} captions")
    logger.info(f"Images with no QA: {empty_count}/{len(prompts)}")

    # Save CSV
    df = pd.DataFrame(qa_records)
    csv_path = os.path.join(args.output_dir, "qa_pairs_final.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    # Stats
    logger.info(f"\n{'='*60}")
    logger.info(f"  QA DATASET STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total QA pairs: {len(df)}")
    logger.info(f"Unique images: {df['image_idx'].nunique()}")
    logger.info(f"Avg QA per image: {len(df) / max(df['image_idx'].nunique(), 1):.1f}")

    logger.info(f"\nBy question type:")
    for qtype in sorted(df['question_type'].unique()):
        n = len(df[df['question_type'] == qtype])
        logger.info(f"  {qtype:12s}: {n:6d} ({n/len(df)*100:.1f}%)")

    df['answer_words'] = df['answer'].apply(lambda x: len(str(x).split()))
    logger.info(f"\nAnswer length:")
    logger.info(f"  Mean: {df['answer_words'].mean():.1f} words")
    logger.info(f"  1 word:    {(df['answer_words']==1).sum()}")
    logger.info(f"  2-3 words: {((df['answer_words']>=2) & (df['answer_words']<=3)).sum()}")
    logger.info(f"  4+ words:  {(df['answer_words']>=4).sum()}")

    logger.info(f"\nSample QA pairs:")
    for _, row in df.sample(min(15, len(df)), random_state=42).iterrows():
        logger.info(f"  [{row['question_type']:8s}] Q: {row['question']}")
        logger.info(f"             A: {row['answer']}")

    # Build HF dataset with images
    logger.info(f"\nBuilding HF dataset with images...")
    image_qa_map = {}
    for _, row in df.iterrows():
        idx = row['image_idx']
        if idx not in image_qa_map:
            image_qa_map[idx] = []
        image_qa_map[idx].append({
            'question': row['question'],
            'answer': row['answer'],
            'question_type': row['question_type'],
        })

    final_records = []
    for img_idx, qa_list in tqdm(image_qa_map.items(), desc="Building dataset"):
        image = raw_dataset[img_idx]['image']
        for qa in qa_list:
            final_records.append({
                'image': image,
                'question': qa['question'],
                'answer': qa['answer'],
                'question_type': qa['question_type'],
            })

    final_dataset = Dataset.from_list(final_records)
    hf_path = os.path.join(args.output_dir, "hf_dataset")
    final_dataset.save_to_disk(hf_path)
    logger.info(f"Saved HF dataset: {hf_path} ({len(final_dataset)} samples)")

    logger.info(f"\n{'='*60}")
    logger.info(f"  DONE!")
    logger.info(f"{'='*60}")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  HF:  {hf_path}")
    logger.info(f"  Total QA pairs: {len(df)}")
    logger.info(f"  Generation time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()