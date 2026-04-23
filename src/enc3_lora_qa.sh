#!/bin/bash
#SBATCH --cluster=gpu
#SBATCH --partition=h200
#SBATCH --job-name=enc3_qa_lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/enc3_qa_lora_%j.out
#SBATCH --error=logs/enc3_qa_lora_%j.err

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate /ix/cs2770_2026s/abn80/conda/envs/medvlm

echo "============================================"
echo "Job: Encoder (epoch 3) + QA LoRA"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "============================================"

cd /ix/cs2770_2026s/abn80/cs2770_project/src

python lora_qa.py \
    --encoder_ckpt ../checkpoints/diffusion_pretrain/checkpoint_epoch_3.pt \
    --qa_dataset ../data/medtrinity-qa/hf_dataset \
    --max_samples 40000 \
    --batch_size 16 \
    --grad_accum 2 \
    --lr 2e-4 \
    --epochs 1 \
    --num_workers 4 \
    --log_every 50

echo "Finished: $(date)"
echo "Evaluate: python eval_aligned.py --model_path ../checkpoints/lora_qa/enc3_qa_lora/merged_model --tag enc3_qa_lora"