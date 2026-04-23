#!/bin/bash
#SBATCH --cluster=gpu
#SBATCH --partition=h200
#SBATCH --job-name=lora_enc3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lora_enc3_%j.out
#SBATCH --error=logs/lora_enc3_%j.err

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate /ix/cs2770_2026s/abn80/conda/envs/medvlm

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "============================================"

cd /ix/cs2770_2026s/abn80/cs2770_project/src


python lora_medtrinity.py \
    --encoder_ckpt ../checkpoints/diffusion_pretrain/checkpoint_epoch_3.pt \
    --max_samples 20000 \
    --epochs 1 \
    --batch_size 16 \
    --grad_accum 2 
echo "Finished: $(date)"