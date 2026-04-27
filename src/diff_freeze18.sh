#!/bin/bash
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --job-name=diff_freeze18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/diff_freeze18_%j.out
#SBATCH --error=logs/diff_freeze18_%j.err

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate /ix/cs2770_2026s/abn80/conda/envs/medvlm

echo "============================================"
echo "Job: Diffusion Pretrain (freeze layers 0-17)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Started: $(date)"
echo "============================================"

cd /ix/cs2770_2026s/abn80/cs2770_project/src

python diffusion_pretrain_freeze.py \
    --freeze_layers 18 \
    --epochs 10 \
    --batch_size 64 \
    --lr 5e-5 \
    --warmup_steps 100 \
    --num_workers 4 \
    --log_every 50

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "Checkpoints: ../checkpoints/diffusion_pretrain_freeze_18/"
echo ""
echo "Next steps:"
echo "  1. Encoder-only eval:"
echo "     python eval_checkpoint_sweep.py --ckpt_dir ../checkpoints/diffusion_pretrain_freeze_18 --epochs 3 5 8 10 --test_only --skip_judge"
echo "  2. Residual blend eval:"
echo "     python eval_residual_blend.py --encoder_ckpt ../checkpoints/diffusion_pretrain_freeze_18/checkpoint_epoch_3.pt --alphas 0.05 0.1 0.15 0.2 0.3 --with_judge"
echo "============================================"