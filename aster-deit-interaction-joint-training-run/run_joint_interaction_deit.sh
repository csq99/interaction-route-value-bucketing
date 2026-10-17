#!/bin/bash
#SBATCH -J deit_joint3
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH -o logs/joint_interaction_%j.out
#SBATCH -e logs/joint_interaction_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-interaction-joint-training-run
PREV=$ROOT/aster-deit-interaction-multibudget-40k-e10-to-e15-run
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
MODEL=$ROOT/models/deit-base-patch16-224

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/logs" "$RUN/checkpoints_joint_e3" "$RUN/results_joint_e3"

test -f "$PREV/checkpoints_e15/shared_adapter_epoch_015.pt"
test -f "$PREV/checkpoints_e15/shared_interaction_router.pt"
test -f "$SPLITS/train.csv"
test -f "$SPLITS/test.csv"

python "$RUN/train_joint_interaction_deit.py" \
  --model_id "$MODEL" \
  --train_csv "$SPLITS/train.csv" \
  --eval_csv "$SPLITS/test.csv" \
  --target_layers 4 6 7 \
  --max_train_samples 40000 \
  --max_eval_samples 1024 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --num_workers 1 \
  --epochs 3 \
  --routes_per_batch 8 \
  --init_adapter_checkpoint "$PREV/checkpoints_e15/shared_adapter_epoch_015.pt" \
  --init_router_checkpoint "$PREV/checkpoints_e15/shared_interaction_router.pt" \
  --checkpoint_dir "$RUN/checkpoints_joint_e3" \
  --output_dir "$RUN/results_joint_e3" \
  --log_interval 10
