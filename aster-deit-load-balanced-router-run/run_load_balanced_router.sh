#!/bin/bash
#SBATCH -J deit_lb_rt
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o logs/load_balanced_router_%j.out
#SBATCH -e logs/load_balanced_router_%j.err
set -euo pipefail

module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-load-balanced-router-run
PREV=$ROOT/aster-deit-interaction-multibudget-40k-e10-to-e15-run
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp/${USER}/deit_load_balanced_router_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$RUN/logs"
test -w "$TMPDIR"
test -f "$PREV/checkpoints_e15/shared_adapter_epoch_015.pt"
test -f "$PREV/checkpoints_e15/shared_interaction_router.pt"

python "$RUN/train_load_balanced_router.py" \
  --model_id "$ROOT/models/deit-base-patch16-224" \
  --train_csv "$SPLITS/train.csv" \
  --eval_csv "$SPLITS/test.csv" \
  --adapter_checkpoint "$PREV/checkpoints_e15/shared_adapter_epoch_015.pt" \
  --reference_router_checkpoint "$PREV/checkpoints_e15/shared_interaction_router.pt" \
  --target_layers 4 6 7 \
  --max_train_samples 40000 --max_eval_samples 1024 \
  --batch_size 16 --eval_batch_size 16 --num_workers 1 \
  --routes_per_batch 8 --router_learning_rate 1e-4 \
  --load_balance_weights 0.0 0.01 0.1 1.0 \
  --load_prior_smoothing 0.25 --top1_proxy_temperature 0.1 \
  --output_dir "$RUN/results/router_v2_40k_e1_eval1024" \
  --checkpoint_dir "$RUN/checkpoints/router_v2_40k_e1"
