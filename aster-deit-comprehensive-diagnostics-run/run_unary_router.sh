#!/bin/bash
# Unary-only Router training with the epoch-10 Adapter frozen.
#SBATCH -J deit_unary
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH -o logs/unary_router_%j.out
#SBATCH -e logs/unary_router_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster
ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-comprehensive-diagnostics-run
MODEL=$ROOT/models/deit-base-patch16-224
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
ADAPTER=$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e10/shared_adapter_epoch_010.pt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -f "$ADAPTER"
mkdir -p "$RUN/logs" "$RUN/results/unary_router"
python "$RUN/train_unary_router.py" --model_id "$MODEL" \
  --train_csv "$SPLITS/train.csv" --eval_csv "$SPLITS/test.csv" \
  --adapter_checkpoint "$ADAPTER" --target_layers 4 6 7 \
  --max_train_samples 40000 --max_eval_samples 5000 --batch_size 16 --eval_batch_size 16 \
  --num_workers 1 --router_epochs 1 --routes_per_batch 8 --output_dir "$RUN/results/unary_router"
