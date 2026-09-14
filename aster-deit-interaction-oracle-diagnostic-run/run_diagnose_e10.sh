#!/bin/bash
#SBATCH -J deit_oracle10
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o logs/oracle_e10_%j.out
#SBATCH -e logs/oracle_e10_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-interaction-oracle-diagnostic-run
MODEL=$ROOT/models/deit-base-patch16-224
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
CHECKPOINT=$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e10/shared_interaction_router.pt

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -f "$CHECKPOINT"
mkdir -p "$RUN/logs" "$RUN/results_epoch10_128"

python "$RUN/diagnose_interaction_router_oracle.py" \
  --model_id "$MODEL" \
  --eval_csv "$SPLITS/test.csv" \
  --checkpoint "$CHECKPOINT" \
  --target_layers 4 6 7 \
  --max_eval_samples 128 \
  --batch_size 8 \
  --num_workers 1 \
  --output_dir "$RUN/results_epoch10_128"
