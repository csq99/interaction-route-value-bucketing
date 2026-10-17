#!/bin/bash
#SBATCH -J deit_fix_e12e15
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o logs/fixed_e12_e15_%j.out
#SBATCH -e logs/fixed_e12_e15_%j.err
set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster
ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-interaction-multibudget-40k-e10-to-e15-run
DIAG=$ROOT/aster-deit-comprehensive-diagnostics-run
MODEL=$ROOT/models/deit-base-patch16-224
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
ROUTER=$RUN/checkpoints_e15/shared_interaction_router.pt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -f "$ROUTER"
mkdir -p "$RUN/logs" "$RUN/results_e15/fixed_route_quality"
for EPOCH in 12 15; do
  PADDED=$(printf "%03d" "$EPOCH")
  CKPT=$RUN/checkpoints_e15/shared_adapter_epoch_${PADDED}.pt
  test -f "$CKPT"
  for TARGET in 4 6 7; do
    python "$DIAG/diagnose_routes.py" \
      --model_id "$MODEL" --eval_csv "$SPLITS/test.csv" \
      --adapter_checkpoint "$CKPT" --router_checkpoint "$ROUTER" \
      --adapter_epoch "$EPOCH" --target_layers "$TARGET" \
      --max_eval_samples 512 --batch_size 8 --num_workers 1 \
      --compare_router_scores \
      --output_dir "$RUN/results_e15/fixed_route_quality/epoch_${EPOCH}/target_${TARGET}"
  done
done
