#!/bin/bash
# Fixed-route quality at adapter epochs 3/5/10. Router comparison is only run for epoch 10.
#SBATCH -J deit_fixqual
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH -o logs/fixed_quality_%j.out
#SBATCH -e logs/fixed_quality_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-comprehensive-diagnostics-run
MODEL=$ROOT/models/deit-base-patch16-224
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
ROUTER=$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e10/shared_interaction_router.pt

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/logs" "$RUN/results/fixed_route_quality" "$RUN/results/oracle_1024"
for EPOCH in 3 5 10; do
  EPOCH_PADDED=$(printf "%03d" "$EPOCH")
  if [ "$EPOCH" -le 3 ]; then CKPT=$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e3/shared_adapter_epoch_${EPOCH_PADDED}.pt; else CKPT=$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e10/shared_adapter_epoch_${EPOCH_PADDED}.pt; fi
  test -f "$CKPT"
  for TARGET in 4 6 7; do
    EXTRA=""
    if [ "$EPOCH" = 10 ]; then EXTRA="--compare_router_scores"; fi
    python "$RUN/diagnose_routes.py" \
      --model_id "$MODEL" --eval_csv "$SPLITS/test.csv" \
      --adapter_checkpoint "$CKPT" --router_checkpoint "$ROUTER" \
      --adapter_epoch "$EPOCH" --target_layers "$TARGET" \
      --max_eval_samples 512 --batch_size 8 --num_workers 1 \
      --output_dir "$RUN/results/fixed_route_quality/epoch_${EPOCH}/target_${TARGET}" $EXTRA
  done
done

# Same epoch-10 checkpoint, larger oracle subset. This is a separate output from fixed-route curves.
for TARGET in 4 6 7; do
  python "$RUN/diagnose_routes.py" \
    --model_id "$MODEL" --eval_csv "$SPLITS/test.csv" \
      --adapter_checkpoint "$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e10/shared_adapter_epoch_010.pt" \
    --router_checkpoint "$ROUTER" --adapter_epoch 10 --target_layers "$TARGET" \
    --max_eval_samples 1024 --batch_size 8 --num_workers 1 --compare_router_scores \
    --output_dir "$RUN/results/oracle_1024/target_${TARGET}"
done
