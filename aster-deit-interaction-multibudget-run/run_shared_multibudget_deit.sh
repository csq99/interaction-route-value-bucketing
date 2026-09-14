#!/bin/bash
#SBATCH -J deit_shared
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH -o logs/shared_multibudget_%j.out
#SBATCH -e logs/shared_multibudget_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-interaction-multibudget-run
MODEL=$ROOT/models/deit-base-patch16-224
SPLITS=$ROOT/aster-deit-early-cls-bce-run/data_splits_parquet

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/logs" "$RUN/results_shared_4_6_7" "$RUN/checkpoints_shared_4_6_7"

python "$RUN/train_interaction_multibudget_deit.py" \
  --model_id "$MODEL" \
  --train_csv "$SPLITS/train.csv" \
  --eval_csv "$SPLITS/test.csv" \
  --target_layers 4 6 7 \
  --max_train_samples 2048 \
  --max_eval_samples 512 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --num_workers 1 \
  --adapter_epochs 10 \
  --router_epochs 1 \
  --routes_per_batch 8 \
  --init_adapter_checkpoint "$ROOT/aster-deit-interaction-route-value-run/checkpoints_deit/interaction_route_value_2x_pilot/coverage_adapter.pt" \
  --checkpoint_dir "$RUN/checkpoints_shared_4_6_7" \
  --output_dir "$RUN/results_shared_4_6_7"

python "$RUN/benchmark_interaction_multibudget_deit.py" \
  --checkpoint "$RUN/checkpoints_shared_4_6_7/shared_interaction_router.pt" \
  --eval_csv "$SPLITS/test.csv" \
  --target_layers 4 6 7 \
  --batch_size 32 \
  --max_eval_samples 512 \
  --num_workers 1 \
  --warmup 1 \
  --repeats 3 \
  --output_dir "$RUN/results_shared_4_6_7/gpu_forward_benchmark"
