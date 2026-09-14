#!/bin/bash
#SBATCH -J deit_intr
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH -o log/interaction_route_value_%j.out
#SBATCH -e log/interaction_route_value_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

RUN_DIR=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-interaction-route-value-run
ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
cd "$RUN_DIR"
mkdir -p log comparison_results checkpoints_deit

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python train_interaction_route_value_deit.py \
  --model_id "$ROOT/models/deit-base-patch16-224" \
  --train_csv "$ROOT/aster-deit-early-cls-bce-run/data_splits_parquet/train.csv" \
  --eval_csv "$ROOT/aster-deit-early-cls-bce-run/data_splits_parquet/test.csv" \
  --target_layers 6 \
  --max_train_samples 2048 \
  --max_eval_samples 512 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --num_workers 1 \
  --coverage_adapter_epochs 1 \
  --router_epochs 1 \
  --routes_per_batch 8 \
  --checkpoint_dir ./checkpoints_deit/interaction_route_value_2x_pilot \
  --output_dir ./comparison_results \
  --output_name interaction_route_value_deit_2x_pilot.md \
  --csv_name interaction_route_value_deit_2x_pilot.csv
