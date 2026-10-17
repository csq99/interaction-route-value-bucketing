#!/bin/bash
#SBATCH -J deit_adapt_ab
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o log/adapter_sampling_%j.out
#SBATCH -e log/adapter_sampling_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

RUN_DIR=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-adapter-sampling-ablation-run
ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
cd "$RUN_DIR"
mkdir -p log checkpoints_adapter_sampling_ablation results_adapter_sampling_ablation

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python compare_adapter_sampling.py \
  --model_id "$ROOT/models/deit-base-patch16-224" \
  --train_csv "$ROOT/aster-deit-early-cls-bce-run/data_splits_parquet/train.csv" \
  --eval_csv "$ROOT/aster-deit-early-cls-bce-run/data_splits_parquet/test.csv" \
  --target_layers 6 \
  --epochs 10 \
  --eval_epochs 0 1 3 5 10 \
  --max_train_samples 2048 \
  --max_eval_samples 512 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --num_workers 1 \
  --edges_per_batch 4 \
  --init_adapter_checkpoint "$ROOT/aster-deit-interaction-route-value-run/checkpoints_deit/interaction_route_value_2x_pilot/coverage_adapter.pt" \
  --checkpoint_dir ./checkpoints_adapter_sampling_ablation \
  --output_dir ./results_adapter_sampling_ablation
