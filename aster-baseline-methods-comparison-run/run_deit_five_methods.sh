#!/bin/bash
#SBATCH -J deit_five
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH -o logs/five_methods_%j.out
#SBATCH -e logs/five_methods_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN_DIR=$ROOT/aster-baseline-methods-comparison-run
cd "$RUN_DIR"
mkdir -p logs results_deit

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python benchmark_deit_five_methods.py \
  --model_id "$ROOT/models/deit-base-patch16-224" \
  --eval_csv "$ROOT/aster-deit-code-split-run/data_splits_parquet/test.csv" \
  --target_specs "1.71x:7,2.0x:6,3.0x:4" \
  --max_eval_samples 5000 \
  --batch_size 16 \
  --adapter_checkpoint "$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e3/shared_adapter_epoch_003.pt" \
  --output_dir "$RUN_DIR/results_deit"
