#!/bin/bash
#SBATCH -J deit_ast15
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH -o logs/original_aster_e5_to_e15_%j.out
#SBATCH -e logs/original_aster_e5_to_e15_%j.err

set -euo pipefail

PROJECT_ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
SOURCE_DIR="$PROJECT_ROOT/aster-deit-code-split-run"
RUN_DIR="$PROJECT_ROOT/aster-deit-original-aster-40k-e5-to-e15-run"
SOURCE_CHECKPOINT="$SOURCE_DIR/checkpoints_deit/unified_original_aster_deit/original_mdp_epoch_5.pt"
SOURCE_RESULTS="$SOURCE_DIR/comparison_results/unified_original_aster_deit_epoch_curve_1to5.csv"

for required_path in \
  "$SOURCE_DIR/benchmark_original_aster_deit_unified_epochs.py" \
  "$SOURCE_CHECKPOINT" \
  "$SOURCE_RESULTS" \
  "$SOURCE_DIR/data_splits_parquet/train.csv" \
  "$SOURCE_DIR/data_splits_parquet/test.csv" \
  "$PROJECT_ROOT/models/deit-base-patch16-224" \
  "$PROJECT_ROOT/imagenet-1k/data"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/checkpoints"
cd "$RUN_DIR"

echo "===== ORIGINAL ASTER-DEIT EPOCH 5 TO 15 START ====="
date
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python "$SOURCE_DIR/benchmark_original_aster_deit_unified_epochs.py" \
  --model_id "$PROJECT_ROOT/models/deit-base-patch16-224" \
  --imagenet_parquet_dir "$PROJECT_ROOT/imagenet-1k/data" \
  --train_csv "$SOURCE_DIR/data_splits_parquet/train.csv" \
  --eval_csv "$SOURCE_DIR/data_splits_parquet/test.csv" \
  --speedups 1.71,2.0,3.0 \
  --num_epochs 15 \
  --start_epoch 6 \
  --resume_checkpoint "$SOURCE_CHECKPOINT" \
  --initial_cumulative_train_time 60990.37473622337 \
  --append_existing_csv "$SOURCE_RESULTS" \
  --batch_size 16 \
  --eval_batch_size 1 \
  --num_workers 4 \
  --max_train_samples 0 \
  --max_eval_samples 0 \
  --checkpoint_dir "$RUN_DIR/checkpoints" \
  --output_dir "$RUN_DIR/results" \
  --output_name original_aster_deit_epoch_curve_1to15.md \
  --csv_name original_aster_deit_epoch_curve_1to15.csv

echo "===== ORIGINAL ASTER-DEIT EPOCH 5 TO 15 END ====="
date
