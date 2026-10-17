#!/bin/bash
#SBATCH -J deit_sh_e15
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH -o logs/shared_e10_to_e15_%j.out
#SBATCH -e logs/shared_e10_to_e15_%j.err
set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster
ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
SOURCE=$ROOT/aster-deit-interaction-multibudget-run
RUN=$ROOT/aster-deit-interaction-multibudget-40k-e10-to-e15-run
MODEL=$ROOT/models/deit-base-patch16-224
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
RESUME=$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e10/shared_adapter_epoch_010.pt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -f "$RESUME"
mkdir -p "$RUN/logs" "$RUN/checkpoints_e15" "$RUN/results_e15"
python "$SOURCE/train_interaction_multibudget_deit.py" \
  --model_id "$MODEL" \
  --train_csv "$SPLITS/train.csv" \
  --eval_csv "$SPLITS/test.csv" \
  --target_layers 4 6 7 \
  --max_train_samples 40000 \
  --max_eval_samples 5000 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --num_workers 1 \
  --adapter_epochs 15 \
  --router_epochs 1 \
  --routes_per_batch 8 \
  --resume_adapter_checkpoint "$RESUME" \
  --save_epochs 12 15 \
  --checkpoint_dir "$RUN/checkpoints_e15" \
  --output_dir "$RUN/results_e15" \
  --log_interval 10

python "$SOURCE/benchmark_interaction_multibudget_deit.py" \
  --checkpoint "$RUN/checkpoints_e15/shared_interaction_router.pt" \
  --eval_csv "$SPLITS/test.csv" \
  --target_layers 4 6 7 \
  --batch_size 32 \
  --max_eval_samples 5000 \
  --num_workers 1 \
  --warmup 1 \
  --repeats 3 \
  --output_dir "$RUN/results_e15/gpu_forward_benchmark"
