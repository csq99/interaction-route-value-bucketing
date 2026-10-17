#!/bin/bash
# Unified GPU-only and parquet end-to-end timing using the epoch-10 shared checkpoint.
#SBATCH -J deit_timing
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH -o logs/unified_timing_%j.out
#SBATCH -e logs/unified_timing_%j.err

set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster
ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-comprehensive-diagnostics-run
MODEL=$ROOT/models/deit-base-patch16-224
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
CHECKPOINT=$ROOT/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e10/shared_interaction_router.pt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/logs" "$RUN/results/unified_timing"
python "$RUN/benchmark_unified.py" --model_id "$MODEL" --eval_csv "$SPLITS/test.csv" \
  --checkpoint "$CHECKPOINT" --target_layers 4 6 7 --batch_size 32 --max_eval_samples 5000 \
  --num_workers 1 --warmup 1 --repeats 3 --e2e_repeats 1 --output_dir "$RUN/results/unified_timing"
