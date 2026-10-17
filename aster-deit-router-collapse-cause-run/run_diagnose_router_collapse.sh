#!/bin/bash
#SBATCH -J deit_rt_cause
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -o logs/router_cause_%j.out
#SBATCH -e logs/router_cause_%j.err
set -euo pipefail
module load anaconda3/2023.09
module unload CUDA 2>/dev/null || true
module load CUDA/12.4
source activate aster

ROOT=/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main
RUN=$ROOT/aster-deit-router-collapse-cause-run
PREV=$ROOT/aster-deit-interaction-multibudget-40k-e10-to-e15-run
E1E3=$ROOT/aster-deit-interaction-router-e1-e3-run
SPLITS=$ROOT/aster-deit-code-split-run/data_splits_parquet
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp/${USER}/deit_router_cause_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$RUN/logs"
test -w "$TMPDIR"
test -f "$PREV/checkpoints_e15/shared_adapter_epoch_015.pt"
test -f "$PREV/checkpoints_e15/shared_interaction_router.pt"
test -f "$E1E3/checkpoints/shared_interaction_router_epoch_02.pt"
test -f "$E1E3/checkpoints/shared_interaction_router_epoch_03.pt"

python "$RUN/diagnose_router_collapse.py" \
  --model_id "$ROOT/models/deit-base-patch16-224" \
  --eval_csv "$SPLITS/test.csv" \
  --adapter_checkpoint "$PREV/checkpoints_e15/shared_adapter_epoch_015.pt" \
  --router_epoch1 "$PREV/checkpoints_e15/shared_interaction_router.pt" \
  --router_epoch2 "$E1E3/checkpoints/shared_interaction_router_epoch_02.pt" \
  --router_epoch3 "$E1E3/checkpoints/shared_interaction_router_epoch_03.pt" \
  --max_eval_samples 1024 --batch_size 8 --num_workers 1 \
  --output_dir "$RUN/results/epoch_15_adapter_routers_1_2_3"
