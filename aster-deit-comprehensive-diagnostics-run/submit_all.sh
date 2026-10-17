#!/bin/bash
set -euo pipefail
mkdir -p logs
FIXED_JOB=$(sbatch --parsable run_fixed_quality.sh)
TIMING_JOB=$(sbatch --parsable run_unified_timing.sh)
UNARY_JOB=$(sbatch --parsable run_unary_router.sh)
echo "fixed_quality=$FIXED_JOB"
echo "unified_timing=$TIMING_JOB"
echo "unary_router=$UNARY_JOB"
