#!/bin/bash
set -euo pipefail
mkdir -p logs
fixed_rng_job=$(sbatch --parsable run_fixed_rng_selection.sh)
router_v2_job=$(sbatch --parsable run_load_balanced_router.sh)
echo "fixed_rng=${fixed_rng_job}"
echo "router_v2=${router_v2_job}"
