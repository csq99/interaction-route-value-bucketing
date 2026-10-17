# DeiT Comprehensive Diagnostics

This directory is independent and does not overwrite previous runs. It uses the
existing shared interaction checkpoint and Adapter checkpoints.

## Experiments

- `run_fixed_quality.sh`: all legal fixed-route accuracies at Adapter epochs 3,
  5, and 10 on 512 test rows; epoch 10 additionally compares interaction and
  unary-only scoring and runs the 1,024-row oracle diagnostic.
- `run_unified_timing.sh`: GPU-forward batch-1 latency, GPU batch-N throughput,
  route-bucketed throughput, and parquet end-to-end timing as separate rows.
- `run_unary_router.sh`: retrains only a unary Router with the epoch-10 Adapter
  frozen, then evaluates dynamic accuracy and route concentration on 5,000 rows.

The 40k/5k split is the existing code split, not a complete ImageNet protocol.
GPU-forward speedups and parquet end-to-end speedups must be reported separately.

## Upload

Upload the entire directory to:

`/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-comprehensive-diagnostics-run`

No model, dataset, or checkpoint files need to be uploaded. The scripts refer to
the existing server paths.

## Server run

```bash
cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-comprehensive-diagnostics-run
sed -i 's/\r$//' run_fixed_quality.sh run_unified_timing.sh run_unary_router.sh submit_all.sh
chmod +x run_fixed_quality.sh run_unified_timing.sh run_unary_router.sh submit_all.sh
bash submit_all.sh
```

Check jobs and logs:

```bash
squeue -u $USER
tail -f logs/fixed_quality_<jobid>.out
tail -f logs/unified_timing_<jobid>.out
tail -f logs/unary_router_<jobid>.out
```

Results are under `results/fixed_route_quality`, `results/oracle_1024`,
`results/unified_timing`, and `results/unary_router`.
