# Shared Multi-Budget Interaction Router for DeiT

One `DynamicAdapter` and one interaction-aware unary-plus-pairwise router are
trained once across `4/12`, `6/12`, and `7/12` executed-layer budgets. Every
adapter-training batch samples one budget uniformly and then a random legal
complete route. The adapter loss uses final CE/logit KD plus hidden-state KD at
each later real layer. The router uses CE+logit-KD route-value labels.

This is an independent experiment. It does not overwrite the interaction pilot
or adapter-sampling ablation directories.

## Server Run

Upload this entire directory to:

```text
/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-interaction-multibudget-run
```

Then run:

```bash
cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-interaction-multibudget-run
sed -i 's/\r$//' run_shared_multibudget_deit.sh
chmod +x run_shared_multibudget_deit.sh
sbatch run_shared_multibudget_deit.sh
```

The script uses the existing `2048` train / `512` test subset and initially
loads the earlier interaction pilot adapter checkpoint. Its result is therefore
an ablation under the validation-derived code-split protocol, not a full
ImageNet reproduction.

After training, a separate benchmark must report GPU-forward batch-1 latency,
GPU-forward batch-N per-sample and route-bucketed throughput, and parquet
end-to-end throughput as separate metrics.

The included forward-only benchmark may be run after training:

```bash
python benchmark_interaction_multibudget_deit.py \
  --checkpoint checkpoints_shared_4_6_7/shared_interaction_router.pt \
  --eval_csv /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-early-cls-bce-run/data_splits_parquet/test.csv \
  --target_layers 4 6 7 --batch_size 32 --warmup 1 --repeats 3 \
  --output_dir results_shared_4_6_7/gpu_forward_benchmark
```
