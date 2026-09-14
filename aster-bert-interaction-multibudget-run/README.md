# Shared Multi-Budget Interaction Router

This independent BERT/SST-2 experiment tests one shared `DynamicAdapter` and
one shared interaction-aware route-value router across `2/6`, `3/6`, `4/6`,
and `5/6` executed-layer budgets. It does not overwrite the fixed-`4/6`
experiment.

Training samples a budget uniformly per batch, then samples a random complete
route from that budget. The router is trained once using the same mixed-budget
schedule; at inference each requested budget constrains its legal route set.

The accompanying benchmark will report batch-1 latency separately from
batch-N per-sample and route-bucketed throughput. SST-2 has no parquet I/O;
this is reported as not applicable.

## Local Run

Use the CUDA Conda interpreter, not the Windows Store `python` shim:

```powershell
$ROOT = 'E:\pycharm\ASTER\ASTER-main'
$PY = 'C:\Users\79867\.conda\envs\aster\python.exe'
& $PY "$ROOT\aster-bert-interaction-multibudget-run\train_interaction_multibudget_bert.py" `
  --target_layers 2 3 4 5 `
  --adapter_epochs 3 `
  --router_epochs 1 `
  --routes_per_batch 3 `
  --output_dir "$ROOT\aster-bert-interaction-multibudget-run\results_shared_2to5" `
  --checkpoint_dir "$ROOT\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5"

& $PY "$ROOT\aster-bert-interaction-multibudget-run\benchmark_interaction_multibudget_bert.py" `
  --checkpoint "$ROOT\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_interaction_router.pt" `
  --target_layers 2 3 4 5 `
  --throughput_batch_size 32 `
  --warmup 1 `
  --repeats 3 `
  --output_dir "$ROOT\aster-bert-interaction-multibudget-run\results_shared_2to5\inference_benchmark"
```
