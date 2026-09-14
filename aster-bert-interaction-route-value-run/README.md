# DistilBERT Interaction-Aware Route-Value Experiment

This is an independent port of `aster-deit-interaction-route-value-run` to
DistilBERT/SST-2. Existing experiment directories are not modified.

The scripts use the same prepared bidirectional attention mask as
`DistilBERT.forward` when manually executing individual encoder layers. Results
from the older `results_4of6_coverage3` directory were produced before this
compatibility fix and are retained only as diagnostic history, not as the
corrected formal run.

The experiment has two stages:

1. Train a frozen-backbone `DynamicAdapter` with randomly sampled complete
   routes. Fixed-route accuracy is evaluated after every coverage epoch.
2. Freeze the adapter and train a one-shot route-value router from the hidden
   state after DistilBERT layer 0. The router predicts unary and pairwise
   interactions, and route labels are measured with `CE + logit-KD`.

For the default target `4/6`, DistilBERT has six legal routes. A target of
`2/6` has only one route (`[0, 5]`) and is therefore not an interaction-routing
test.

## Local smoke test

Use the Conda environment that contains CUDA PyTorch:

```powershell
$ROOT = 'E:\pycharm\ASTER\ASTER-main'
$PY = 'C:\Users\79867\.conda\envs\aster\python.exe'
& $PY "$ROOT\aster-bert-interaction-route-value-run\train_interaction_route_value_bert.py" `
  --max_train_samples 64 `
  --max_eval_samples 32 `
  --batch_size 8 `
  --eval_batch_size 8 `
  --coverage_adapter_epochs 1 `
  --router_epochs 1 `
  --routes_per_batch 3 `
  --max_train_batches 2 `
  --output_dir "$ROOT\aster-bert-interaction-route-value-run\smoke_results" `
  --checkpoint_dir "$ROOT\aster-bert-interaction-route-value-run\smoke_checkpoints"
```

## Full local run

```powershell
$ROOT = 'E:\pycharm\ASTER\ASTER-main'
$PY = 'C:\Users\79867\.conda\envs\aster\python.exe'
& $PY "$ROOT\aster-bert-interaction-route-value-run\train_interaction_route_value_bert.py" `
  --target_layers 4 `
  --coverage_adapter_epochs 3 `
  --router_epochs 1 `
  --routes_per_batch 3 `
  --output_dir "$ROOT\aster-bert-interaction-route-value-run\results_4of6" `
  --checkpoint_dir "$ROOT\aster-bert-interaction-route-value-run\checkpoints_4of6"
```

Outputs include `coverage_history.csv`, fixed-route accuracy CSVs for every
adapter epoch, `router_history.csv`, a Markdown report, and checkpoints.

## Inference benchmark

Run the inference benchmark only after the full experiment creates its router
checkpoint. It records batch-1 latency separately from batch-32 throughput.
It excludes model loading, cached dataset loading, and tokenization from both
metrics. SST-2 is not parquet-backed, so parquet I/O is reported as not
applicable rather than merged into inference time.

```powershell
$ROOT = 'E:\pycharm\ASTER\ASTER-main'
$PY = 'C:\Users\79867\.conda\envs\aster\python.exe'
& $PY "$ROOT\aster-bert-interaction-route-value-run\benchmark_interaction_route_value_bert.py" `
  --checkpoint "$ROOT\aster-bert-interaction-route-value-run\checkpoints_4of6_coverage3\interaction_route_value_router.pt" `
  --throughput_batch_size 32 `
  --warmup 1 `
  --repeats 3
```
