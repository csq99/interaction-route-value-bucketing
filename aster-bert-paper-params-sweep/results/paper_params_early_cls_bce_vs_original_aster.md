# DistilBERT/SST-2 Paper-Parameter Comparison

## Controlled Configuration

- Base model: `distilbert-base-uncased-finetuned-sst-2-english`
- Dataset: GLUE/SST-2 validation (872 samples)
- Train batch size: `16`
- Gradient accumulation steps: `4`
- Learning rate: `5e-5` for the new predictor and adapter
- Training epochs: `1` offline predictor-pretraining epoch plus `3` adapter-task-training epochs
- Maximum sequence length: `128`
- New method: execute layer 0 and layer 5, predict top-k middle layers once from layer-0 CLS using multi-label BCE, then run the selected route with DynamicAdapter.
- Original baseline: existing target-controlled ASTER MDP checkpoints trained with the same base training hyperparameters.

## Batch-1 Latency Comparison

| Target | Executed Layers | Full Acc | Original ASTER Acc | Original ASTER Time (s) | New Router Acc | New Router Per-sample Time (s) | New Router Speedup vs Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.2x | 5 / 6 | 0.9106 | 0.8635 | 5.7431 | 0.8842 | 4.1800 | 1.1000x |
| 1.5x | 4 / 6 | 0.9106 | 0.8716 | 4.7803 | 0.8647 | 3.8329 | 1.2442x |
| 3.0x | 2 / 6 | 0.9106 | 0.8291 | 3.2383 | 0.8257 | 1.9061 | 2.0775x |

## Batch-32 Throughput Comparison For New Router

Original ASTER is intentionally omitted here because its public implementation dispatches its greedy MDP sample by sample. The new method is shown both before and after route bucketing.

| Target | Full Time (s) | Per-sample Dynamic Time (s) | Bucketed Time (s) | Bucketed Acc | Bucketed Speedup vs Full |
|---|---:|---:|---:|---:|---:|
| 1.2x | 1.8424 | 2.8853 | 1.6159 | 0.8842 | 1.1401x |
| 1.5x | 1.7769 | 2.5521 | 1.3600 | 0.8647 | 1.3065x |
| 3.0x | 1.6181 | 0.9841 | 0.5603 | 0.8257 | 2.8878x |

## Interpretation Rules

- Theoretical speedup is fixed by the layer budget: `6 / executed_layers`.
- Batch-1 measures latency. Batch-32 measures throughput. They must not be compared as the same metric.
- Route bucketing does not turn routing into a static route: the predictor first selects each sample's route; only samples with identical selected routes are run together.
