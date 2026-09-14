# DistilBERT Interaction Route-Value Inference Benchmark

This report times inference only. The model, checkpoint, cached SST-2 dataset, and tokenized tensors are prepared before timing. Timed loops include device transfer, layer execution, adapter execution, route-value scoring, and classification.

## Setup

- Checkpoint: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-route-value-run\checkpoints_4of6_coverage3\interaction_route_value_router.pt`
- Dataset: cached GLUE/SST-2 validation
- Evaluated samples: `872`
- Timed repeats: `3`; warm-up passes: `1`
- Target: `4 / 6` layers; interaction router scores six legal complete routes.

## Batch-1 Latency

| Method | Accuracy | Time (s) | ms/sample | Speedup vs Full |
|---|---:|---:|---:|---:|
| Full DistilBERT | 0.9106 | 3.6030 | 4.132 | 1.0000x |
| Interaction router, per-sample dispatch | 0.8750 | 3.2408 | 3.716 | 1.1118x |

## Batch-N Throughput

Batch size: `32`

| Method | Accuracy | Time (s) | Samples/s | Speedup vs Full |
|---|---:|---:|---:|---:|
| Full DistilBERT | 0.9106 | 1.4408 | 605.21 | 1.0000x |
| Interaction router, per-sample dispatch | 0.8750 | 1.9176 | 454.73 | 0.7513x |
| Interaction router, route-bucketed dispatch | 0.8750 | 1.0038 | 868.73 | 1.4354x |

## Route Distribution

Batch-1 dynamic route counts:

- `[0, 1, 3, 5]`: 518
- `[0, 2, 3, 5]`: 349
- `[0, 1, 2, 5]`: 5

Batch-N route-bucketed counts:

- `[0, 1, 3, 5]`: 518
- `[0, 2, 3, 5]`: 349
- `[0, 1, 2, 5]`: 5

## I/O Boundary

- Parquet I/O: not applicable. This DistilBERT SST-2 experiment reads Hugging Face cached Arrow data, not ImageNet parquet shards.
- Model loading, dataset cache discovery/loading, and tokenization are excluded from both tables. They must not be interpreted as inference latency or throughput.
- Batch-1 latency and batch-N throughput are separate metrics and are not interchangeable.
