# Shared Multi-Budget Interaction Router: Inference Benchmark

Timing includes tensor transfer, one-shot route scoring, adapter execution, selected encoder layers, and classification. Model loading, dataset loading, and tokenization are outside timed loops.

## Setup

- Checkpoint: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_interaction_router.pt`
- Repeats: `3`; warm-up: `1`
- PyTorch: `2.12.0+cu130`; CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`

## Batch-1 Latency

| Budget | Method | Accuracy | ms/sample | Speedup vs full |
|---:|---|---:|---:|---:|
| 2/6 | Full DistilBERT | 0.9106 | 4.100 | 1.0000x |
| 2/6 | Interaction router, per-sample dispatch | 0.8337 | 2.490 | 1.6467x |
| 3/6 | Full DistilBERT | 0.9106 | 4.118 | 1.0000x |
| 3/6 | Interaction router, per-sample dispatch | 0.8612 | 3.282 | 1.2546x |
| 4/6 | Full DistilBERT | 0.9106 | 4.128 | 1.0000x |
| 4/6 | Interaction router, per-sample dispatch | 0.8761 | 3.794 | 1.0879x |
| 5/6 | Full DistilBERT | 0.9106 | 4.149 | 1.0000x |
| 5/6 | Interaction router, per-sample dispatch | 0.9083 | 4.104 | 1.0109x |

## Batch-N Throughput

Batch size: `32`

| Budget | Method | Accuracy | Samples/s | Speedup vs full |
|---:|---|---:|---:|---:|
| 2/6 | Full DistilBERT | 0.9106 | 607.82 | 1.0000x |
| 2/6 | Interaction router, per-sample dispatch | 0.8337 | 942.37 | 1.5504x |
| 2/6 | Interaction router, route-bucketed dispatch | 0.8337 | 1667.29 | 2.7431x |
| 3/6 | Full DistilBERT | 0.9106 | 600.30 | 1.0000x |
| 3/6 | Interaction router, per-sample dispatch | 0.8612 | 551.93 | 0.9194x |
| 3/6 | Interaction router, route-bucketed dispatch | 0.8612 | 1059.98 | 1.7657x |
| 4/6 | Full DistilBERT | 0.9106 | 602.35 | 1.0000x |
| 4/6 | Interaction router, per-sample dispatch | 0.8761 | 432.71 | 0.7184x |
| 4/6 | Interaction router, route-bucketed dispatch | 0.8761 | 832.97 | 1.3829x |
| 5/6 | Full DistilBERT | 0.9106 | 606.10 | 1.0000x |
| 5/6 | Interaction router, per-sample dispatch | 0.9083 | 383.16 | 0.6322x |
| 5/6 | Interaction router, route-bucketed dispatch | 0.9083 | 694.22 | 1.1454x |

## I/O Boundary

- Parquet I/O: not applicable. SST-2 uses cached Hugging Face Arrow data, not parquet.
- Batch-1 latency and batch-N throughput are separate metrics. Route bucketing is measured only for batch-N.
