# Five-method DistilBERT/SST-2 comparison

Early-exit uses the final classification head at intermediate depth and has no auxiliary exit-head training. SkipDecode and ShortGPT are classification transfer proxies, not full reproductions of their decoder-generation methods.

- Samples: `128`; target: `4/6`; adapter: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_adapter_epoch_003.pt`
- ShortGPT proxy route: `[0, 2, 3, 5]`; early threshold: `0.95`; SkipDecode threshold: `0.85`

| Batch | Method | Accuracy | Avg layers | Time (s) | Samples/s | Speedup vs full | Routes | Max route share |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline_full | 0.9141 | 6.00 | 0.6334 | 202.09 | 1.0000x | 1 | 1.0000 |
| 1 | early_exit | 0.9141 | 5.13 | 0.9921 | 129.02 | 0.6384x | 2 | 0.8672 |
| 1 | random_skip | 0.8438 | 4.00 | 0.5076 | 252.17 | 1.2478x | 6 | 0.2188 |
| 1 | skipdecode_proxy | 0.9141 | 6.00 | 0.6444 | 198.64 | 0.9829x | 1 | 1.0000 |
| 1 | shortgpt_proxy | 0.8750 | 4.00 | 0.5008 | 255.57 | 1.2646x | 1 | 1.0000 |
| 32 | baseline_full | 0.9141 | 6.00 | 0.1985 | 644.99 | 1.0000x | 1 | 1.0000 |
| 32 | early_exit | 0.9141 | 5.13 | 0.1840 | 695.69 | 1.0786x | 2 | 0.8672 |
| 32 | random_skip | 0.8438 | 4.00 | 0.1614 | 793.09 | 1.2296x | 6 | 0.2188 |
| 32 | skipdecode_proxy | 0.9141 | 6.00 | 0.4569 | 280.16 | 0.4344x | 1 | 1.0000 |
| 32 | shortgpt_proxy | 0.8750 | 4.00 | 0.1435 | 892.29 | 1.3834x | 1 | 1.0000 |

Model loading, tokenization, and dataset cache discovery are excluded from timed loops. Batch=1 latency and batch-N throughput are separate metrics.
