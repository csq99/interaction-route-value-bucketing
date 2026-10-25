# Five-method DistilBERT/SST-2 comparison

Early-exit uses the final classification head at intermediate depth and has no auxiliary exit-head training. SkipDecode and ShortGPT are classification transfer proxies, not full reproductions of their decoder-generation methods.

- Samples: `128`; target: `4/6`; adapter: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_adapter_epoch_003.pt`
- ShortGPT proxy route: `[0, 2, 3, 5]`; early threshold: `0.95`; SkipDecode threshold: `0.85`

| Batch | Method | Accuracy | Avg layers | Time (s) | Samples/s | Speedup vs full | Routes | Max route share |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline_full | 0.9141 | 6.00 | 0.5628 | 227.42 | 1.0000x | 1 | 1.0000 |
| 1 | early_exit | 0.9141 | 5.13 | 0.8854 | 144.56 | 0.6357x | 2 | 0.8672 |
| 1 | random_skip | 0.8516 | 4.00 | 0.4792 | 267.11 | 1.1745x | 6 | 0.2109 |
| 1 | skipdecode_proxy | 0.9141 | 6.00 | 0.6073 | 210.78 | 0.9268x | 1 | 1.0000 |
| 1 | shortgpt_proxy | 0.8750 | 4.00 | 0.4899 | 261.27 | 1.1489x | 1 | 1.0000 |
| 32 | baseline_full | 0.9141 | 6.00 | 0.2004 | 638.79 | 1.0000x | 1 | 1.0000 |
| 32 | early_exit | 0.9141 | 5.13 | 0.1846 | 693.52 | 1.0857x | 2 | 0.8672 |
| 32 | random_skip | 0.8281 | 4.00 | 0.1574 | 813.00 | 1.2727x | 6 | 0.2188 |
| 32 | skipdecode_proxy | 0.9141 | 6.00 | 0.4279 | 299.14 | 0.4683x | 1 | 1.0000 |
| 32 | shortgpt_proxy | 0.8750 | 4.00 | 0.1373 | 932.22 | 1.4594x | 1 | 1.0000 |

Model loading, tokenization, and dataset cache discovery are excluded from timed loops. Batch=1 latency and batch-N throughput are separate metrics.
