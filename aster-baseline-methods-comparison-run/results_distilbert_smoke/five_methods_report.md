# Five-method DistilBERT/SST-2 comparison

Early-exit uses the final classification head at intermediate depth and has no auxiliary exit-head training. SkipDecode and ShortGPT are classification transfer proxies, not full reproductions of their decoder-generation methods.

- Samples: `16`; target: `4/6`; adapter: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_adapter_epoch_003.pt`
- ShortGPT proxy route: `[0, 2, 3, 5]`; early threshold: `0.95`; SkipDecode threshold: `0.85`

| Batch | Method | Accuracy | Avg layers | Time (s) | Samples/s | Speedup vs full | Routes | Max route share |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline_full | 1.0000 | 6.00 | 0.0878 | 182.29 | 1.0000x | 1 | 1.0000 |
| 1 | early_exit | 1.0000 | 5.12 | 0.2293 | 69.78 | 0.3828x | 2 | 0.8750 |
| 1 | random_skip | 0.9375 | 4.00 | 0.0693 | 231.03 | 1.2673x | 6 | 0.3125 |
| 1 | skipdecode_proxy | 1.0000 | 6.00 | 0.0753 | 212.59 | 1.1662x | 1 | 1.0000 |
| 1 | shortgpt_proxy | 1.0000 | 4.00 | 0.0605 | 264.46 | 1.4507x | 1 | 1.0000 |
| 8 | baseline_full | 1.0000 | 6.00 | 0.0268 | 598.02 | 1.0000x | 1 | 1.0000 |
| 8 | early_exit | 1.0000 | 5.12 | 0.0283 | 565.77 | 0.9461x | 2 | 0.8750 |
| 8 | random_skip | 1.0000 | 4.00 | 0.0339 | 471.75 | 0.7889x | 5 | 0.2500 |
| 8 | skipdecode_proxy | 1.0000 | 6.00 | 0.0622 | 257.41 | 0.4304x | 1 | 1.0000 |
| 8 | shortgpt_proxy | 1.0000 | 4.00 | 0.0199 | 805.11 | 1.3463x | 1 | 1.0000 |

Model loading, tokenization, and dataset cache discovery are excluded from timed loops. Batch=1 latency and batch-N throughput are separate metrics.
