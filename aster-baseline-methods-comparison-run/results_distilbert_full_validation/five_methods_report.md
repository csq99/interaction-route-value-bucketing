# Five-method DistilBERT/SST-2 comparison

Early-exit uses the final classification head at intermediate depth and has no auxiliary exit-head training. SkipDecode and ShortGPT are classification transfer proxies, not full reproductions of their decoder-generation methods.

- Samples: `872`; target: `4/6`; adapter: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_adapter_epoch_003.pt`
- ShortGPT proxy route: `[0, 2, 3, 5]`; early threshold: `0.95`; SkipDecode threshold: `0.85`

| Batch | Method | Accuracy | Avg layers | Time (s) | Samples/s | Speedup vs full | Routes | Max route share |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline_full | 0.9106 | 6.00 | 3.7698 | 231.31 | 1.0000x | 1 | 1.0000 |
| 1 | early_exit | 0.9117 | 5.15 | 5.3490 | 163.02 | 0.7048x | 2 | 0.8486 |
| 1 | random_skip | 0.8612 | 4.00 | 3.4065 | 255.98 | 1.1066x | 6 | 0.1892 |
| 1 | skipdecode_proxy | 0.9106 | 5.99 | 4.8339 | 180.39 | 0.7799x | 2 | 0.9943 |
| 1 | shortgpt_proxy | 0.8750 | 4.00 | 3.5674 | 244.43 | 1.0567x | 1 | 1.0000 |
| 32 | baseline_full | 0.9106 | 6.00 | 1.4083 | 619.19 | 1.0000x | 1 | 1.0000 |
| 32 | early_exit | 0.9117 | 5.15 | 1.3287 | 656.27 | 1.0599x | 2 | 0.8486 |
| 32 | random_skip | 0.8612 | 4.00 | 1.1321 | 770.24 | 1.2439x | 6 | 0.1892 |
| 32 | skipdecode_proxy | 0.9106 | 5.99 | 3.1628 | 275.70 | 0.4453x | 2 | 0.9943 |
| 32 | shortgpt_proxy | 0.8750 | 4.00 | 1.0514 | 829.37 | 1.3394x | 1 | 1.0000 |

Model loading, tokenization, and dataset cache discovery are excluded from timed loops. Batch=1 latency and batch-N throughput are separate metrics.
