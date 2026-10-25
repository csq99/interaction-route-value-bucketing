# Five-method DistilBERT/SST-2 comparison

Early-exit uses the final classification head at intermediate depth and has no auxiliary exit-head training. SkipDecode and ShortGPT are classification transfer proxies, not full reproductions of their decoder-generation methods.

- Samples: `872`; budgets: `1.2x:5,1.5x:4,3.0x:2`; adapter: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_adapter_epoch_003.pt`
- Early threshold: `0.95`; SkipDecode threshold: `0.85`

| Budget | Target layers | Batch | Method | Accuracy | Avg layers actually run | Time (s) | Samples/s | Speedup vs full | Routes | Max route share |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1.2x | 5 | 1 | baseline_full | 0.9106 | 6.00 | 3.6704 | 237.57 | 1.0000x | 1 | 1.0000 |
| 1.2x | 5 | 1 | early_exit | 0.9025 | 5.00 | 5.0075 | 174.14 | 0.7330x | 1 | 1.0000 |
| 1.2x | 5 | 1 | random_skip | 0.8853 | 5.00 | 3.6076 | 241.71 | 1.0174x | 4 | 0.2672 |
| 1.2x | 5 | 1 | skipdecode_proxy | 0.8670 | 4.89 | 3.8649 | 225.62 | 0.9497x | 3 | 0.8888 |
| 1.2x | 5 | 1 | shortgpt_proxy | 0.9071 | 5.00 | 3.5832 | 243.36 | 1.0244x | 1 | 1.0000 |
| 1.2x | 5 | 32 | baseline_full | 0.9106 | 6.00 | 1.4090 | 618.89 | 1.0000x | 1 | 1.0000 |
| 1.2x | 5 | 32 | early_exit | 0.9025 | 5.00 | 1.2505 | 697.33 | 1.1267x | 1 | 1.0000 |
| 1.2x | 5 | 32 | random_skip | 0.8853 | 5.00 | 1.2749 | 683.96 | 1.1051x | 4 | 0.2672 |
| 1.2x | 5 | 32 | skipdecode_proxy | 0.8670 | 4.89 | 2.6161 | 333.32 | 0.5386x | 3 | 0.8888 |
| 1.2x | 5 | 32 | shortgpt_proxy | 0.9071 | 5.00 | 1.2239 | 712.45 | 1.1512x | 1 | 1.0000 |
| 1.5x | 4 | 1 | baseline_full | 0.9106 | 6.00 | 3.6586 | 238.34 | 1.0000x | 1 | 1.0000 |
| 1.5x | 4 | 1 | early_exit | 0.7580 | 4.00 | 4.0124 | 217.33 | 0.9118x | 1 | 1.0000 |
| 1.5x | 4 | 1 | random_skip | 0.8612 | 4.00 | 3.2639 | 267.17 | 1.1209x | 6 | 0.1892 |
| 1.5x | 4 | 1 | skipdecode_proxy | 0.8567 | 3.97 | 3.3699 | 258.76 | 1.0857x | 2 | 0.9748 |
| 1.5x | 4 | 1 | shortgpt_proxy | 0.8750 | 4.00 | 3.3367 | 261.34 | 1.0965x | 1 | 1.0000 |
| 1.5x | 4 | 32 | baseline_full | 0.9106 | 6.00 | 1.4245 | 612.16 | 1.0000x | 1 | 1.0000 |
| 1.5x | 4 | 32 | early_exit | 0.7580 | 4.00 | 1.0254 | 850.39 | 1.3892x | 1 | 1.0000 |
| 1.5x | 4 | 32 | random_skip | 0.8612 | 4.00 | 1.0938 | 797.19 | 1.3023x | 6 | 0.1892 |
| 1.5x | 4 | 32 | skipdecode_proxy | 0.8567 | 3.97 | 2.0983 | 415.56 | 0.6789x | 2 | 0.9748 |
| 1.5x | 4 | 32 | shortgpt_proxy | 0.8750 | 4.00 | 1.0139 | 860.05 | 1.4049x | 1 | 1.0000 |
| 3.0x | 2 | 1 | baseline_full | 0.9106 | 6.00 | 3.7071 | 235.23 | 1.0000x | 1 | 1.0000 |
| 3.0x | 2 | 1 | early_exit | 0.4908 | 2.00 | 2.3330 | 373.77 | 1.5890x | 1 | 1.0000 |
| 3.0x | 2 | 1 | random_skip | 0.8337 | 2.00 | 2.2382 | 389.60 | 1.6563x | 1 | 1.0000 |
| 3.0x | 2 | 1 | skipdecode_proxy | 0.8337 | 2.00 | 2.2702 | 384.10 | 1.6329x | 1 | 1.0000 |
| 3.0x | 2 | 1 | shortgpt_proxy | 0.8337 | 2.00 | 2.2479 | 387.92 | 1.6491x | 1 | 1.0000 |
| 3.0x | 2 | 32 | baseline_full | 0.9106 | 6.00 | 1.4376 | 606.56 | 1.0000x | 1 | 1.0000 |
| 3.0x | 2 | 32 | early_exit | 0.4908 | 2.00 | 0.5232 | 1666.57 | 2.7476x | 1 | 1.0000 |
| 3.0x | 2 | 32 | random_skip | 0.8337 | 2.00 | 0.5345 | 1631.33 | 2.6895x | 1 | 1.0000 |
| 3.0x | 2 | 32 | skipdecode_proxy | 0.8337 | 2.00 | 1.0369 | 840.94 | 1.3864x | 1 | 1.0000 |
| 3.0x | 2 | 32 | shortgpt_proxy | 0.8337 | 2.00 | 0.5187 | 1681.20 | 2.7717x | 1 | 1.0000 |

Model loading, tokenization, and dataset cache discovery are excluded from timed loops. Batch=1 latency and batch-N throughput are separate metrics.
