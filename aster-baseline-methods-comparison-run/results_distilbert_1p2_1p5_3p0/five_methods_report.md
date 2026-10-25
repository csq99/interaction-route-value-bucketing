# Five-method DistilBERT/SST-2 comparison

Early-exit uses the final classification head at intermediate depth and has no auxiliary exit-head training. SkipDecode and ShortGPT are classification transfer proxies, not full reproductions of their decoder-generation methods.

- Samples: `872`; budgets: `1.2x:5,1.5x:4,3.0x:2`; adapter: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_adapter_epoch_003.pt`
- Early threshold: `0.95`; SkipDecode threshold: `0.85`

| Budget | Target layers | Batch | Method | Accuracy | Avg layers actually run | Time (s) | Samples/s | Speedup vs full | Routes | Max route share |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1.2x | 5 | 1 | baseline_full | 0.9106 | 6.00 | 3.6632 | 238.04 | 1.0000x | 1 | 1.0000 |
| 1.2x | 5 | 1 | early_exit | 0.9117 | 5.15 | 5.1724 | 168.59 | 0.7082x | 2 | 0.8486 |
| 1.2x | 5 | 1 | random_skip | 0.8853 | 5.00 | 3.6128 | 241.36 | 1.0140x | 4 | 0.2672 |
| 1.2x | 5 | 1 | skipdecode_proxy | 0.9106 | 5.99 | 4.3788 | 199.14 | 0.8366x | 2 | 0.9943 |
| 1.2x | 5 | 1 | shortgpt_proxy | 0.9071 | 5.00 | 3.6274 | 240.39 | 1.0099x | 1 | 1.0000 |
| 1.2x | 5 | 32 | baseline_full | 0.9106 | 6.00 | 1.4114 | 617.82 | 1.0000x | 1 | 1.0000 |
| 1.2x | 5 | 32 | early_exit | 0.9117 | 5.15 | 1.3167 | 662.25 | 1.0719x | 2 | 0.8486 |
| 1.2x | 5 | 32 | random_skip | 0.8853 | 5.00 | 1.2708 | 686.16 | 1.1106x | 4 | 0.2672 |
| 1.2x | 5 | 32 | skipdecode_proxy | 0.9106 | 5.99 | 2.9500 | 295.60 | 0.4785x | 2 | 0.9943 |
| 1.2x | 5 | 32 | shortgpt_proxy | 0.9071 | 5.00 | 1.2277 | 710.26 | 1.1496x | 1 | 1.0000 |
| 1.5x | 4 | 1 | baseline_full | 0.9106 | 6.00 | 3.6587 | 238.33 | 1.0000x | 1 | 1.0000 |
| 1.5x | 4 | 1 | early_exit | 0.9117 | 5.15 | 4.9761 | 175.24 | 0.7353x | 2 | 0.8486 |
| 1.5x | 4 | 1 | random_skip | 0.8612 | 4.00 | 3.3005 | 264.20 | 1.1085x | 6 | 0.1892 |
| 1.5x | 4 | 1 | skipdecode_proxy | 0.9106 | 5.99 | 4.3196 | 201.87 | 0.8470x | 2 | 0.9943 |
| 1.5x | 4 | 1 | shortgpt_proxy | 0.8750 | 4.00 | 3.3749 | 258.38 | 1.0841x | 1 | 1.0000 |
| 1.5x | 4 | 32 | baseline_full | 0.9106 | 6.00 | 1.4269 | 611.11 | 1.0000x | 1 | 1.0000 |
| 1.5x | 4 | 32 | early_exit | 0.9117 | 5.15 | 1.3237 | 658.74 | 1.0779x | 2 | 0.8486 |
| 1.5x | 4 | 32 | random_skip | 0.8612 | 4.00 | 1.1002 | 792.58 | 1.2970x | 6 | 0.1892 |
| 1.5x | 4 | 32 | skipdecode_proxy | 0.9106 | 5.99 | 3.0670 | 284.32 | 0.4652x | 2 | 0.9943 |
| 1.5x | 4 | 32 | shortgpt_proxy | 0.8750 | 4.00 | 1.0060 | 866.76 | 1.4183x | 1 | 1.0000 |
| 3.0x | 2 | 1 | baseline_full | 0.9106 | 6.00 | 3.6621 | 238.11 | 1.0000x | 1 | 1.0000 |
| 3.0x | 2 | 1 | early_exit | 0.9117 | 5.15 | 5.0498 | 172.68 | 0.7252x | 2 | 0.8486 |
| 3.0x | 2 | 1 | random_skip | 0.8337 | 2.00 | 2.2331 | 390.49 | 1.6399x | 1 | 1.0000 |
| 3.0x | 2 | 1 | skipdecode_proxy | 0.9106 | 5.99 | 4.2106 | 207.10 | 0.8697x | 2 | 0.9943 |
| 3.0x | 2 | 1 | shortgpt_proxy | 0.8337 | 2.00 | 2.2317 | 390.73 | 1.6409x | 1 | 1.0000 |
| 3.0x | 2 | 32 | baseline_full | 0.9106 | 6.00 | 1.4342 | 607.99 | 1.0000x | 1 | 1.0000 |
| 3.0x | 2 | 32 | early_exit | 0.9117 | 5.15 | 1.3088 | 666.24 | 1.0958x | 2 | 0.8486 |
| 3.0x | 2 | 32 | random_skip | 0.8337 | 2.00 | 0.5379 | 1621.21 | 2.6665x | 1 | 1.0000 |
| 3.0x | 2 | 32 | skipdecode_proxy | 0.9106 | 5.99 | 2.9310 | 297.51 | 0.4893x | 2 | 0.9943 |
| 3.0x | 2 | 32 | shortgpt_proxy | 0.8337 | 2.00 | 0.5248 | 1661.49 | 2.7328x | 1 | 1.0000 |

Model loading, tokenization, and dataset cache discovery are excluded from timed loops. Batch=1 latency and batch-N throughput are separate metrics.
