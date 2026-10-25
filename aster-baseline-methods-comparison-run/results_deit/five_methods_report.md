# Five method DeiT comparison

This is a classification benchmark. SkipDecode and ShortGPT are labelled transfer proxies, not claims of reproducing their decoder-generation algorithms. Early-exit uses the final head at intermediate depths and has no auxiliary exit-head training.

- Samples: `5000`; budgets: `1.71x:7,2.0x:6,3.0x:4`; adapter: `/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-interaction-multibudget-40k-e3-run/checkpoints_40k_e3/shared_adapter_epoch_003.pt`
- Early threshold: `0.95`; SkipDecode proxy threshold: `0.85`

| Budget | Target layers | Batch | Method | Accuracy | Avg layers actually run | Time (s) | Samples/s | Speedup vs full | Route count | Max route share |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1.71x | 7 | 1 | baseline_full | 0.8144 | 12.00 | 26.3127 | 190.02 | 1.0000x | 1 | 1.0000 |
| 1.71x | 7 | 1 | early_exit | 0.0066 | 7.00 | 25.7592 | 194.11 | 1.0215x | 1 | 1.0000 |
| 1.71x | 7 | 1 | random_skip | 0.5406 | 7.00 | 20.2108 | 247.39 | 1.3019x | 252 | 0.0070 |
| 1.71x | 7 | 1 | skipdecode_proxy | 0.4368 | 6.96 | 15.7128 | 318.21 | 1.6746x | 2 | 0.9590 |
| 1.71x | 7 | 1 | shortgpt_proxy | 0.5082 | 7.00 | 15.1222 | 330.64 | 1.7400x | 1 | 1.0000 |
| 1.71x | 7 | 32 | baseline_full | 0.8150 | 12.00 | 8.8503 | 564.95 | 1.0000x | 1 | 1.0000 |
| 1.71x | 7 | 32 | early_exit | 0.0066 | 7.00 | 4.0815 | 1225.03 | 2.1684x | 1 | 1.0000 |
| 1.71x | 7 | 32 | random_skip | 0.5402 | 7.00 | 15.3777 | 325.15 | 0.5755x | 252 | 0.0070 |
| 1.71x | 7 | 32 | skipdecode_proxy | 0.4366 | 6.96 | 12.7690 | 391.57 | 0.6931x | 2 | 0.9590 |
| 1.71x | 7 | 32 | shortgpt_proxy | 0.5080 | 7.00 | 3.1928 | 1566.03 | 2.7720x | 1 | 1.0000 |
| 2.0x | 6 | 1 | baseline_full | 0.8144 | 12.00 | 26.0828 | 191.70 | 1.0000x | 1 | 1.0000 |
| 2.0x | 6 | 1 | early_exit | 0.0028 | 6.00 | 17.8699 | 279.80 | 1.4596x | 1 | 1.0000 |
| 2.0x | 6 | 1 | random_skip | 0.3862 | 6.00 | 15.8857 | 314.75 | 1.6419x | 210 | 0.0076 |
| 2.0x | 6 | 1 | skipdecode_proxy | 0.1790 | 6.00 | 13.5209 | 369.80 | 1.9291x | 2 | 0.9974 |
| 2.0x | 6 | 1 | shortgpt_proxy | 0.3900 | 6.00 | 13.6871 | 365.31 | 1.9057x | 1 | 1.0000 |
| 2.0x | 6 | 32 | baseline_full | 0.8150 | 12.00 | 17.3041 | 288.95 | 1.0000x | 1 | 1.0000 |
| 2.0x | 6 | 32 | early_exit | 0.0028 | 6.00 | 7.4895 | 667.60 | 2.3104x | 1 | 1.0000 |
| 2.0x | 6 | 32 | random_skip | 0.3862 | 6.00 | 13.9604 | 358.15 | 1.2395x | 210 | 0.0076 |
| 2.0x | 6 | 32 | skipdecode_proxy | 0.1792 | 6.00 | 11.0453 | 452.68 | 1.5666x | 2 | 0.9974 |
| 2.0x | 6 | 32 | shortgpt_proxy | 0.3900 | 6.00 | 2.7905 | 1791.80 | 6.2011x | 1 | 1.0000 |
| 3.0x | 4 | 1 | baseline_full | 0.8144 | 12.00 | 25.6898 | 194.63 | 1.0000x | 1 | 1.0000 |
| 3.0x | 4 | 1 | early_exit | 0.0010 | 4.00 | 13.1605 | 379.93 | 1.9520x | 1 | 1.0000 |
| 3.0x | 4 | 1 | random_skip | 0.0962 | 4.00 | 12.1112 | 412.84 | 2.1212x | 45 | 0.0278 |
| 3.0x | 4 | 1 | skipdecode_proxy | 0.0074 | 4.00 | 9.6878 | 516.11 | 2.6518x | 1 | 1.0000 |
| 3.0x | 4 | 1 | shortgpt_proxy | 0.0974 | 4.00 | 10.8488 | 460.88 | 2.3680x | 1 | 1.0000 |
| 3.0x | 4 | 32 | baseline_full | 0.8150 | 12.00 | 10.0491 | 497.56 | 1.0000x | 1 | 1.0000 |
| 3.0x | 4 | 32 | early_exit | 0.0010 | 4.00 | 3.7517 | 1332.74 | 2.6786x | 1 | 1.0000 |
| 3.0x | 4 | 32 | random_skip | 0.0962 | 4.00 | 8.8262 | 566.50 | 1.1386x | 45 | 0.0278 |
| 3.0x | 4 | 32 | skipdecode_proxy | 0.0074 | 4.00 | 7.2737 | 687.41 | 1.3816x | 1 | 1.0000 |
| 3.0x | 4 | 32 | shortgpt_proxy | 0.0972 | 4.00 | 2.0189 | 2476.55 | 4.9774x | 1 | 1.0000 |
