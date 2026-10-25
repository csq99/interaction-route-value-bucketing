# 五种基础方法实验总汇总

## 1. 结果口径

本文件只汇总 `aster-baseline-methods-comparison-run` 目录中已经落盘的结果，不补造数据。

| 模型 | 主结果目录 | 评估样本 | Transformer 层数 | 目标预算 |
|---|---|---:|---:|---|
| DistilBERT / SST-2 | `results_distilbert_budget_controlled` | 872 | 6 | 1.2x=5/6，1.5x=4/6，3.0x=2/6 |
| DeiT / ImageNet split | `results_deit` | 5000 | 12 | 1.71x=7/12，2.0x=6/12，3.0x=4/12 |

Baseline (full) 没有倍率，始终执行全部 Transformer 层。脚本在每个预算对比轮次中重复测量 full 作为 speedup 参照，但这些重复测量不代表 full 存在不同倍率版本。

五种方法的定义如下：Baseline (full)、Early-exit、Random-skip、SkipDecode proxy、ShortGPT proxy。Early-exit 使用中间深度的最终分类头，没有额外训练 exit head；SkipDecode 和 ShortGPT 是迁移到分类任务的 proxy，不是原始 decoder-generation 算法的完整复现。

计时均为模型加载和 tokenization 完成后的推理循环时间。batch=1 latency 与 batch=32 throughput 分开报告；不把 parquet/数据读取时间混入 GPU-forward 时间。

## 2. DistilBERT / SST-2 主结果

### 2.1 准确率

Baseline full 固定执行 6 层，准确率为 `91.06%`，它不是不同倍率下的不同模型。跳层方法的准确率如下：

| 方法 | Accuracy |
|---|---:|
| Baseline (full) | 91.06% |

| 跳层方法 | 1.2x（最多 5/6 层） | 1.5x（最多 4/6 层） | 3.0x（最多 2/6 层） |
|---|---:|---:|---:|
| Early-exit | 90.25% | 75.80% | 49.08% |
| Random-skip | 88.53% | 86.12% | 83.37% |
| SkipDecode proxy | 86.70% | 85.67% | 83.37% |
| ShortGPT proxy | 90.71% | 87.50% | 83.37% |

表中的 full 行只是为了方便按预算对照；实际模型始终执行 6 层。

### 2.2 Batch=1 推理时间

时间是 872 条样本的计时循环总时长；speedup 是相同对比轮次中相对 full 的实测比值。

| 预算 | 方法 | Accuracy | 平均执行层数 | 总时长 (s) | Samples/s | Speedup vs full |
|---|---|---:|---:|---:|---:|---:|
| Full reference | Baseline (full) | 91.06% | 6.00 | 3.6704 / 3.6586 / 3.7071 | 237.57 / 238.34 / 235.23 | 1.000x |
| 1.2x | Early-exit | 90.25% | 5.00 | 5.0075 | 174.14 | 0.733x |
| 1.2x | Random-skip | 88.53% | 5.00 | 3.6076 | 241.71 | 1.017x |
| 1.2x | SkipDecode proxy | 86.70% | 4.89 | 3.8649 | 225.62 | 0.950x |
| 1.2x | ShortGPT proxy | 90.71% | 5.00 | 3.5832 | 243.36 | 1.024x |
| 1.5x | Early-exit | 75.80% | 4.00 | 4.0124 | 217.33 | 0.912x |
| 1.5x | Random-skip | 86.12% | 4.00 | 3.2639 | 267.17 | 1.121x |
| 1.5x | SkipDecode proxy | 85.67% | 3.97 | 3.3699 | 258.76 | 1.086x |
| 1.5x | ShortGPT proxy | 87.50% | 4.00 | 3.3367 | 261.34 | 1.096x |
| 3.0x | Early-exit | 49.08% | 2.00 | 2.3330 | 373.77 | 1.589x |
| 3.0x | Random-skip | 83.37% | 2.00 | 2.2382 | 389.60 | 1.656x |
| 3.0x | SkipDecode proxy | 83.37% | 2.00 | 2.2702 | 384.10 | 1.633x |
| 3.0x | ShortGPT proxy | 83.37% | 2.00 | 2.2479 | 387.92 | 1.649x |

### 2.3 Batch=32 吞吐时间

| 预算 | 方法 | Accuracy | 平均执行层数 | 总时长 (s) | Samples/s | Speedup vs full |
|---|---|---:|---:|---:|---:|---:|
| Full reference | Baseline (full) | 91.06% | 6.00 | 1.4090 / 1.4245 / 1.4376 | 618.89 / 612.16 / 606.56 | 1.000x |
| 1.2x | Early-exit | 90.25% | 5.00 | 1.2505 | 697.33 | 1.127x |
| 1.2x | Random-skip | 88.53% | 5.00 | 1.2749 | 683.96 | 1.105x |
| 1.2x | SkipDecode proxy | 86.70% | 4.89 | 2.6161 | 333.32 | 0.539x |
| 1.2x | ShortGPT proxy | 90.71% | 5.00 | 1.2239 | 712.45 | 1.151x |
| 1.5x | Early-exit | 75.80% | 4.00 | 1.0254 | 850.39 | 1.389x |
| 1.5x | Random-skip | 86.12% | 4.00 | 1.0938 | 797.19 | 1.302x |
| 1.5x | SkipDecode proxy | 85.67% | 3.97 | 2.0983 | 415.56 | 0.679x |
| 1.5x | ShortGPT proxy | 87.50% | 4.00 | 1.0139 | 860.05 | 1.405x |
| 3.0x | Early-exit | 49.08% | 2.00 | 0.5232 | 1666.57 | 2.748x |
| 3.0x | Random-skip | 83.37% | 2.00 | 0.5345 | 1631.33 | 2.689x |
| 3.0x | SkipDecode proxy | 83.37% | 2.00 | 1.0369 | 840.94 | 1.386x |
| 3.0x | ShortGPT proxy | 83.37% | 2.00 | 0.5187 | 1681.20 | 2.772x |

### 2.4 DistilBERT 结论

- 全量 validation 上，1.2x 和 1.5x 的最高跳层准确率分别是 ShortGPT proxy 的 `90.71%` 和 `87.50%`。
- 3.0x 下 Random-skip、SkipDecode proxy 和 ShortGPT proxy 都是 `83.37%`；Early-exit 降至 `49.08%`。
- batch=1 下，Early-exit 在 1.2x 和 1.5x 反而慢于 full；Random-skip 和 ShortGPT 在三个预算中都有约 1.02x--1.66x 的实测 speedup。
- batch=32 下，3.0x 的 Random-skip 和 ShortGPT 分别达到 `2.689x` 和 `2.772x`；SkipDecode proxy 因控制开销只达到 `1.386x`，低于其它静态方法。

## 3. DeiT / ImageNet split 主结果

这是现有 5000 张 test split 的结果，不是完整 ImageNet 训练协议。Baseline full 固定执行 12 层，准确率约 `81.44%--81.50%`。

### 3.1 Batch=1 推理时间

| 预算 | 方法 | Accuracy | 平均执行层数 | 总时长 (s) | Samples/s | Speedup vs full |
|---|---|---:|---:|---:|---:|---:|
| Full reference | Baseline (full) | 81.44% | 12.00 | 26.3127 / 26.0828 / 25.6898 | 190.02 / 191.70 / 194.63 | 1.000x |
| 1.71x | Early-exit | 0.66% | 7.00 | 25.7592 | 194.11 | 1.021x |
| 1.71x | Random-skip | 54.06% | 7.00 | 20.2108 | 247.39 | 1.302x |
| 1.71x | SkipDecode proxy | 43.68% | 6.96 | 15.7128 | 318.21 | 1.675x |
| 1.71x | ShortGPT proxy | 50.82% | 7.00 | 15.1222 | 330.64 | 1.740x |
| 2.0x | Early-exit | 0.28% | 6.00 | 17.8699 | 279.80 | 1.460x |
| 2.0x | Random-skip | 38.62% | 6.00 | 15.8857 | 314.75 | 1.642x |
| 2.0x | SkipDecode proxy | 17.90% | 6.00 | 13.5209 | 369.80 | 1.929x |
| 2.0x | ShortGPT proxy | 39.00% | 6.00 | 13.6871 | 365.31 | 1.906x |
| 3.0x | Early-exit | 0.10% | 4.00 | 13.1605 | 379.93 | 1.952x |
| 3.0x | Random-skip | 9.62% | 4.00 | 12.1112 | 412.84 | 2.121x |
| 3.0x | SkipDecode proxy | 0.74% | 4.00 | 9.6878 | 516.11 | 2.652x |
| 3.0x | ShortGPT proxy | 9.74% | 4.00 | 10.8488 | 460.88 | 2.368x |

### 3.2 Batch=32 吞吐时间

| 预算 | 方法 | Accuracy | 平均执行层数 | 总时长 (s) | Samples/s | Speedup vs full |
|---|---|---:|---:|---:|---:|---:|
| Full reference | Baseline (full) | 81.50% | 12.00 | 8.8503 / 17.3041 / 10.0491 | 564.95 / 288.95 / 497.56 | 1.000x |
| 1.71x | Early-exit | 0.66% | 7.00 | 4.0815 | 1225.03 | 2.168x |
| 1.71x | Random-skip | 54.02% | 7.00 | 15.3777 | 325.15 | 0.576x |
| 1.71x | SkipDecode proxy | 43.66% | 6.96 | 12.7690 | 391.57 | 0.693x |
| 1.71x | ShortGPT proxy | 50.80% | 7.00 | 3.1928 | 1566.03 | 2.772x |
| 2.0x | Early-exit | 0.28% | 6.00 | 7.4895 | 667.60 | 2.310x |
| 2.0x | Random-skip | 38.62% | 6.00 | 13.9604 | 358.15 | 1.240x |
| 2.0x | SkipDecode proxy | 17.92% | 6.00 | 11.0453 | 452.68 | 1.567x |
| 2.0x | ShortGPT proxy | 39.00% | 6.00 | 2.7905 | 1791.80 | 6.201x |
| 3.0x | Early-exit | 0.10% | 4.00 | 3.7517 | 1332.74 | 2.679x |
| 3.0x | Random-skip | 9.62% | 4.00 | 8.8262 | 566.50 | 1.139x |
| 3.0x | SkipDecode proxy | 0.74% | 4.00 | 7.2737 | 687.41 | 1.382x |
| 3.0x | ShortGPT proxy | 9.72% | 4.00 | 2.0189 | 2476.55 | 4.977x |

### 3.3 DeiT 结论与注意事项

- 在现有实现下，DeiT 跳层后的准确率大幅下降：1.71x 最高为 Random-skip `54.06%`，2.0x 最高为 ShortGPT proxy `39.00%`，3.0x 最高为 ShortGPT proxy `9.74%`。
- DeiT 的 batch=1 计时显示跳层确实减少了推理循环时间，但 batch=32 的结果波动很大，Random-skip 和 SkipDecode 在 1.71x 甚至慢于 full；因此不能只引用理论层数减少来宣称稳定端到端加速。
- 这些 DeiT 五方法结果是分类任务迁移 benchmark，不能写成 Early-exit、SkipDecode、ShortGPT 原论文方法的完整复现。
- 现有数据集是项目中的 5000 张 test split，不能写成完整 ImageNet 训练/评估协议。

## 4. 其它结果目录的定位

| 目录 | 样本/设置 | 用途 | 是否作为主结果 |
|---|---|---|---|
| `results_distilbert_budget_controlled` | 872 条；1.2x/1.5x/3.0x；预算约束修正版 | 正式 DistilBERT 主结果 | 是 |
| `results_deit` | 5000 条；1.71x/2.0x/3.0x | 正式 DeiT 五方法结果 | 是 |
| `results_distilbert_1p2_1p5_3p0` | 872 条；多预算，但早期脚本/口径 | 历史版本，数值与修正版不同 | 否，保留作审计记录 |
| `results_distilbert_full_validation` | 872 条；单一 4/6 层目标 | 旧单预算结果 | 否，不能与三预算主表混用 |
| `results_distilbert_128` | 128 条；4/6 层 | 小样本诊断 | 否 |
| `results_distilbert_128_v2` | 128 条；4/6 层；重复诊断 | 小样本重复诊断 | 否 |
| `results_distilbert_smoke` | 16 条；4/6 层 | 代码 smoke test | 否 |

### 4.1 128 条诊断结果

两个 128 条目录不是全量验证，且属于同一 4/6 层设置。它们显示结果对计时和随机路由有波动，不应替代 872 条主结果。

| 目录 | 方法 | Accuracy | Batch=1 time (s) | Batch=32 time (s) |
|---|---|---:|---:|---:|
| `results_distilbert_128` | Baseline full | 91.41% | 0.5628 | 0.2004 |
| `results_distilbert_128` | Early-exit | 91.41% | 0.8854 | 0.1846 |
| `results_distilbert_128` | Random-skip | 85.16% | 0.4792 | 0.1574 |
| `results_distilbert_128` | SkipDecode proxy | 91.41% | 0.6073 | 0.4279 |
| `results_distilbert_128` | ShortGPT proxy | 87.50% | 0.4899 | 0.1373 |
| `results_distilbert_128_v2` | Baseline full | 91.41% | 0.6334 | 0.1985 |
| `results_distilbert_128_v2` | Early-exit | 91.41% | 0.9921 | 0.1840 |
| `results_distilbert_128_v2` | Random-skip | 84.38% | 0.5076 | 0.1614 |
| `results_distilbert_128_v2` | SkipDecode proxy | 91.41% | 0.6444 | 0.4569 |
| `results_distilbert_128_v2` | ShortGPT proxy | 87.50% | 0.5008 | 0.1435 |

### 4.2 早期 872 条多预算结果

`results_distilbert_1p2_1p5_3p0` 也是 872 条 SST-2 validation，但它是在预算约束口径修正前得到的历史结果。它与主结果的差异主要出现在 Early-exit 和 SkipDecode proxy，因此不能直接和 `results_distilbert_budget_controlled` 拼成一张主表。表里的 full 在每个目标预算轮次均执行完整 6 层，只是重复作为计时参照。

| 预算 | 方法 | Accuracy | Batch=1 总时长 (s) | Batch=32 总时长 (s) |
|---|---|---:|---:|---:|
| 1.2x | Baseline full | 91.06% | 3.6632 | 1.4114 |
| 1.2x | Early-exit | 91.17% | 5.1724 | 1.3167 |
| 1.2x | Random-skip | 88.53% | 3.6128 | 1.2708 |
| 1.2x | SkipDecode proxy | 91.06% | 4.3788 | 2.9500 |
| 1.2x | ShortGPT proxy | 90.71% | 3.6274 | 1.2277 |
| 1.5x | Baseline full | 91.06% | 3.6587 | 1.4269 |
| 1.5x | Early-exit | 91.17% | 4.9761 | 1.3237 |
| 1.5x | Random-skip | 86.12% | 3.3005 | 1.1002 |
| 1.5x | SkipDecode proxy | 91.06% | 4.3196 | 3.0670 |
| 1.5x | ShortGPT proxy | 87.50% | 3.3749 | 1.0060 |
| 3.0x | Baseline full | 91.06% | 3.6621 | 1.4342 |
| 3.0x | Early-exit | 91.17% | 5.0498 | 1.3088 |
| 3.0x | Random-skip | 83.37% | 2.2331 | 0.5379 |
| 3.0x | SkipDecode proxy | 91.06% | 4.2106 | 2.9310 |
| 3.0x | ShortGPT proxy | 83.37% | 2.2317 | 0.5248 |

### 4.3 其它 872 条、16 条结果

`results_distilbert_full_validation` 是旧的单预算 4/6 层结果；`results_distilbert_smoke` 只有 16 条样本，仅用于代码流程检查。

| 目录 | 方法 | Accuracy | Batch=1 time (s) | Batch=32/8 time (s) |
|---|---|---:|---:|---:|
| `results_distilbert_full_validation` | Baseline full | 91.06% | 3.7698 | 1.4083 |
| `results_distilbert_full_validation` | Early-exit | 91.17% | 5.3490 | 1.3287 |
| `results_distilbert_full_validation` | Random-skip | 86.12% | 3.4065 | 1.1321 |
| `results_distilbert_full_validation` | SkipDecode proxy | 91.06% | 4.8339 | 3.1628 |
| `results_distilbert_full_validation` | ShortGPT proxy | 87.50% | 3.5674 | 1.0514 |
| `results_distilbert_smoke` | Baseline full | 100.00% | 0.0878 | 0.0268 |
| `results_distilbert_smoke` | Early-exit | 100.00% | 0.2293 | 0.0283 |
| `results_distilbert_smoke` | Random-skip | 93.75% | 0.0693 | 0.0339 |
| `results_distilbert_smoke` | SkipDecode proxy | 100.00% | 0.0753 | 0.0622 |
| `results_distilbert_smoke` | ShortGPT proxy | 100.00% | 0.0605 | 0.0199 |

## 5. 原始数据和日志

- DistilBERT 主报告：`results_distilbert_budget_controlled/five_methods_report.md`
- DistilBERT 主 CSV：`results_distilbert_budget_controlled/five_methods_summary.csv`
- DeiT 主报告：`results_deit/five_methods_report.md`
- DeiT 主 CSV：`results_deit/five_methods_summary.csv`
- DeiT 作业日志：`logs/five_methods_277433.out`、`logs/five_methods_277433.err`

DeiT 日志已写出 `Saved .../results_deit/five_methods_summary.csv`，stderr 只有模型权重加载进度，没有报错信息。

