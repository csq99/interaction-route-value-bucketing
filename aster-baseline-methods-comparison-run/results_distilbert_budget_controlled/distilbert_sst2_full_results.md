# DistilBERT / SST-2 五种基础方法对比

## 实验范围

- 数据：SST-2 validation，全量 `872` 条样本。
- 模型：DistilBERT，6 个 Transformer blocks。
- Baseline (full)：固定执行全部 6 层，不属于任何加速倍率。
- 跳层方法的目标预算：`1.2x = 5/6` 层、`1.5x = 4/6` 层、`3.0x = 2/6` 层。
- 比较方法：Baseline (full)、Early-exit、Random-skip、SkipDecode proxy、ShortGPT proxy。
- 跳层方法使用已有 adapter checkpoint：`aster-bert-interaction-multibudget-run/checkpoints_shared_2to5/shared_adapter_epoch_003.pt`。
- Early-exit 使用最终分类头在中间深度直接分类，没有单独训练 auxiliary exit head。SkipDecode 与 ShortGPT 是迁移到分类任务的 proxy，不是其 decoder-generation 原始算法的完整复现。
- Early-exit 阈值为 `0.95`，SkipDecode proxy 阈值为 `0.85`。对于这两种动态方法，目标层数是最大层数约束，报告同时列实际平均执行层数。

## 准确率

| 方法 | Accuracy |
|---|---:|
| Baseline (full, 固定 6 层) | 91.06% |

下表仅列受目标预算约束的跳层方法；Baseline full 不按倍率拆分。

| 跳层方法 | 1.2x (最多 5/6 层) | 1.5x (最多 4/6 层) | 3.0x (最多 2/6 层) |
|---|---:|---:|---:|
| Early-exit | 90.25% | 75.80% | 49.08% |
| Random-skip | 88.53% | 86.12% | 83.37% |
| SkipDecode proxy | 86.70% | 85.67% | 83.37% |
| ShortGPT proxy | 90.71% | 87.50% | 83.37% |

## Batch=1 延迟

下表的总时长是对全部 872 条样本逐条推理的计时循环耗时；平均单样本延迟由总时长除以 872 得出。Speedup 是相同对比轮次中相对配对 full baseline 计时的实测比值，不是理论 FLOPs 加速比。脚本在三个预算对比轮次里都重复测量 full 作为参照，但每次 full 都执行全部 6 层；这些是重复测量，不代表 full 有 1.2x/1.5x/3.0x 版本。

| 对比轮次（仅用于配对计时） | Full Accuracy | 实际层数 | 872 条 batch=1 总时长 (s) | 平均单样本延迟 (ms) | Samples/s |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 | 91.06% | 6.00 | 3.6704 | 4.209 | 237.57 |
| 第 2 轮 | 91.06% | 6.00 | 3.6586 | 4.196 | 238.34 |
| 第 3 轮 | 91.06% | 6.00 | 3.7071 | 4.251 | 235.23 |

| 预算 | 方法 | Accuracy | 实际平均层数 | 872 条总时长 (s) | 平均单样本延迟 (ms) | Samples/s | Speedup vs 配对 full |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.2x | Early-exit | 90.25% | 5.00 | 5.0075 | 5.742 | 174.14 | 0.733x |
| 1.2x | Random-skip | 88.53% | 5.00 | 3.6076 | 4.137 | 241.71 | 1.017x |
| 1.2x | SkipDecode proxy | 86.70% | 4.89 | 3.8649 | 4.432 | 225.62 | 0.950x |
| 1.2x | ShortGPT proxy | 90.71% | 5.00 | 3.5832 | 4.109 | 243.36 | 1.024x |
| 1.5x | Early-exit | 75.80% | 4.00 | 4.0124 | 4.601 | 217.33 | 0.912x |
| 1.5x | Random-skip | 86.12% | 4.00 | 3.2639 | 3.743 | 267.17 | 1.121x |
| 1.5x | SkipDecode proxy | 85.67% | 3.97 | 3.3699 | 3.864 | 258.76 | 1.086x |
| 1.5x | ShortGPT proxy | 87.50% | 4.00 | 3.3367 | 3.827 | 261.34 | 1.096x |
| 3.0x | Early-exit | 49.08% | 2.00 | 2.3330 | 2.675 | 373.77 | 1.589x |
| 3.0x | Random-skip | 83.37% | 2.00 | 2.2382 | 2.567 | 389.60 | 1.656x |
| 3.0x | SkipDecode proxy | 83.37% | 2.00 | 2.2702 | 2.603 | 384.10 | 1.633x |
| 3.0x | ShortGPT proxy | 83.37% | 2.00 | 2.2479 | 2.578 | 387.92 | 1.649x |

## Batch=32 吞吐计时

这里的总时长仍是覆盖全部 872 条样本的计时循环耗时，但按 batch=32 执行。Samples/s 是该批量设置下的吞吐，不应与 batch=1 单样本延迟混为一谈。以下三条 full 记录同样是三个对比轮次里的重复参考测量，每次都执行完整 6 层，不代表不同倍率。

| 对比轮次（仅用于配对计时） | Full Accuracy | 实际层数 | 872 条 batch=32 总时长 (s) | Samples/s |
|---|---:|---:|---:|---:|
| 第 1 轮 | 91.06% | 6.00 | 1.4090 | 618.89 |
| 第 2 轮 | 91.06% | 6.00 | 1.4245 | 612.16 |
| 第 3 轮 | 91.06% | 6.00 | 1.4376 | 606.56 |

| 预算 | 方法 | Accuracy | 实际平均层数 | 872 条总时长 (s) | Samples/s | Speedup vs 配对 full |
|---|---|---:|---:|---:|---:|---:|
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

## 结果解读与计时边界

- 1.2x 下，ShortGPT proxy 的准确率最高（90.71%）；batch=1 计时相对 full 仅快约 1.02x，batch=32 约快 1.15x。Early-exit 虽有 90.25% 准确率，但 batch=1 比 full 慢。
- 1.5x 下，ShortGPT proxy 的准确率最高（87.50%）；batch=1 和 batch=32 分别约为 full 的 1.10x、1.40x 吞吐/速度比。
- 3.0x 下，Random-skip、SkipDecode proxy、ShortGPT proxy 的准确率相同（83.37%）。SkipDecode proxy 的实际吞吐明显较低，显示动态控制开销会抵消部分跳层收益。
- 计时循环不包含模型加载、tokenization 和数据缓存发现，也不是端到端数据读取计时；没有据此声称端到端加速。
- SkipDecode proxy 在 batch=32 时存在明显额外开销，尤其是 1.2x、1.5x；因此不能只根据跳过的层数推断实际速度。
- 这是一次本地 872 条 validation 的实验记录。文档中的时长和准确率来自落盘 CSV/报告；平均单样本延迟为按已记录总时长计算的派生值。

## 原始记录

- `five_methods_report.md`
- `five_methods_summary.csv`

