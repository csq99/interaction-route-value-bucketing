# Early-CLS Top-K BCE DeiT Code-Split Run

This folder is for the proposed improvement, separated from the original ASTER-DeiT reproduction folder.

Method:

- Use the ASTER released-code-style ImageNet validation split: 40K train / 5K test.
- Always execute the first and final DeiT encoder layers.
- Execute layer 0 first.
- Use the layer-0 CLS representation to predict multi-label top-k middle-layer selections.
- Train the predictor with `BCEWithLogitsLoss` from true top-k labels derived from per-layer `cls_cosine_delta`.
- Execute selected layers in ascending order.
- Use ASTER `DynamicAdapter` for skipped-layer compensation.

Upload this whole folder to:

```bash
/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-early-cls-bce-run
```

Run:

```bash
cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-early-cls-bce-run
chmod +x run_early_cls_topk_bce_deit_code_split.sh
sbatch run_early_cls_topk_bce_deit_code_split.sh
```

Outputs:

```bash
comparison_results/early_cls_topk_bce_deit_code_split_speedups_1epoch.md
comparison_results/early_cls_topk_bce_deit_code_split_speedups_1epoch.csv
```

## Unified 1-5 Epoch Curve

Use this script when you want one predictor+adapter checkpoint to control all target speedups by changing `top-k`, instead of retraining one model per speedup:

```bash
cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-early-cls-bce-run
chmod +x run_unified_early_cls_topk_bce_deit_epochs.sh
sbatch run_unified_early_cls_topk_bce_deit_epochs.sh
```

It trains one predictor+adapter for 5 epochs. The predictor loss averages all requested top-k budgets in each batch, and the adapter is trained across all requested target layer counts in each batch. After each epoch, it evaluates the same checkpoint at `1.71x`, `2.0x`, and `3.0x`.

Outputs:

```bash
comparison_results/unified_early_cls_topk_bce_deit_epoch_curve_1to5.md
comparison_results/unified_early_cls_topk_bce_deit_epoch_curve_1to5.csv
```

Compare against original ASTER results in:

```bash
/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-code-split-run/comparison_results/original_aster_deit_code_split_speedups_1epoch.md
```
