# Early-CLS Dynamic Route Bucketing BERT Benchmark

This benchmark reuses one completed Early-CLS+BCE predictor and adapter checkpoint. It changes only inference scheduling: per-sample route execution versus grouping samples with identical dynamically predicted routes inside each batch.

## Setup

- Checkpoint: `E:\pycharm\ASTER\ASTER-main\aster-bert-paper-params-sweep\checkpoints_distilbert_sst2\early_cls_bce_1_5x\early_cls_topk_contribution_router_bert_checkpoint.pt`
- Target executed layers: `4` / 6
- Eval batch size: `32`
- Timed repeats: `3`
- Warm-up passes: `1`

## Median Results

| Method | Accuracy | Time (s) | Samples/s | Speedup vs Full |
|---|---:|---:|---:|---:|
| Full baseline | 0.9106 | 1.7769 | 490.74 | 1.0000x |
| Dynamic router, per-sample dispatch | 0.8647 | 2.5521 | 341.68 | 0.6963x |
| Dynamic router, route-bucketed dispatch | 0.8647 | 1.3600 | 641.16 | 1.3065x |

## Dynamic Route Distribution

- Unique routes in final bucketed pass: `2`
- `[0, 2, 4, 5]`: 762
- `[0, 1, 4, 5]`: 110

## Interpretation

Per-sample and bucketed modes use exactly the same checkpoint and top-k route rule. Any output mismatch is a correctness failure; timing differences isolate execution scheduling. Batch size 1 cannot benefit from grouping, while larger batches can only group samples whose routes the router independently selected as identical.
