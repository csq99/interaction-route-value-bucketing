# Early-CLS Dynamic Route Bucketing BERT Benchmark

This benchmark reuses one completed Early-CLS+BCE predictor and adapter checkpoint. It changes only inference scheduling: per-sample route execution versus grouping samples with identical dynamically predicted routes inside each batch.

## Setup

- Checkpoint: `E:\pycharm\ASTER\ASTER-main\aster-bert-paper-params-sweep\checkpoints_distilbert_sst2\early_cls_bce_3_0x\early_cls_topk_contribution_router_bert_checkpoint.pt`
- Target executed layers: `2` / 6
- Eval batch size: `32`
- Timed repeats: `3`
- Warm-up passes: `1`

## Median Results

| Method | Accuracy | Time (s) | Samples/s | Speedup vs Full |
|---|---:|---:|---:|---:|
| Full baseline | 0.9106 | 1.6181 | 538.92 | 1.0000x |
| Dynamic router, per-sample dispatch | 0.8257 | 0.9841 | 886.10 | 1.6442x |
| Dynamic router, route-bucketed dispatch | 0.8257 | 0.5603 | 1556.27 | 2.8878x |

## Dynamic Route Distribution

- Unique routes in final bucketed pass: `1`
- `[0, 5]`: 872

## Interpretation

Per-sample and bucketed modes use exactly the same checkpoint and top-k route rule. Any output mismatch is a correctness failure; timing differences isolate execution scheduling. Batch size 1 cannot benefit from grouping, while larger batches can only group samples whose routes the router independently selected as identical.
