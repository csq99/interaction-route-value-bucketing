# Early-CLS Top-K Contribution Router BERT Experiment

This experiment first executes DistilBERT layer 0, uses the layer-0 CLS representation to predict contribution scores for middle layers, selects top-k middle layers, and always executes the final layer.

This version can train the contribution predictor either by regression/ranking or by multi-label BCE. The multi-label BCE variant converts each sample's true top-k middle layers into selected/not-selected labels, so route mistakes such as predicting `[2,4]` when the label is `[1,4]` are directly penalized.

## Setup

- Label metric: per-middle-layer `cls_cosine_delta = 1 - cosine(input_CLS, output_CLS)`
- Predictor loss: multi-label `BCEWithLogitsLoss` on true top-k middle-layer selections
- Forced layers: `[0, 5]`
- Candidate middle layers: `[1, 2, 3, 4]`
- Target executed layers: `4` / `6`
- Middle layers selected per sample: `2`
- Checkpoint: `./checkpoints_distilbert_sst2\early_cls_bce_1_5x\early_cls_topk_contribution_router_bert_checkpoint.pt`

## Results

| Method | Accuracy | Correct / Total | Time (s) | Samples/s | ms/sample |
|---|---:|---:|---:|---:|---:|
| Full baseline bs=1 | 0.9106 | 794 / 872 | 6.3132 | 138.12 | 7.240 |
| Early-CLS top-k router bs=1 | 0.8647 | 754 / 872 | 6.4733 | 134.71 | 7.423 |

## Runtime And Routing

- Predictor training wall time: `183.48` seconds
- Adapter training wall time: `2021.50` seconds
- Average executed layers: `4.0000` / `6`
- Average skipped layers: `2.0000` / `6`
- Theoretical speedup from avg executed layers: `1.5000x`
- Actual speedup vs full bs=1: `0.9753x`

## Executed Route Distribution

| Route | Samples | Accuracy |
|---|---:|---:|
| `[0, 2, 4, 5]` | 762 | 0.8766 |
| `[0, 1, 4, 5]` | 110 | 0.7818 |
