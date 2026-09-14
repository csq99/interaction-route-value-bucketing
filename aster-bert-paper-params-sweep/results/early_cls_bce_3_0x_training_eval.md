# Early-CLS Top-K Contribution Router BERT Experiment

This experiment first executes DistilBERT layer 0, uses the layer-0 CLS representation to predict contribution scores for middle layers, selects top-k middle layers, and always executes the final layer.

This version can train the contribution predictor either by regression/ranking or by multi-label BCE. The multi-label BCE variant converts each sample's true top-k middle layers into selected/not-selected labels, so route mistakes such as predicting `[2,4]` when the label is `[1,4]` are directly penalized.

## Setup

- Label metric: per-middle-layer `cls_cosine_delta = 1 - cosine(input_CLS, output_CLS)`
- Predictor loss: multi-label `BCEWithLogitsLoss` on true top-k middle-layer selections
- Forced layers: `[0, 5]`
- Candidate middle layers: `[1, 2, 3, 4]`
- Target executed layers: `2` / `6`
- Middle layers selected per sample: `0`
- Checkpoint: `./checkpoints_distilbert_sst2\early_cls_bce_3_0x\early_cls_topk_contribution_router_bert_checkpoint.pt`

## Results

| Method | Accuracy | Correct / Total | Time (s) | Samples/s | ms/sample |
|---|---:|---:|---:|---:|---:|
| Full baseline bs=1 | 0.9106 | 794 / 872 | 5.6485 | 154.38 | 6.478 |
| Early-CLS top-k router bs=1 | 0.8257 | 720 / 872 | 3.4948 | 249.51 | 4.008 |

## Runtime And Routing

- Predictor training wall time: `179.02` seconds
- Adapter training wall time: `1061.50` seconds
- Average executed layers: `2.0000` / `6`
- Average skipped layers: `4.0000` / `6`
- Theoretical speedup from avg executed layers: `3.0000x`
- Actual speedup vs full bs=1: `1.6163x`

## Executed Route Distribution

| Route | Samples | Accuracy |
|---|---:|---:|
| `[0, 5]` | 872 | 0.8257 |
