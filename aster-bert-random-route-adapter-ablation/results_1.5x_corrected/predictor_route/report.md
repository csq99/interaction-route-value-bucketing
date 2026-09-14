# DistilBERT adapter route ablation: `predictor_route`

- Train/eval samples: `67349` / `872`
- Target: `4/6` layers
- Epochs: `3`
- Backbone and predictor frozen; both branches start from the same adapter initialization.

| Epoch | Train loss | Predictor-route accuracy | Best fixed-route accuracy |
|---:|---:|---:|---:|
| 0 | 0.000000 | 0.5092 | 0.5550 |
| 1 | 2.962082 | 0.8521 | 0.8555 |
| 2 | 2.309209 | 0.8658 | 0.8739 |
| 3 | 2.110308 | 0.8578 | 0.8635 |
