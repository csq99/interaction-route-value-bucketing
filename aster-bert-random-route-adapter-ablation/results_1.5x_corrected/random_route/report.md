# DistilBERT adapter route ablation: `random_route`

- Train/eval samples: `67349` / `872`
- Target: `4/6` layers
- Epochs: `3`
- Backbone and predictor frozen; both branches start from the same adapter initialization.

| Epoch | Train loss | Predictor-route accuracy | Best fixed-route accuracy |
|---:|---:|---:|---:|
| 0 | 0.000000 | 0.5092 | 0.5550 |
| 1 | 2.896017 | 0.8463 | 0.8704 |
| 2 | 2.247864 | 0.8601 | 0.8727 |
| 3 | 2.065946 | 0.8578 | 0.8842 |
