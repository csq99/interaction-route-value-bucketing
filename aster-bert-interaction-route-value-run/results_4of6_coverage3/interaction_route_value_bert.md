# DistilBERT Interaction-Aware Route-Value Router

The router predicts unary and pairwise route values from the layer-0 hidden state. The adapter is first trained using randomly sampled complete routes.

## Setup

- Train/eval samples: `67349` / `872`
- Target executed layers: `4` / `6`
- Legal complete routes: `6`
- Random-route adapter epochs: `3`
- Router epochs: `1`
- Routes sampled per router batch: `3`
- Coverage training wall time: `708.06` seconds
- Router training/eval wall time: `302.35` seconds

## Random-route adapter coverage

| Epoch | Train loss | Average fixed-route accuracy | Best fixed-route accuracy | Best route |
|---:|---:|---:|---:|---|
| 0 | 0.000000 | 0.5474 | 0.7523 | `[0, 1, 2, 5]` |
| 1 | 2.477715 | 0.8316 | 0.8761 | `[0, 1, 3, 5]` |
| 2 | 1.915694 | 0.8427 | 0.8761 | `[0, 1, 3, 5]` |
| 3 | 1.828054 | 0.8620 | 0.8784 | `[0, 1, 3, 5]` |

## Interaction router

| Epoch | Route-value loss | Router accuracy | Best fixed-route accuracy |
|---:|---:|---:|---:|
| 1 | 1.098389 | 0.8658 | 0.8819 |

## Final route distribution

- Full DistilBERT accuracy: `0.9106`
- Interaction router accuracy: `0.8658`
- `[0, 2, 3, 5]`: 818, accuracy=`0.8631`
- `[0, 1, 3, 5]`: 53, accuracy=`0.9057`
- `[0, 1, 2, 5]`: 1, accuracy=`1.0000`

- Final checkpoint: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-route-value-run\checkpoints_4of6_coverage3\interaction_route_value_router.pt`
