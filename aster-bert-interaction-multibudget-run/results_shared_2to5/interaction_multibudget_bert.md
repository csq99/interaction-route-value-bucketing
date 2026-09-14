# Shared Multi-Budget Interaction Route-Value Router

One adapter and one interaction router are trained once across all listed budgets. Each training batch samples a budget uniformly, then samples a legal complete route within it.

## Setup

- Train/eval samples: `67349` / `872`
- Target layers: `[2, 3, 4, 5]` / `6`
- Shared adapter epochs: `3`; router epochs: `1`
- Adapter wall time: `644.52` seconds; router wall time: `271.10` seconds

## Accuracy and route diversity

| Layers | Theory | Legal routes | Full acc | Best fixed | Dynamic acc | Selected routes | Max route share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2/6 | 3.00x | 1 | 0.9106 | 0.8337 | 0.8337 | 1 | 1.0000 |
| 3/6 | 2.00x | 4 | 0.9106 | 0.8670 | 0.8612 | 3 | 0.8670 |
| 4/6 | 1.50x | 6 | 0.9106 | 0.8888 | 0.8761 | 3 | 0.9358 |
| 5/6 | 1.20x | 4 | 0.9106 | 0.9071 | 0.9083 | 2 | 0.9083 |

- Checkpoint: `E:\pycharm\ASTER\ASTER-main\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_interaction_router.pt`
