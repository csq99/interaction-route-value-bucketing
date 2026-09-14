# Shared Multi-Budget Interaction Route-Value Router for DeiT

A single adapter and a single unary-plus-pairwise router are trained once across the listed budgets. Each batch samples a budget uniformly, then a random legal complete route.

## Setup

- Train/eval samples: `40000` / `5000`
- Targets: `[4, 6, 7]` / `12`; legal-route counts: `{4: 45, 6: 210, 7: 252}`
- Adapter epochs: `10`; router epochs: `1`
- Initial adapter: `random initialization`
- Resumed adapter: `local checkpoint (not included in this repository)`
- Adapter wall time: `156543.41` seconds; router wall time: `22167.36` seconds

## Dynamic Accuracy and Route Diversity

| Budget | Theory | Legal routes | Full acc | Dynamic acc | Selected routes | Max route share |
|---:|---:|---:|---:|---:|---:|---:|
| 4/12 | 3.00x | 45 | 0.8150 | 0.2602 | 1 | 1.0000 |
| 6/12 | 2.00x | 210 | 0.8150 | 0.6106 | 3 | 0.5282 |
| 7/12 | 1.71x | 252 | 0.8150 | 0.6984 | 3 | 0.9908 |

- Checkpoint: `local checkpoint (not included in this repository)`
