# Contribution-Route BERT Experiment
Date: 2026-05-22

## Notes

- This is a first prototype for the proposed representation-change route idea.
- Layer contribution is computed offline from full-model hidden states.
- Static routes remove the per-layer MDP scorer overhead but still use the trained ASTER adapter for skipped jumps.
- This validates runtime/accuracy tendencies; a publishable version should add a one-shot router trained from contribution-derived route labels.

## Layer Contribution

| Layer | CLS Relative Change | Token Relative Change |
|---:|---:|---:|
| 0 | 0.466704 | 0.952476 |
| 1 | 0.657140 | 0.787035 |
| 2 | 0.687797 | 0.598006 |
| 3 | 0.418597 | 0.650658 |
| 4 | 0.909166 | 0.730099 |
| 5 | 0.870598 | 0.824715 |

## Routes

| Route Name | Executed Layers | Theoretical Speedup |
|---|---|---:|
| full | [0, 1, 2, 3, 4, 5] | 1.0000x |
| contribution_4_layers | [0, 2, 4, 5] | 1.5000x |
| contribution_3_layers | [0, 4, 5] | 2.0000x |
| contribution_2_layers | [0, 5] | 3.0000x |
| aster_observed_3_layers | [0, 2, 5] | 2.0000x |

## Runtime And Accuracy

| Method | Accuracy | Correct / Total | Time (s) | Samples/s | ms/sample |
|---|---:|---:|---:|---:|---:|
| Full baseline bs=1 | 0.9106 | 794 / 872 | 3.4161 | 255.26 | 3.918 |
| Full baseline bs=16 | 0.9106 | 794 / 872 | 1.3819 | 631.00 | 1.585 |
| Static contribution_4_layers bs=1 | 0.7924 | 691 / 872 | 2.8068 | 310.67 | 3.219 |
| Static contribution_3_layers bs=1 | 0.7844 | 684 / 872 | 2.1027 | 414.71 | 2.411 |
| Static contribution_2_layers bs=1 | 0.5092 | 444 / 872 | 1.6983 | 513.45 | 1.948 |
| Static aster_observed_3_layers bs=1 | 0.8257 | 720 / 872 | 2.2997 | 379.17 | 2.637 |

## Transition Counts

### Static contribution_4_layers bs=1

- 0 -> 2: 872
- 2 -> 4: 872
- 4 -> 5: 872

### Static contribution_3_layers bs=1

- 0 -> 4: 872
- 4 -> 5: 872

### Static contribution_2_layers bs=1

- 0 -> 5: 872

### Static aster_observed_3_layers bs=1

- 0 -> 2: 872
- 2 -> 5: 872

## Comparison With Previous ASTER-MDP Benchmark

Previous `benchmark_bert_aster.py` result:

| Method | Accuracy | Time (s) | ms/sample | Actual Speedup vs Full bs=1 |
|---|---:|---:|---:|---:|
| Full baseline bs=1 | 0.9106 | 3.6094 | 4.139 | 1.0000x |
| ASTER greedy MDP bs=1 | 0.8257 | 3.5390 | 4.059 | 1.0199x |

Current static ASTER-observed route result:

| Method | Accuracy | Time (s) | ms/sample | Actual Speedup vs Full bs=1 |
|---|---:|---:|---:|---:|
| Full baseline bs=1 | 0.9106 | 3.4161 | 3.918 | 1.0000x |
| Static `[0, 2, 5]` route bs=1 | 0.8257 | 2.2997 | 2.637 | 1.4854x |

The static `[0, 2, 5]` route keeps the same accuracy as the trained ASTER greedy policy because it uses the same observed route and the same trained adapter. However, it removes the per-layer scorer decision overhead, improving actual wall-clock speed from about `1.02x` to `1.49x` versus the full model under batch size 1.

This supports the hypothesis that the MDP/scorer decision overhead is a major reason why theoretical layer speedup does not translate into actual runtime speedup on DistilBERT.

## Interpretation For The Proposed Method

The contribution scores show that layers differ significantly in representation change:

- Lowest CLS-change layers: layer 3, layer 0, layer 1
- Highest CLS-change layers: layer 4, layer 5, layer 2

Using these scores to build static routes gives real speedups, but accuracy drops:

- `[0, 2, 4, 5]`: `0.7924` accuracy at `1.2171x` actual speedup
- `[0, 4, 5]`: `0.7844` accuracy at `1.6246x` actual speedup
- `[0, 5]`: `0.5092` accuracy at `2.0114x` actual speedup

The ASTER-observed route `[0, 2, 5]` is a stronger 3-layer route:

- Accuracy: `0.8257`
- Actual speedup vs full bs=1: `1.4854x`

This means raw input-output contribution alone is not yet sufficient to choose the best route. It is useful as a signal, but should be combined with task loss, KD, or a learned router.

## Research Implication

The experiment supports a refined version of the proposed improvement:

```text
Do not use contribution scores as a naive fixed path selector.
Use contribution scores as supervision or regularization for a low-overhead one-shot router.
```

Recommended next method:

1. Offline compute per-layer representation-change contribution.
2. Generate route candidates with measured real latency.
3. For each training sample, select a pseudo-label route by optimizing:

   ```text
   task_loss(route, x) + lambda * measured_latency(route)
   ```

4. Train a lightweight one-shot router to predict the route before entering the transformer stack.
5. Use contribution-aware KD to preserve accuracy when high-contribution layers are skipped.

This keeps the paper-level innovation beyond fixed paths while directly addressing the measured runtime overhead of MDP-based layer skipping.
