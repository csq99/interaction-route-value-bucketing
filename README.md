# Interaction Route-Value Bucketing

This repository contains the code and evidence for an interaction-aware
dynamic layer-skipping study on DistilBERT/SST-2 and a controlled DeiT/ImageNet
subset. It is organized around the evidence used in the paper rather than
around a single end-to-end training command.

## Evidence scope

The repository has two separate tracks:

- **DistilBERT/SST-2:** the primary positive study. It evaluates the
  interaction-aware route-value router, the two-stage adapter/router training
  protocol, and route-bucketed execution on the 872-example SST-2 validation
  split.
- **DeiT/ImageNet subset:** a controlled pilot and diagnostic study using the
  project split of 40,000 training images and 5,000 evaluation images. It is
  not a complete ImageNet-1K reproduction and is used to document transfer
  limitations, adapter/path quality, and route-selection behavior.

The standard Interaction pipeline is:

1. Train one shared DynamicAdapter with randomly sampled complete routes and
   budgets.
2. Freeze the backbone and adapter, then train a one-shot route-value router
   with unary and pairwise layer terms.
3. Select a legal ordered route from an early representation and optionally
   group samples with identical routes into route buckets before execution.

The DynamicAdapter architecture is inherited from ASTER. The adapter training
procedure in this repository is separate from router training and should not
be interpreted as the original ASTER training procedure.

## Directory map

### Main Interaction experiments

| Directory | Role in the paper |
|---|---|
| `aster-bert-interaction-route-value-run/` | DistilBERT route-value implementation and fixed-budget experiment |
| `aster-bert-interaction-multibudget-run/` | DistilBERT shared adapter/router checkpoint and timing reports |
| `aster-deit-interaction-route-value-run/` | DeiT pilot route-value implementation |
| `aster-deit-interaction-multibudget-run/` | DeiT shared adapter/router training and GPU-forward benchmark |
| `aster-deit-original-aster-40k-e5-to-e15-run/` | Local original-ASTER DeiT subset comparison curve |
| `results/` | Selected accuracy, route-distribution, timing, and oracle summaries |

### Diagnostics and controls

| Directory | Role in the paper |
|---|---|
| `aster-deit-comprehensive-diagnostics-run/` | Fixed-route quality, oracle, unary-router, and timing diagnostics |
| `aster-deit-interaction-oracle-diagnostic-run/` | Interaction router oracle diagnostic |
| `aster-deit-router-collapse-cause-run/` | Router-epoch and route-concentration diagnosis |
| `aster-deit-load-balanced-router-run/` | Load-balancing and selection-rule diagnostics |
| `aster-deit-interaction-joint-training-run/` | Separate joint-training diagnostic; not the standard checkpoint |

The repository also retains Early-CLS+BCE and adapter-sampling experiments as
historical or supplementary evidence. They are not substituted for the
standard Interaction checkpoint in the main results.

### Cross-method comparison

The cross-method table in the paper uses a separate common evaluation harness
named `aster-baseline-methods-comparison-run/`. That directory is not part of
this anonymous repository snapshot yet. To make the table's evaluation code
available from this artifact, add that directory before publishing the final
artifact link. It evaluates full execution, early-exit, random-skip, SkipDecode
proxy, and ShortGPT proxy on the supplied DistilBERT and DeiT protocols.
SkipDecode and ShortGPT are classification-transfer proxies; their original
papers target decoder-generation or decoder-layer pruning settings,
respectively.

The harness reports accuracy, executed-layer counts, batch-one latency, and
batch-32 throughput separately. The DeiT benchmark requires the server-side
model and ImageNet split files. Parquet I/O, model loading, and calibration are
outside GPU-forward timing.

## Reproducibility boundaries

- Model checkpoints, model weights, and parquet data are not included.
- DistilBERT uses the locally supplied model and the GLUE/SST-2 validation
  split with 872 examples.
- DeiT results use the stated 40K/5K project split and should be reported as a
  pilot/subset protocol, not as full ImageNet-1K validation.
- GPU-forward timing, batch-one latency, batch-32 throughput, and
  input-inclusive end-to-end serving time are different metrics. This artifact
  does not establish a full serving speedup including data loading.
- The experiments do not establish that route collapse is solved. The DeiT
  evidence supports only conditional mitigation under the reported fixed-
  adapter comparisons.
- The original ASTER paper reports additional ViT-Base/ImageNet and
  Llama-3.1-8B/BoolQ configurations. Corresponding implementations were not
  available in the inspected code snapshot and are not claimed here.

## Running experiments

The shell scripts document the server paths and expected checkpoint arguments.
Before running them, edit the root, model, split, and checkpoint paths for the
local environment. The required data and weights must be supplied separately.

The saved CSV and Markdown reports are evidence snapshots from the completed
runs; they are not model checkpoints.

## Citation

If you use this artifact, cite the accompanying paper and reference the
anonymous repository URL supplied with the paper submission.
