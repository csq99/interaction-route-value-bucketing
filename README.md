# Interaction Route-Value Bucketing

This repository contains the code and evidence for an interaction-aware dynamic
layer-skipping study on DistilBERT/SST-2 and DeiT/ImageNet. The repository has
two deliberately separate evidence tracks:

- DistilBERT/SST-2 is the main positive runtime/accuracy study. It includes
  Early-CLS + multi-label BCE one-shot routing, route bucketing, original ASTER
  comparisons, and interaction-aware router extensions.
- DeiT/ImageNet is a controlled 40K/5K protocol with pilot and diagnostic
  results. It is included to document transfer limitations, not as a complete
  ImageNet reproduction or a successful vision benchmark.

The interaction-aware pipeline is:

1. train one shared `DynamicAdapter` with randomly sampled complete routes;
2. train a one-shot route-value router with unary and pairwise layer terms;
3. score legal routes from one early hidden-state summary and optionally group
   samples by route (`route bucketing`) before executing the transformer layers.

The router is trained after the adapter. Adapter quality is therefore measured
separately from routing quality; the router checkpoint contains both states only
to make a complete evaluation reproducible.

## Scope and evidence

The published DeiT experiment uses the project's 40,000-train/5,000-test
code-split protocol. This is not a claim of a complete ImageNet-1K reproduction.
The DistilBERT experiments use the GLUE/SST-2 validation split (872 examples)
with the model and data supplied locally by the user.
The included diagnostic is a deterministic 128-image subset and must not be
reported as a 5,000-image result.

Timing files contain GPU-forward measurements only. Batch-1 latency,
batch-N throughput, and parquet end-to-end I/O are distinct metrics; no
end-to-end parquet timing is claimed by the included forward-only benchmark.

The `results/` directory contains compact summaries copied from the local
experiment logs. They are evidence snapshots, not model checkpoints.

## Layout

- `aster-bert/`: selected DistilBERT/ASTER components and benchmark entry
  points used by the reported BERT experiments.
- `aster-bert-paper-params-sweep/`: the three target-budget Early-CLS+BCE
  comparisons and their formal reports.
- `aster-bert-route-bucket-run/`: batch-1 control and batch-32 route-bucketing
  benchmark for the Early-CLS+BCE router.
- `aster-bert-interaction-route-value-run/`: fixed-budget DistilBERT
  interaction-aware route-value experiment and timing reports.
- `aster-bert-interaction-multibudget-run/`: shared DistilBERT adapter/router
  across multiple budgets and timing reports.
- `aster-bert-random-route-adapter-ablation/`: DistilBERT adapter route-sampling
  ablation and its formal CSV/Markdown summaries.
- `aster-baseline-methods-comparison-run/`: five-method classification
  comparison harness and compact DistilBERT/DeiT evidence tables. The
  SkipDecode and ShortGPT entries are explicitly labelled classification
  transfer proxies.

- `aster-deit-interaction-route-value-run/`: single-budget pilot router.
- `aster-deit-interaction-multibudget-run/`: shared adapter/router across
  4/12, 6/12, and 7/12 budgets, plus the GPU-forward benchmark.
- `aster-deit-interaction-oracle-diagnostic-run/`: exhaustive route-quality
  diagnosis against per-image route oracles.
- `aster-deit-early-cls-bce-run/`: shared DeiT data/model utility modules used
  by the interaction experiments.
- `aster-deit-comprehensive-diagnostics-run/`: fixed-route quality, oracle,
  and GPU-forward versus parquet timing diagnostics.
- `aster-deit-router-collapse-cause-run/`: interaction versus unary-only
  router collapse comparison.
- `aster-deit-load-balanced-router-run/`: load-balancing and route-selection
  rule ablations.
- `aster-deit-interaction-joint-training-run/`: adapter/router joint-training
  diagnostic.
- `aster-deit-interaction-multibudget-40k-e10-to-e15-run/`: extended DeiT
  adapter/router curve and fixed-route quality reports.
- `aster-deit-original-aster-40k-e5-to-e15-run/`: original ASTER subset
  baseline curve.
- `aster-deit-adapter-sampling-ablation-run/`: completed random-route adapter
  sampling evidence; incomplete local-edge results are excluded.
- `results/`: selected accuracy, route-distribution, timing, and diagnostic
  summaries.

## Running on the server

The shell scripts contain the server layout used for these experiments. Before
running, edit the `ROOT`, model, and split paths for your environment, then:

```bash
cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-interaction-multibudget-run
sed -i 's/\r$//' run_shared_multibudget_deit.sh
chmod +x run_shared_multibudget_deit.sh
sbatch run_shared_multibudget_deit.sh
```

The benchmark is intentionally separate from training:

```bash
python benchmark_interaction_multibudget_deit.py \
  --checkpoint checkpoints_shared_4_6_7/shared_interaction_router.pt \
  --eval_csv /path/to/test.csv --target_layers 4 6 7 \
  --batch_size 32 --warmup 1 --repeats 3 \
  --output_dir results_shared_4_6_7/gpu_forward_benchmark
```

Required model weights and parquet data are intentionally not included. They
must be supplied locally or on the server.

## Limitations

The reported runs do not establish that route collapse is solved, nor do they
establish an end-to-end speedup including data loading. Please preserve the
protocol and metric boundaries when extending the experiments.
