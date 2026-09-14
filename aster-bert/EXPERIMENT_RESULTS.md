# ASTER-BERT SST-2 Experiment Results

Date: 2026-05-21

## Experiment Scope

This experiment runs the `aster-bert` part of the repository.

- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Dataset: GLUE / SST-2 validation split
- Validation samples: 872
- Device: NVIDIA GeForce RTX 5060 Laptop GPU
- PyTorch: `2.12.0+cu130`
- Checkpoint: `checkpoints_distilbert_sst2/aster_bert_checkpoint.pt`

This corresponds to the DistilBERT / SST-2 setting in the paper table, but the current code does not expose fixed target acceleration settings such as 1.2x, 1.5x, or 3.0x. The trained greedy policy selected a fixed 2.0x theoretical layer-speedup path.

## Commands Run

```powershell
& 'C:\Users\79867\.conda\envs\aster\python.exe' train_bert.py
& 'C:\Users\79867\.conda\envs\aster\python.exe' evaluate_bert_aster.py
& 'C:\Users\79867\.conda\envs\aster\python.exe' benchmark_bert_aster.py
```

## Accuracy Results

| Method | Batch Size | Accuracy | Correct / Total |
|---|---:|---:|---:|
| Baseline full DistilBERT | 1 | 0.9106 | 794 / 872 |
| Baseline full DistilBERT | 16 | 0.9106 | 794 / 872 |
| ASTER greedy | 1 | 0.8257 | 720 / 872 |

## Layer Skipping Results

| Metric | Value |
|---|---:|
| Total DistilBERT layers | 6 |
| Average executed layers | 3.0000 / 6 |
| Average skipped layers | 3.0000 / 6 |
| Computational savings | 50.00% |
| Theoretical layer speedup | 2.0000x |

Executed layer count distribution:

| Executed Layers | Count | Ratio |
|---:|---:|---:|
| 3 | 872 | 100.00% |

Skipped layer count distribution:

| Skipped Layers | Count | Ratio |
|---:|---:|---:|
| 3 | 872 | 100.00% |

Layer transition distribution:

| Transition | Skipped Layers Between | Count |
|---|---:|---:|
| 0 -> 2 | 1 | 872 |
| 2 -> 5 | 2 | 872 |

Decision path distribution:

| Path | Count | Ratio |
|---|---:|---:|
| 0 -> 2 -> 5 | 872 | 100.00% |

## Runtime Results

Timing only covers the inference loop. It excludes model loading, dataset loading, tokenization, and Hugging Face network checks.

| Method | Batch Size | Inference Time | Samples / Sec | ms / Sample |
|---|---:|---:|---:|---:|
| Baseline full DistilBERT | 1 | 3.6094 s | 241.59 | 4.139 |
| Baseline full DistilBERT | 16 | 1.4167 s | 615.53 | 1.625 |
| ASTER greedy | 1 | 3.5390 s | 246.40 | 4.059 |

Actual runtime speedup:

| Comparison | Actual Speedup |
|---|---:|
| ASTER vs baseline batch_size=1 | 1.0199x |
| ASTER vs baseline batch_size=16 | 0.4003x |

## Summary

The trained ASTER policy achieved a theoretical 2.0x layer speedup by executing only 3 out of 6 DistilBERT layers for every SST-2 validation sample.

However, the measured wall-clock inference speedup is only 1.0199x compared with baseline batch_size=1. Compared with the more efficient baseline batch_size=16 setting, ASTER is slower, with an actual speedup of 0.4003x.

The likely reason is that this implementation performs per-sample dynamic decisions with extra scorer, adapter, Python control flow, and small GPU kernel launches. These overheads largely offset the compute saved by skipping Transformer layers.

## Comparison With Paper Table

Paper table DistilBERT / SST-2 values for Ours:

| Target Speedup | Paper Accuracy |
|---:|---:|
| 1.2x | 88.30% |
| 1.5x | 84.75% |
| 3.0x | 81.65% |

This run:

| Observed Theoretical Speedup | Accuracy |
|---:|---:|
| 2.0x | 82.57% |

The result is not a direct match to a specific table column because the current code does not provide a target-speedup control for 1.2x, 1.5x, or 3.0x. The greedy policy selected the same 2.0x theoretical path for all validation samples.
