# Five baseline methods: DeiT and DistilBERT

This directory contains a common evaluation harness for five methods:

- `baseline_full`: execute every transformer block.
- `early_exit`: stop when the final classification head is confident. This is
  an untrained-head early-exit proxy, not a trained early-exit model.
- `random_skip`: sample a legal fixed-budget route uniformly at random. If an
  adapter checkpoint is supplied, it compensates skipped spans.
- `skipdecode_proxy`: a classification transfer proxy that skips the next
  block when the current classification confidence exceeds a threshold and
  always continues to the final block. SkipDecode was designed for decoder
  generation, so this is not the original generative algorithm.
- `shortgpt_proxy`: a static route selected from a calibration set by keeping
  layers with the largest change from their predecessor. ShortGPT was designed
  for decoder layer pruning, so this is a classification transfer proxy.

The scripts report accuracy, average executed layers, batch=1 latency, and
batch-N throughput separately. Each run can evaluate several target budgets in
one job. DeiT reads the existing parquet split CSVs; parquet I/O is not
included in GPU-forward timing.

## Current status

The DistilBERT comparison has been run locally on all 872 SST-2 validation
examples with the existing 3-epoch adapter checkpoint at 1.2x/1.5x/3.0x. The DeiT comparison is
implemented and syntax-checked, but its five-method benchmark still needs to
be submitted on the server because the local machine does not contain the
DeiT model and ImageNet parquet files.

## DeiT on the server

Upload only these source files into:

```text
/HOME/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-baseline-methods-comparison-run/
```

```text
benchmark_deit_five_methods.py
benchmark_distilbert_five_methods.py
run_deit_five_methods.sh
README.md
```

Then run:

```bash
cd /HOME/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-baseline-methods-comparison-run
sed -i 's/\r$//' run_deit_five_methods.sh
chmod +x run_deit_five_methods.sh
sbatch run_deit_five_methods.sh
```

The default script evaluates all 5000 images in the existing test split at
7/12, 6/12, and 4/12 layers (1.71x, 2.0x, and 3.0x) and loads the existing
3-epoch shared adapter checkpoint. It does not retrain any method.
Results are written to `results_deit/five_methods_summary.csv` and
`results_deit/five_methods_report.md`.

The target-budget syntax is `--target_specs "1.71x:7,2.0x:6,3.0x:4"` for
DeiT and `--target_specs "1.2x:5,1.5x:4,3.0x:2"` for DistilBERT. Random-skip
and ShortGPT use exactly the requested fixed route length. Early-exit and the
SkipDecode classification proxy have the requested layer count as a hard
maximum; their report also records the actual average number of layers.

Batch=1 latency and batch=32 throughput are reported separately. Model loading,
calibration, and parquet I/O are outside the timed loops.

## DistilBERT local validation

Use the existing CUDA Conda environment:

```powershell
$ROOT = 'E:\pycharm\ASTER\ASTER-main'
$PY = 'C:\Users\79867\.conda\envs\aster\python.exe'
& $PY "$ROOT\aster-baseline-methods-comparison-run\benchmark_distilbert_five_methods.py" `
  --max_eval_samples -1 `
  --batch_size 16 `
  --throughput_batch_size 32 `
  --calibration_samples 128 `
  --target_specs "1.2x:5,1.5x:4,3.0x:2" `
  --adapter_checkpoint "$ROOT\aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_adapter_epoch_003.pt" `
  --output_dir "$ROOT\aster-baseline-methods-comparison-run\results_distilbert_full_validation"
```

The completed report is
`results_distilbert_full_validation/five_methods_report.md`.
