$ErrorActionPreference = 'Stop'
$root = 'E:\pycharm\ASTER\ASTER-main'
$python = 'C:\Users\79867\.conda\envs\aster\python.exe'
$run = Join-Path $root 'aster-baseline-methods-comparison-run'
$checkpoint = Join-Path $root 'aster-bert-interaction-multibudget-run\checkpoints_shared_2to5\shared_adapter_epoch_003.pt'
$output = Join-Path $run 'results_distilbert_full_validation'

& $python (Join-Path $run 'benchmark_distilbert_five_methods.py') `
  --max_eval_samples -1 `
  --batch_size 16 `
  --throughput_batch_size 32 `
  --calibration_samples 128 `
  --target_layers 4 `
  --adapter_checkpoint $checkpoint `
  --output_dir $output
