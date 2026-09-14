# BERT Route Bucketing Benchmark

This isolated experiment reuses the completed `Early-CLS + BCE` DistilBERT router
checkpoint. It does not retrain the predictor or adapter.

It compares the same dynamic top-k decisions under two execution schedules:

- Per-sample dispatch: each sample's selected route is executed separately.
- Route-bucketed dispatch: samples that independently select the same route in a
  batch are grouped and executed together.

Run from the repository root:

```powershell
& 'C:\Users\79867\.conda\envs\aster\python.exe' .\aster-bert-route-bucket-run\benchmark_early_cls_route_bucketing.py --batch_size 32
& 'C:\Users\79867\.conda\envs\aster\python.exe' .\aster-bert-route-bucket-run\benchmark_early_cls_route_bucketing.py --batch_size 1 --output route_bucket_benchmark_bce_cls4_bs1.md
```

The report is written into this directory. Only the batch-size-32 run can benefit
from route grouping; batch-size-1 is retained as a single-request latency control.
