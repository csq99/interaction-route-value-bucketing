# DeiT Router Collapse Cause Diagnostic

This diagnostic fixes the epoch-15 Adapter and the same 1,024 examples from the
existing 5,000-row test split, then evaluates the saved interaction Router at
epochs 1, 2, and 3. It compares interaction scoring with unary-only scoring.
The output includes accuracy, selected-route count, maximum route share, mean
top-1/top-2 score margin, normalized score entropy, and route usage.

This isolates whether continued Router training changes route concentration or
accuracy under a fixed Adapter. It does not train either component, exhaustively
enumerate route-oracle quality, or measure latency. Existing fixed-route
quality and oracle results should be used as Adapter/path-quality context. The
40k/5k data split is a code-split pilot, not the full ImageNet protocol.

## Upload

Upload only `diagnose_router_collapse.py`, `run_diagnose_router_collapse.sh`,
and this README to:

`/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-router-collapse-cause-run`

Do not upload model, data, or checkpoints; the script uses existing server files.

## Submit

```bash
cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-router-collapse-cause-run
mkdir -p logs
sed -i 's/\r$//' run_diagnose_router_collapse.sh
chmod +x run_diagnose_router_collapse.sh
sbatch run_diagnose_router_collapse.sh
```

The job evaluates 1,024 examples, 3 Router checkpoints, 3 budgets, and 2
scoring modes. Runtime depends on queue and GPU conditions; this is expected to
be substantially cheaper than retraining a Router, but no duration is assumed.
Results are written to `results/epoch_15_adapter_routers_1_2_3/`.
