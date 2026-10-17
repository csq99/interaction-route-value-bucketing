# Quality-aware load-balanced Interaction Router diagnostic

This directory contains two independent DeiT experiments. Neither experiment
overwrites an existing result or checkpoint.

1. `run_fixed_rng_selection.sh` repeats the saved epoch-15 Adapter / epoch-1
   Router selection-rule diagnostic. Each stochastic stream owns one persistent
   generator across all batches, fixing the earlier per-batch seed reset.
2. `run_load_balanced_router.sh` freezes the epoch-15 Adapter and trains three
   fresh Interaction Routers for one epoch on the same 40,000-row training
   split. The four Routers share every expensive route-loss computation and
   differ only in load-balance weight (`0.0`, `0.01`, `0.1`, `1.0`). They start
   from identical weights; `0.0` is the within-run control.

The Router v2 objective is:

```text
listwise route-value loss
+ lambda * KL(quality-aware load prior || sharp top-1 load proxy)
```

The prior is 75% the batch-average CE+logit-KD route target and 25% uniform.
The top-1 proxy z-normalizes each sample's route scores and uses temperature
0.1. This targets the hard-argmax concentration observed in prior diagnostics;
ordinary softmax entropy is already high and is not used as the balance signal.

All five Routers (the saved reference plus four same-run variants) are evaluated by
deterministic argmax on the same first 1,024 test rows for 4/12, 6/12 and 7/12.
The experiment records accuracy, selected-route count and maximum route share.
It does not measure latency. The 40k/5k/5k validation-derived split is not the
full ImageNet training protocol.

## Upload

Upload this whole directory to:

```text
/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-load-balanced-router-run
```

Required files:

```text
train_load_balanced_router.py
diagnose_selection_rules_fixed_rng.py
run_load_balanced_router.sh
run_fixed_rng_selection.sh
submit_all.sh
README.md
```

Do not upload model data or checkpoints. Both jobs load the existing server
model, split CSVs, epoch-15 Adapter and epoch-1 reference Router.

## Submit

```bash
cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-load-balanced-router-run
mkdir -p logs
sed -i 's/\r$//' *.sh
chmod +x *.sh
bash submit_all.sh
```

The two jobs are independent and may run concurrently. To submit only Router
v2, use `sbatch run_load_balanced_router.sh`. New results are rejected if their
target directory is already non-empty.
