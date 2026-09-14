# DistilBERT Paper-Parameter Sweep

This isolated experiment compares the existing target-controlled original ASTER MDP checkpoints with the current Early-CLS + multi-label BCE dynamic top-k router.

All new-router runs use the original ASTER-BERT hyperparameters requested for the comparison:

- train batch size 16
- gradient accumulation 4
- learning rate 5e-5
- 1 offline predictor-pretraining epoch and 3 adapter-task-training epochs
- maximum sequence length 128

Run all paper targets locally:

```powershell
& 'C:\Users\79867\.conda\envs\aster\python.exe' .\aster-bert-paper-params-sweep\run_paper_parameter_sweep.py
```

Outputs are written below `results/`; command logs are written below `logs/`. The runner does not overwrite prior ASTER or router checkpoints.
