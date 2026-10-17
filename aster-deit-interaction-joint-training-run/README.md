# Joint Adapter and Interaction Router Training

This isolated experiment continues the epoch-15 Adapter and interaction Router checkpoints with both trainable at the same time. The DeiT backbone stays frozen. On each batch, the Adapter receives one uniformly sampled complete route and is trained with subsequent-layer hidden-state KD, final classification CE, and final-logit KD. The Router receives detached CE+logit-KD route-quality labels for sampled candidate routes and is trained with a listwise objective. Adapter route sampling is independent of the Router's current top route to preserve coverage.

The first run uses 40,000 training samples, 1,024 evaluation samples, budgets 4/12, 6/12, and 7/12, and three joint epochs. This is a pilot on the existing validation-derived code split, not the full ImageNet protocol. It records the starting checkpoint as epoch 0 and evaluates after every epoch on the same fixed sample set. Each epoch writes a joint checkpoint containing both models and both optimizer states.

## Upload to the server

Upload this entire new directory:

```text
aster-deit-interaction-joint-training-run/
  README.md
  train_joint_interaction_deit.py
  run_joint_interaction_deit.sh
```

The run also requires the existing server files and checkpoints under `aster-deit-interaction-route-value-run`, `aster-deit-interaction-multibudget-40k-e10-to-e15-run`, `aster-deit-code-split-run`, and `models/deit-base-patch16-224`.

## Submit on the server

```bash
cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-interaction-joint-training-run
sed -i 's/\r$//' run_joint_interaction_deit.sh
chmod +x run_joint_interaction_deit.sh
sbatch run_joint_interaction_deit.sh
```

Check progress and logs with:

```bash
squeue -u "$USER"
tail -f logs/joint_interaction_<jobid>.out
tail -f logs/joint_interaction_<jobid>.err
```

Replace `<jobid>` with the ID printed by `sbatch`. Accuracy and route-use curves are written under `results_joint_e3/`; resumable checkpoints are written under `checkpoints_joint_e3/`.
