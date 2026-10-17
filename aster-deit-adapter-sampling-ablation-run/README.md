# Adapter sampling ablation

This independent experiment compares two adapter-only training schedules from
the same interaction-pilot adapter checkpoint:

1. `random_route`: one randomly sampled legal 2x complete route per batch;
2. `balanced_local_edge`: deterministic balanced sampling of local skip edges,
   with the destination and subsequent real layers executed and hidden-state KD.

The router and DeiT backbone remain frozen. The default run trains through
epoch 10 and evaluates the same fixed routes at epochs 0, 1, 3, 5, and 10.
Use `--eval_epochs` to change those checkpoints without changing the training
horizon.
