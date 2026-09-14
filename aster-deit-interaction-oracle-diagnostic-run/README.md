# Epoch-10 Router-vs-Oracle Diagnostic

This independent run does not train or overwrite the epoch-10 interaction checkpoint.

For a fixed, deterministic validation subset, it enumerates every legal route at each configured budget. It compares the interaction router's selected route against the lowest CE-plus-logit-KD-loss route for the same image, and records fixed-route quality across the subset.

The default server script diagnoses the first 128 rows of the existing 5k test split. Its outputs are diagnostic-only and must not be presented as 5k evaluation results.
