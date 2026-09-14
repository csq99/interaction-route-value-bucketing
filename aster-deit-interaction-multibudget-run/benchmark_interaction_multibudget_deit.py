"""Separate DeiT forward-only and parquet end-to-end timing for shared routing."""

import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

import train_interaction_multibudget_deit as train

impl = train.impl


def sync():
    torch.cuda.synchronize()


def load(args):
    checkpoint = torch.load(args.checkpoint, map_location=impl.config.DEVICE, weights_only=False)
    saved_args = checkpoint["args"]
    impl.config.MODEL_ID = saved_args["model_id"]
    model, transform = impl.load_model_and_processor()
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model)
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(model.config.hidden_size, saved_args["adapter_bottleneck_dim"], len(layers), dtype).to(impl.config.DEVICE)
    router = impl.InteractionRouteValueRouter(model.config.hidden_size, len(checkpoint["candidate_layers"]), saved_args["router_hidden_dim"], dtype).to(impl.config.DEVICE)
    adapter.load_state_dict(checkpoint["adapter_state_dict"]); router.load_state_dict(checkpoint["router_state_dict"])
    adapter.eval(); router.eval()
    return checkpoint, model, transform, layers, adapter, router, dtype


def spec(checkpoint, target, dtype):
    candidates = checkpoint["candidate_layers"]
    routes = impl.all_middle_routes(len(candidates), target - 2, impl.config.DEVICE)
    mask = impl.pair_masks(len(candidates), routes, dtype)[1]
    return candidates, routes, mask


@torch.inference_mode()
def dynamic(model, layers, adapter, router, images, candidates, routes, pair_mask, bucketed):
    hidden0 = impl.layer_hidden_state(layers[0](impl.run_embeddings(model, images)))
    unary, pairwise = router(hidden0)
    indexes = impl.score_routes(unary, pairwise, routes, pair_mask).argmax(1)
    selected = routes.index_select(0, indexes)
    if bucketed:
        state = train.execute_grouped_routes(layers, adapter, hidden0, selected, candidates)
    else:
        state = torch.cat([train.execute_route(layers, adapter, hidden0[i:i + 1], selected[i], candidates)
                           for i in range(images.size(0))], 0)
    return impl.logits_from_hidden(model, state), selected


@torch.inference_mode()
def execute(method, batches, components, current_spec):
    _, model, _, layers, adapter, router, _ = components
    candidates, routes, pair_mask = current_spec
    predictions, selected_routes = [], []
    sync(); start = time.perf_counter()
    for images, _ in batches:
        if method == "full": logits = model(images).logits
        else:
            logits, selected = dynamic(model, layers, adapter, router, images, candidates, routes, pair_mask, method == "bucketed")
            selected_routes.extend(tuple(route) for route in selected.cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
    sync(); return time.perf_counter() - start, predictions, selected_routes


def median_timing(method, batches, components, current_spec, warmup, repeats):
    for _ in range(warmup): execute(method, batches, components, current_spec)
    runs = [execute(method, batches, components, current_spec) for _ in range(repeats)]
    return min(runs, key=lambda item: abs(item[0] - statistics.median(run[0] for run in runs)))


def main(args):
    components = load(args)
    checkpoint, _, transform, _, _, _, dtype = components
    targets = args.target_layers or checkpoint["target_layers"]
    dataset_loader, count = train.make_loader(args.eval_csv, transform, args.batch_size, args.max_eval_samples, args.num_workers, False)
    # GPU-forward-only input cache: parquet reads, decoding and transforms happen before timing.
    gpu_batches = [(images.to(impl.config.DEVICE), labels.to(impl.config.DEVICE)) for images, labels in dataset_loader]
    rows = []
    for target in targets:
        current_spec = spec(checkpoint, target, dtype)
        for batch_size, metric in ((1, "gpu_forward_batch_1_latency"), (args.batch_size, "gpu_forward_batch_n_throughput")):
            batches = [(images[i:i + batch_size], labels[i:i + batch_size]) for images, labels in gpu_batches for i in range(0, images.size(0), batch_size)]
            full_t, full_pred, _ = median_timing("full", batches, components, current_spec, args.warmup, args.repeats)
            dynamic_t, dynamic_pred, _ = median_timing("per_sample", batches, components, current_spec, args.warmup, args.repeats)
            if full_pred is None or len(dynamic_pred) != count: raise RuntimeError("Invalid dynamic output length")
            methods = [("Full DeiT", full_t), ("Interaction per-sample", dynamic_t)]
            if batch_size > 1:
                bucket_t, bucket_pred, _ = median_timing("bucketed", batches, components, current_spec, args.warmup, args.repeats)
                if bucket_pred != dynamic_pred: raise RuntimeError("Bucketing changed predictions")
                methods.append(("Interaction route-bucketed", bucket_t))
            for name, seconds in methods:
                rows.append({"metric": metric, "target_layers": target, "method": name, "samples": count,
                             "seconds": seconds, "milliseconds_per_sample": 1000 * seconds / count,
                             "samples_per_second": count / seconds, "speedup_vs_full": full_t / seconds})
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "gpu_forward_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Saved GPU-forward timing: {output / 'gpu_forward_summary.csv'}", flush=True)
    print("Parquet end-to-end timing is intentionally separate and not implemented in this forward-only script.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--target_layers", type=int, nargs="+"); parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_eval_samples", type=int, default=512); parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1); parser.add_argument("--repeats", type=int, default=3); parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
