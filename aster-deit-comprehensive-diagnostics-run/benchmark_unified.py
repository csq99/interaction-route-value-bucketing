"""Unified batch-1, batch-N, route-bucketed, and parquet end-to-end benchmark."""

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aster-deit-interaction-multibudget-run"))
import train_interaction_multibudget_deit as train

impl = train.impl


def load(args):
    checkpoint = torch.load(args.checkpoint, map_location=impl.config.DEVICE, weights_only=False)
    saved = checkpoint["args"]
    impl.config.MODEL_ID = args.model_id
    model, transform = impl.load_model_and_processor()
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model)
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(model.config.hidden_size, saved["adapter_bottleneck_dim"], len(layers), dtype).to(impl.config.DEVICE)
    adapter.load_state_dict(checkpoint["adapter_state_dict"]); adapter.eval()
    router = impl.InteractionRouteValueRouter(model.config.hidden_size, len(checkpoint["candidate_layers"]), saved["router_hidden_dim"], dtype).to(impl.config.DEVICE)
    router.load_state_dict(checkpoint["router_state_dict"]); router.eval()
    return checkpoint, model, transform, layers, adapter, router


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


@torch.inference_mode()
def dynamic_batch(model, layers, adapter, router, images, candidates, routes, pair_mask, bucketed):
    hidden0 = impl.layer_hidden_state(layers[0](impl.run_embeddings(model, images)))
    unary, pairwise = router(hidden0)
    chosen = routes.index_select(0, impl.score_routes(unary, pairwise, routes, pair_mask).argmax(1))
    if bucketed:
        state = train.execute_grouped_routes(layers, adapter, hidden0, chosen, candidates)
    else:
        state = torch.cat([
            train.execute_route(layers, adapter, hidden0[i:i + 1], chosen[i], candidates)
            for i in range(images.size(0))
        ], dim=0)
    return impl.logits_from_hidden(model, state)


def gpu_run(method, batches, model, layers, adapter, router, candidates, routes, pair_mask):
    sync(); start = time.perf_counter()
    for images, _ in batches:
        if method == "full":
            model(images)
        else:
            dynamic_batch(model, layers, adapter, router, images, candidates, routes, pair_mask, method == "bucketed")
    sync(); return time.perf_counter() - start


def e2e_run(method, loader, model, layers, adapter, router, candidates, routes, pair_mask):
    # The timer intentionally starts before DataLoader iteration: parquet read, decode,
    # transform, host-to-device copy, and model execution are all included.
    sync(); start = time.perf_counter(); count = 0
    for images, labels in tqdm(loader, desc=f"e2e {method}", leave=False):
        images, labels = images.to(impl.config.DEVICE), labels.to(impl.config.DEVICE)
        if method == "full":
            model(images)
        else:
            dynamic_batch(model, layers, adapter, router, images, candidates, routes, pair_mask, method == "bucketed")
        count += labels.numel()
    sync(); return time.perf_counter() - start, count


def median_call(fn, warmup, repeats):
    for _ in range(warmup): fn()
    values = [fn() for _ in range(repeats)]
    return min(values, key=lambda x: abs(x - statistics.median(values)))


def main(args):
    checkpoint, model, transform, layers, adapter, router = load(args)
    candidates = checkpoint["candidate_layers"]
    all_rows = []
    eval_loader, count = train.make_loader(args.eval_csv, transform, args.batch_size, args.max_eval_samples, args.num_workers, False)
    gpu_cache = [(x.to(impl.config.DEVICE), y.to(impl.config.DEVICE)) for x, y in eval_loader]
    for target in args.target_layers:
        routes = impl.all_middle_routes(len(candidates), target - 2, impl.config.DEVICE)
        pair_mask = impl.pair_masks(len(candidates), routes, next(model.parameters()).dtype)[1]
        for batch_size, metric in ((1, "gpu_forward_batch_1_latency"), (args.batch_size, "gpu_forward_batch_n_throughput")):
            batches = [(x[i:i + batch_size], y[i:i + batch_size]) for x, y in gpu_cache for i in range(0, x.size(0), batch_size)]
            full = median_call(lambda: gpu_run("full", batches, model, layers, adapter, router, candidates, routes, pair_mask), args.warmup, args.repeats)
            per_sample = median_call(lambda: gpu_run("per_sample", batches, model, layers, adapter, router, candidates, routes, pair_mask), args.warmup, args.repeats)
            values = [("Full DeiT", full), ("Interaction per-sample", per_sample)]
            if batch_size > 1:
                bucketed = median_call(lambda: gpu_run("bucketed", batches, model, layers, adapter, router, candidates, routes, pair_mask), args.warmup, args.repeats)
                values.append(("Interaction route-bucketed", bucketed))
            for name, seconds in values:
                all_rows.append({"scope": "gpu_forward_only", "metric": metric, "target_layers": target, "method": name,
                                 "samples": count, "seconds": seconds, "milliseconds_per_sample": 1000 * seconds / count,
                                 "samples_per_second": count / seconds, "speedup_vs_full": full / seconds})
        for method in ("full", "bucketed"):
            def run_e2e(method=method):
                loader, _ = train.make_loader(args.eval_csv, transform, args.batch_size, args.max_eval_samples, args.num_workers, False)
                return e2e_run(method, loader, model, layers, adapter, router, candidates, routes, pair_mask)[0]
            seconds = median_call(run_e2e, 0, args.e2e_repeats)
            all_rows.append({"scope": "parquet_end_to_end", "metric": "parquet_e2e_batch_n", "target_layers": target,
                             "method": "Full DeiT" if method == "full" else "Interaction route-bucketed", "samples": count,
                             "seconds": seconds, "milliseconds_per_sample": 1000 * seconds / count,
                             "samples_per_second": count / seconds, "speedup_vs_full": "pending"})
        full_e2e = [r["seconds"] for r in all_rows if r["scope"] == "parquet_end_to_end" and r["target_layers"] == target and r["method"] == "Full DeiT"][-1]
        for row in all_rows:
            if row["scope"] == "parquet_end_to_end" and row["target_layers"] == target:
                row["speedup_vs_full"] = full_e2e / row["seconds"]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "unified_timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys()); writer.writeheader(); writer.writerows(all_rows)
    print(f"Saved unified timing: {output / 'unified_timing_summary.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True); parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--target_layers", type=int, nargs="+", default=[4, 6, 7])
    parser.add_argument("--batch_size", type=int, default=32); parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=1); parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3); parser.add_argument("--e2e_repeats", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
