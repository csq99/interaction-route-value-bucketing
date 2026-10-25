import argparse
import csv
import itertools
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EARLY_DIR = HERE.parent / "aster-deit-early-cls-bce-run"
ROUTE_DIR = HERE.parent / "aster-deit-interaction-route-value-run"
sys.path.insert(0, str(ROUTE_DIR))
import train_interaction_route_value_deit as impl
sys.path.insert(0, str(EARLY_DIR))
import config
from components import DynamicAdapter


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def layer_hidden(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def logits(model, state):
    return model.classifier(state[:, 0])


def route_list(num_layers, target):
    candidates = list(range(1, num_layers - 1))
    return [[0, *middle, num_layers - 1]
            for middle in itertools.combinations(candidates, target - 2)]


def execute_route(model, layers, adapter, hidden0, route):
    state = hidden0
    previous = 0
    for layer_index in route[1:]:
        if layer_index > previous + 1 and adapter is not None:
            state = adapter(state, previous, layer_index)
        state = layer_hidden(layers[layer_index](state))
        previous = layer_index
    return state


def execute_grouped(model, layers, adapter, hidden0, routes):
    groups = defaultdict(list)
    for i, route in enumerate(routes): groups[tuple(route)].append(i)
    out = torch.empty_like(hidden0)
    for route, indices in groups.items():
        ix = torch.tensor(indices, device=hidden0.device)
        state = execute_route(model, layers, adapter, hidden0.index_select(0, ix), list(route))
        out.index_copy_(0, ix, state)
    return out


def load_adapter(model, checkpoint_path, dtype):
    if not checkpoint_path:
        return None
    state = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    args = state.get("args", {})
    bottleneck = int(args.get("adapter_bottleneck_dim", 128))
    adapter = DynamicAdapter(model.config.hidden_size, bottleneck, 12, dtype).to(config.DEVICE)
    adapter.load_state_dict(state["adapter_state_dict"])
    adapter.eval()
    return adapter


@torch.no_grad()
def calibration_scores(model, layers, loader, limit):
    sums = torch.zeros(len(layers) - 1, device=config.DEVICE)
    count = 0
    for images, _ in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        out = model(images, output_hidden_states=True)
        hs = out.hidden_states
        for i in range(1, len(layers)):
            a, b = hs[i - 1], hs[i]
            sums[i - 1] += F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).sum()
        count += images.size(0)
        if limit > 0 and count >= limit: break
    return (sums / max(1, count)).detach().cpu().numpy()


def shortgpt_route(num_layers, target, redundancy):
    candidates = list(range(1, num_layers - 1))
    # Keep layers with the largest change, i.e. lowest predecessor cosine.
    keep = sorted(candidates, key=lambda x: float(redundancy[x - 1]))[:target - 2]
    return [0, *sorted(keep), num_layers - 1]


def uniform_route(num_layers, target):
    candidates = list(range(1, num_layers - 1))
    if target <= 2: return [0, num_layers - 1]
    positions = np.linspace(0, len(candidates) - 1, target - 2)
    keep = sorted({candidates[int(round(x))] for x in positions})
    for c in candidates:
        if len(keep) >= target - 2: break
        if c not in keep: keep.append(c)
    return [0, *sorted(keep[:target - 2]), num_layers - 1]


@torch.no_grad()
def early_exit_batch(model, layers, images, threshold, min_layers, max_layers):
    state = layer_hidden(layers[0](impl.run_embeddings(model, images)))
    active = torch.arange(images.size(0), device=images.device)
    routes = [[0] for _ in range(images.size(0))]
    finished = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
    final_states = torch.empty_like(state)
    # max_layers is a hard compute budget for this table column. If no
    # confidence exit fires, return the deepest state allowed by the budget.
    for layer_index in range(1, min(len(layers), max_layers)):
        if active.numel() == 0: break
        current = layer_hidden(layers[layer_index](state.index_select(0, active)))
        state.index_copy_(0, active, current)
        for local, sample in enumerate(active.tolist()): routes[sample].append(layer_index)
        confidence = F.softmax(logits(model, current), dim=-1).max(dim=-1).values
        can_exit = (confidence >= threshold) & (layer_index + 1 >= min_layers)
        done = active[can_exit]
        if done.numel():
            final_states.index_copy_(0, done, state.index_select(0, done))
            finished[done] = True
        active = active[~can_exit]
    if active.numel():
        final_states.index_copy_(0, active, state.index_select(0, active))
        finished[active] = True
    return logits(model, final_states), routes


@torch.no_grad()
def skipdecode_batch(model, layers, adapter, images, threshold, max_layers):
    initial = layer_hidden(layers[0](impl.run_embeddings(model, images)))
    states, routes = [], []
    # Keep a separate current depth per sample; confidence decisions can
    # differ within one batch.
    for i in range(images.size(0)):
        state = initial[i:i + 1]
        previous = 0
        route = [0]
        while previous < len(layers) - 1 and len(route) < max_layers:
            confidence = F.softmax(logits(model, state), dim=-1).max().item()
            jump = 2 if confidence >= threshold and previous + 2 < len(layers) else 1
            # Keep enough room to reach the final layer within max_layers.
            remaining_nodes = max_layers - (len(route) + 1)
            minimum_next = max(previous + 1, len(layers) - 1 - remaining_nodes)
            next_index = min(max(previous + jump, minimum_next), len(layers) - 1)
            if next_index > previous + 1 and adapter is not None:
                state = adapter(state, previous, next_index)
            state = layer_hidden(layers[next_index](state))
            route.append(next_index)
            previous = next_index
        states.append(state)
        routes.append(route)
    return logits(model, torch.cat(states, dim=0)), routes


def evaluate_method(method, model, layers, adapter, batches, target, static_route,
                    early_threshold, skip_threshold, batch_size=None):
    correct = total = 0
    executed = 0
    route_counts = defaultdict(int)
    start = time.perf_counter(); sync()
    for images, labels in batches:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        if method == "baseline_full":
            out = model(images).logits; routes = [list(range(len(layers)))] * images.size(0)
        elif method == "random_skip":
            legal = route_list(len(layers), target)
            routes = [random.Random(1000 + total + i).choice(legal)
                      for i in range(images.size(0))]
            hidden0 = layer_hidden(layers[0](impl.run_embeddings(model, images)))
            out = logits(model, execute_grouped(model, layers, adapter, hidden0, routes))
        elif method == "shortgpt_proxy":
            routes = [static_route] * images.size(0)
            hidden0 = layer_hidden(layers[0](impl.run_embeddings(model, images)))
            out = logits(model, execute_grouped(model, layers, adapter, hidden0, routes))
        elif method == "early_exit":
            out, routes = early_exit_batch(model, layers, images, early_threshold, 2, target)
        elif method == "skipdecode_proxy":
            out, routes = skipdecode_batch(model, layers, adapter, images, skip_threshold, target)
        else: raise ValueError(method)
        pred = out.argmax(-1)
        correct += int(pred.eq(labels).sum()); total += labels.numel()
        executed += sum(len(route) for route in routes)
        for route in routes: route_counts[tuple(route)] += 1
    sync(); elapsed = time.perf_counter() - start
    result = {"method": method, "accuracy": correct / max(1, total),
              "samples": total, "seconds": elapsed,
              "avg_executed_layers": executed / max(1, total),
              "samples_per_second": total / max(elapsed, 1e-9),
              "routes": dict(route_counts)}
    if batch_size is not None:
        result["batch_size"] = batch_size
    return result


def run_condition(methods, model, layers, adapter, dataset, batch_size, args, static_route):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    batches = [(x, y) for x, y in loader]
    rows = []
    for method in methods:
        result = evaluate_method(method, model, layers, adapter, batches, args.target_layers,
                                 static_route, args.early_exit_threshold,
                                 args.skipdecode_threshold, batch_size=batch_size)
        routes = result.pop("routes")
        result["target_layers"] = args.target_layers
        result["static_route"] = str(static_route)
        result["route_count"] = len(routes)
        result["max_route_share"] = max(routes.values()) / result["samples"]
        rows.append(result)
        print(result, flush=True)
    full_seconds = next(r["seconds"] for r in rows if r["method"] == "baseline_full")
    for row in rows:
        row["speedup_vs_full"] = full_seconds / row["seconds"]
    return rows


def main(args):
    set_seed(args.seed); config.MODEL_ID = args.model_id
    model, transform = impl.load_model_and_processor(); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    layers = impl.get_encoder_layers(model); dtype = next(model.parameters()).dtype
    dataset = impl.ImageNetParquetSplitDataset(args.eval_csv, transform=transform,
                                               max_samples=None if args.max_eval_samples <= 0 else args.max_eval_samples)
    calibration_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    calibration = calibration_scores(model, layers, calibration_loader, args.calibration_samples)
    adapter = load_adapter(model, args.adapter_checkpoint, dtype)
    methods = ["baseline_full", "early_exit", "random_skip", "skipdecode_proxy", "shortgpt_proxy"]
    rows = []
    target_specs = []
    for item in args.target_specs.split(","):
        label, target = item.split(":", 1)
        target_specs.append((label, int(target)))
    for target_label, target in target_specs:
        args.target_layers = target
        static_route = shortgpt_route(len(layers), target, calibration)
        print(f"Budget {target_label} ({target}/{len(layers)} layers); ShortGPT route: {static_route}; adapter={'loaded' if adapter else 'identity skip'}", flush=True)
        for batch_size in (1, args.throughput_batch_size):
            condition_rows = run_condition(methods, model, layers, adapter, dataset, batch_size, args, static_route)
            for row in condition_rows:
                row["target_label"] = target_label
                row["target_layers"] = target
                row["target_speedup"] = float(target_label.rstrip("x"))
            rows.extend(condition_rows)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "five_methods_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k not in ("routes",)]); writer.writeheader(); writer.writerows(rows)
    with (out / "five_methods_report.md").open("w", encoding="utf-8") as f:
        f.write("# Five method DeiT comparison\n\n")
        f.write("This is a classification benchmark. SkipDecode and ShortGPT are labelled transfer proxies, not claims of reproducing their decoder-generation algorithms. Early-exit uses the final head at intermediate depths and has no auxiliary exit-head training.\n\n")
        f.write(f"- Samples: `{len(dataset)}`; budgets: `{args.target_specs}`; adapter: `{args.adapter_checkpoint or 'none'}`\n")
        f.write(f"- Early threshold: `{args.early_exit_threshold}`; SkipDecode proxy threshold: `{args.skipdecode_threshold}`\n\n")
        f.write("| Budget | Target layers | Batch | Method | Accuracy | Avg layers actually run | Time (s) | Samples/s | Speedup vs full | Route count | Max route share |\n|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows: f.write(f"| {r['target_label']} | {r['target_layers']} | {r['batch_size']} | {r['method']} | {r['accuracy']:.4f} | {r['avg_executed_layers']:.2f} | {r['seconds']:.4f} | {r['samples_per_second']:.2f} | {r['speedup_vs_full']:.4f}x | {r['route_count']} | {r['max_route_share']:.4f} |\n")
    print(f"Saved {out / 'five_methods_summary.csv'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True); p.add_argument("--eval_csv", required=True)
    p.add_argument("--target_layers", type=int, default=6); p.add_argument("--target_specs", default="1.71x:7,2.0x:6,3.0x:4"); p.add_argument("--max_eval_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16); p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--throughput_batch_size", type=int, default=32)
    p.add_argument("--calibration_samples", type=int, default=128); p.add_argument("--adapter_checkpoint", default="")
    p.add_argument("--early_exit_threshold", type=float, default=0.95); p.add_argument("--skipdecode_threshold", type=float, default=0.85)
    p.add_argument("--output_dir", required=True); p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
