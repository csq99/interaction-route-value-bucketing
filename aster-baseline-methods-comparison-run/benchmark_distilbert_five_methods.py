"""Compare five layer-skipping policies on the local DistilBERT/SST-2 setup.

The last two policies are explicitly classification transfer proxies:
SkipDecode was designed for decoder generation and ShortGPT for decoder-layer
pruning.  This script measures their transferred policy, not the original
generative algorithms.
"""

import argparse
import csv
import itertools
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BERT_RUN = ROOT / "aster-bert-interaction-route-value-run"
sys.path.insert(0, str(BERT_RUN))
import train_interaction_route_value_bert as impl

config = impl.config


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def hidden(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logits_from_hidden(model, state):
    return impl.logits_from_hidden(model, state)


def routes_for(num_layers, target):
    middle = target - 2
    if middle < 0 or middle > num_layers - 2:
        raise ValueError(f"target_layers must be in [2, {num_layers}]")
    return [[0, *combo, num_layers - 1]
            for combo in itertools.combinations(range(1, num_layers - 1), middle)]


def load_adapter(model, path):
    if not path:
        return None
    checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=False)
    args = checkpoint.get("args", {})
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(
        model.config.hidden_size,
        int(args.get("adapter_bottleneck_dim", 128)),
        len(model.distilbert.transformer.layer),
        dtype,
    ).to(config.DEVICE)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    return adapter


@torch.no_grad()
def prepare(model, input_ids, attention_mask):
    return impl.prepare_layer0(model, input_ids, attention_mask)


@torch.no_grad()
def execute(model, adapter, state, layer_mask, route):
    return impl.execute_route(model, adapter, state, layer_mask, route)


@torch.no_grad()
def shortgpt_calibration(model, loader, limit):
    layers = model.distilbert.transformer.layer
    # hidden_states includes the embedding output at index 0, followed by
    # one output per transformer layer.
    sums = torch.zeros(len(layers), device=config.DEVICE)
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        hs = output.hidden_states
        for i in range(1, len(hs)):
            sums[i - 1] += F.cosine_similarity(
                hs[i - 1][:, 0, :], hs[i][:, 0, :], dim=-1,
            ).sum()
        count += input_ids.size(0)
        if limit > 0 and count >= limit:
            break
    return (sums / max(count, 1)).detach().cpu().numpy()


def shortgpt_route(num_layers, target, redundancy):
    candidates = list(range(1, num_layers - 1))
    keep = sorted(candidates, key=lambda i: float(redundancy[i - 1]))[:target - 2]
    return [0, *sorted(keep), num_layers - 1]


@torch.no_grad()
def early_exit_batch(model, input_ids, attention_mask, threshold, min_layers, max_layers):
    layers = model.distilbert.transformer.layer
    state, layer_mask = prepare(model, input_ids, attention_mask)
    active = torch.arange(input_ids.size(0), device=config.DEVICE)
    routes = [[0] for _ in range(input_ids.size(0))]
    final = torch.empty_like(state)
    for layer_index in range(1, min(len(layers), max_layers)):
        if active.numel() == 0:
            break
        current = hidden(layers[layer_index](
            state.index_select(0, active), attention_mask=layer_mask.index_select(0, active),
        ))
        state.index_copy_(0, active, current)
        for local, sample in enumerate(active.tolist()):
            routes[sample].append(layer_index)
        confidence = F.softmax(logits_from_hidden(model, current), dim=-1).max(dim=-1).values
        can_exit = (confidence >= threshold) & (layer_index + 1 >= min_layers)
        finished = active[can_exit]
        if finished.numel():
            final.index_copy_(0, finished, state.index_select(0, finished))
        active = active[~can_exit]
    if active.numel():
        final.index_copy_(0, active, state.index_select(0, active))
    return logits_from_hidden(model, final), routes


@torch.no_grad()
def skipdecode_batch(model, adapter, input_ids, attention_mask, threshold, max_layers):
    """Classification proxy: confidence at the current depth may skip one block."""
    layers = model.distilbert.transformer.layer
    initial, layer_mask = prepare(model, input_ids, attention_mask)
    outputs, routes = [], []
    for sample_index in range(input_ids.size(0)):
        state = initial[sample_index:sample_index + 1]
        sample_mask = layer_mask[sample_index:sample_index + 1]
        previous = 0
        route = [0]
        while previous < len(layers) - 1 and len(route) < max_layers:
            confidence = F.softmax(logits_from_hidden(model, state), dim=-1).max().item()
            jump = 2 if confidence >= threshold and previous + 2 < len(layers) else 1
            remaining_nodes = max_layers - (len(route) + 1)
            minimum_next = max(previous + 1, len(layers) - 1 - remaining_nodes)
            next_layer = min(max(previous + jump, minimum_next), len(layers) - 1)
            if next_layer > previous + 1 and adapter is not None:
                state = adapter(state, previous, next_layer)
            state = hidden(layers[next_layer](state, attention_mask=sample_mask))
            route.append(next_layer)
            previous = next_layer
        outputs.append(state)
        routes.append(route)
    return logits_from_hidden(model, torch.cat(outputs, dim=0)), routes


@torch.no_grad()
def evaluate(method, model, adapter, loader, target, static_route,
             early_threshold, skip_threshold):
    layers = model.distilbert.transformer.layer
    legal = routes_for(len(layers), target)
    total = correct = executed = 0
    route_counts = defaultdict(int)
    sync()
    start = time.perf_counter()
    for batch in loader:
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        if method == "baseline_full":
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            routes = [list(range(len(layers)))] * labels.numel()
        elif method == "early_exit":
            outputs, routes = early_exit_batch(
                model, input_ids, attention_mask, early_threshold, min_layers=2,
                max_layers=target,
            )
        elif method in ("random_skip", "shortgpt_proxy"):
            chosen = ([list(static_route) for _ in range(labels.numel())]
                      if method == "shortgpt_proxy" else [
                          random.Random(1000 + total + i).choice(legal)
                          for i in range(labels.numel())
                      ])
            state, layer_mask = prepare(model, input_ids, attention_mask)
            grouped = defaultdict(list)
            for index, route in enumerate(chosen):
                grouped[tuple(route)].append(index)
            final = torch.empty_like(state)
            for route, indices in grouped.items():
                ix = torch.tensor(indices, device=config.DEVICE)
                result = execute(model, adapter, state.index_select(0, ix),
                                 layer_mask.index_select(0, ix), list(route))
                final.index_copy_(0, ix, result)
            outputs, routes = logits_from_hidden(model, final), chosen
        elif method == "skipdecode_proxy":
            outputs, routes = skipdecode_batch(
                model, adapter, input_ids, attention_mask, skip_threshold,
                max_layers=target,
            )
        else:
            raise ValueError(method)
        correct += int(outputs.argmax(-1).eq(labels).sum())
        total += labels.numel()
        executed += sum(len(route) for route in routes)
        for route in routes:
            route_counts[tuple(route)] += 1
    sync()
    seconds = time.perf_counter() - start
    return {
        "method": method,
        "accuracy": correct / max(total, 1),
        "samples": total,
        "seconds": seconds,
        "avg_executed_layers": executed / max(total, 1),
        "samples_per_second": total / max(seconds, 1e-9),
        "routes": dict(route_counts),
    }


def run_condition(args, model, adapter, dataset, batch_size, static_route):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    methods = ["baseline_full", "early_exit", "random_skip", "skipdecode_proxy", "shortgpt_proxy"]
    rows = []
    for method in methods:
        row = evaluate(method, model, adapter, loader, args.target_layers, static_route,
                       args.early_exit_threshold, args.skipdecode_threshold)
        routes = row.pop("routes")
        row["batch_size"] = batch_size
        row["route_count"] = len(routes)
        row["max_route_share"] = max(routes.values()) / row["samples"]
        row["static_route"] = str(static_route)
        rows.append(row)
        print(row, flush=True)
    full_seconds = next(r["seconds"] for r in rows if r["method"] == "baseline_full")
    for row in rows:
        row["speedup_vs_full"] = full_seconds / row["seconds"]
    return rows


def main(args):
    set_seed(args.seed)
    config.MODEL_ID = impl.resolve_model_id(args.model_id)
    model, tokenizer = impl.load_model_and_tokenizer()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    datasets = impl.bert_impl.load_sst2(tokenizer)
    dataset = datasets["validation"]
    if args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    calibration_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    redundancy = shortgpt_calibration(model, calibration_loader, args.calibration_samples)
    num_layers = len(model.distilbert.transformer.layer)
    static_route = shortgpt_route(num_layers, args.target_layers, redundancy)
    adapter = load_adapter(model, args.adapter_checkpoint)
    print(f"ShortGPT proxy route: {static_route}; adapter={'loaded' if adapter else 'none'}", flush=True)
    all_rows = []
    target_specs = []
    for item in args.target_specs.split(","):
        label, target = item.split(":", 1)
        target_specs.append((label, int(target)))
    for target_label, target in target_specs:
        args.target_layers = target
        static_route = shortgpt_route(num_layers, target, redundancy)
        print(f"Budget {target_label} ({target}/{num_layers} layers); ShortGPT route: {static_route}", flush=True)
        for batch_size in [1, args.throughput_batch_size]:
            condition_rows = run_condition(args, model, adapter, dataset, batch_size, static_route)
            for row in condition_rows:
                row["target_label"] = target_label
                row["target_layers"] = target
                row["target_speedup"] = float(target_label.rstrip("x"))
            all_rows.extend(condition_rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in all_rows:
        row.pop("static_route", None)
    with (out / "five_methods_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)
    with (out / "five_methods_report.md").open("w", encoding="utf-8") as handle:
        handle.write("# Five-method DistilBERT/SST-2 comparison\n\n")
        handle.write("Early-exit uses the final classification head at intermediate depth and has no auxiliary exit-head training. "
                     "SkipDecode and ShortGPT are classification transfer proxies, not full reproductions of their decoder-generation methods.\n\n")
        handle.write(f"- Samples: `{len(dataset)}`; budgets: `{args.target_specs}`; adapter: `{args.adapter_checkpoint or 'none'}`\n")
        handle.write(f"- Early threshold: `{args.early_exit_threshold}`; SkipDecode threshold: `{args.skipdecode_threshold}`\n\n")
        handle.write("| Budget | Target layers | Batch | Method | Accuracy | Avg layers actually run | Time (s) | Samples/s | Speedup vs full | Routes | Max route share |\n|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in all_rows:
            handle.write(f"| {row['target_label']} | {row['target_layers']} | {row['batch_size']} | {row['method']} | {row['accuracy']:.4f} | {row['avg_executed_layers']:.2f} | {row['seconds']:.4f} | {row['samples_per_second']:.2f} | {row['speedup_vs_full']:.4f}x | {row['route_count']} | {row['max_route_share']:.4f} |\n")
        handle.write("\nModel loading, tokenization, and dataset cache discovery are excluded from timed loops. Batch=1 latency and batch-N throughput are separate metrics.\n")
    print(f"Saved {out / 'five_methods_summary.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--target_layers", type=int, default=4)
    parser.add_argument("--target_specs", default="1.2x:5,1.5x:4,3.0x:2")
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--throughput_batch_size", type=int, default=32)
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--adapter_checkpoint", default="")
    parser.add_argument("--early_exit_threshold", type=float, default=0.95)
    parser.add_argument("--skipdecode_threshold", type=float, default=0.85)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
