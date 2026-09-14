"""Inference-only latency and throughput benchmark for interaction route-value BERT."""

import argparse
import csv
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader

import train_interaction_route_value_bert as impl


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_components(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=impl.config.DEVICE, weights_only=False)
    args = checkpoint["args"]
    impl.config.MODEL_ID = impl.resolve_model_id(args["model_id"])
    model, tokenizer = impl.load_model_and_tokenizer()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    num_layers = len(model.distilbert.transformer.layer)
    target_layers = checkpoint["target_layers"]
    legal_routes = impl.complete_routes(num_layers, target_layers)
    candidate_layers = checkpoint["candidate_layers"]
    route_positions = {layer: index for index, layer in enumerate(candidate_layers)}
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(
        model.config.hidden_size, args["adapter_bottleneck_dim"], num_layers, dtype,
    ).to(impl.config.DEVICE)
    router = impl.InteractionRouteValueRouter(
        model.config.hidden_size, len(candidate_layers), args["router_hidden_dim"], dtype,
    ).to(impl.config.DEVICE)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    router.load_state_dict(checkpoint["router_state_dict"])
    adapter.eval()
    router.eval()
    _, pair_mask = impl.pair_masks(candidate_layers, legal_routes, impl.config.DEVICE, dtype)
    route_position_tensor = torch.tensor(
        [[route_positions[layer] for layer in route[1:-1]] for route in legal_routes],
        device=impl.config.DEVICE,
        dtype=torch.long,
    )
    return model, tokenizer, adapter, router, legal_routes, pair_mask, route_position_tensor


def select_routes(model, router, legal_routes, pair_mask, route_positions, input_ids, attention_mask):
    hidden, layer_attention_mask = impl.prepare_layer0(model, input_ids, attention_mask)
    unary, pairwise = router(hidden)
    selected_indices = impl.score_routes(unary, pairwise, route_positions, pair_mask).argmax(dim=1)
    routes = [legal_routes[index] for index in selected_indices.detach().cpu().tolist()]
    return hidden, layer_attention_mask, routes


def run_interaction_per_sample(model, adapter, router, legal_routes, pair_mask, route_positions,
                               input_ids, attention_mask):
    hidden, layer_attention_mask, routes = select_routes(
        model, router, legal_routes, pair_mask, route_positions, input_ids, attention_mask,
    )
    states = []
    for index, route in enumerate(routes):
        states.append(impl.execute_route(
            model, adapter, hidden[index:index + 1], layer_attention_mask[index:index + 1], route,
        ))
    return torch.cat(states, dim=0), routes


def run_interaction_bucketed(model, adapter, router, legal_routes, pair_mask, route_positions,
                             input_ids, attention_mask):
    hidden, layer_attention_mask, routes = select_routes(
        model, router, legal_routes, pair_mask, route_positions, input_ids, attention_mask,
    )
    grouped = defaultdict(list)
    for index, route in enumerate(routes):
        grouped[tuple(route)].append(index)
    final_hidden = torch.empty_like(hidden)
    for route, indices in grouped.items():
        index_tensor = torch.tensor(indices, device=hidden.device, dtype=torch.long)
        grouped_hidden = impl.execute_route(
            model,
            adapter,
            hidden.index_select(0, index_tensor),
            layer_attention_mask.index_select(0, index_tensor),
            list(route),
        )
        final_hidden.index_copy_(0, index_tensor, grouped_hidden)
    return final_hidden, routes


def evaluate(method, loader, model, adapter, router, legal_routes, pair_mask, route_positions):
    correct = total = 0
    predictions = []
    route_counter = Counter()
    sync_if_cuda()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(impl.config.DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(impl.config.DEVICE, non_blocking=True)
            labels = batch["labels"].to(impl.config.DEVICE, non_blocking=True)
            if method == "full":
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            elif method == "per_sample":
                final_hidden, routes = run_interaction_per_sample(
                    model, adapter, router, legal_routes, pair_mask, route_positions,
                    input_ids, attention_mask,
                )
                route_counter.update(tuple(route) for route in routes)
                logits = impl.logits_from_hidden(model, final_hidden)
            elif method == "bucketed":
                final_hidden, routes = run_interaction_bucketed(
                    model, adapter, router, legal_routes, pair_mask, route_positions,
                    input_ids, attention_mask,
                )
                route_counter.update(tuple(route) for route in routes)
                logits = impl.logits_from_hidden(model, final_hidden)
            else:
                raise ValueError(f"Unknown benchmark method: {method}")
            predicted = logits.argmax(dim=-1)
            correct += int(predicted.eq(labels).sum())
            total += labels.numel()
            predictions.extend(predicted.detach().cpu().tolist())
    sync_if_cuda()
    seconds = time.perf_counter() - start
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "seconds": seconds,
        "samples_per_second": total / seconds,
        "milliseconds_per_sample": 1000 * seconds / total,
        "predictions": predictions,
        "routes": route_counter,
    }


def run_condition(batch_size, args, components, eval_dataset):
    model, _, adapter, router, legal_routes, pair_mask, route_positions = components
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    # A batch of one has no route grouping opportunity. Benchmark only the
    # actual per-request dynamic path there; compare bucketing at batch N.
    methods = ("full", "per_sample") if batch_size == 1 else ("full", "per_sample", "bucketed")
    for _ in range(args.warmup):
        for method in methods:
            evaluate(method, loader, model, adapter, router, legal_routes, pair_mask, route_positions)
    rows = {method: [] for method in methods}
    for repeat in range(1, args.repeats + 1):
        for method in methods:
            row = evaluate(method, loader, model, adapter, router, legal_routes, pair_mask, route_positions)
            rows[method].append(row)
            print(
                f"batch={batch_size} repeat={repeat} method={method} "
                f"accuracy={row['accuracy']:.4f} time={row['seconds']:.4f}s",
                flush=True,
            )
    if "bucketed" in rows:
        for index in range(args.repeats):
            if rows["per_sample"][index]["predictions"] != rows["bucketed"][index]["predictions"]:
                raise RuntimeError("Route bucketing changed predictions relative to per-sample dispatch.")
    return rows


def median_row(rows):
    median_seconds = statistics.median(row["seconds"] for row in rows)
    selected = min(rows, key=lambda row: abs(row["seconds"] - median_seconds))
    return selected | {"seconds": median_seconds,
                       "samples_per_second": selected["total"] / median_seconds,
                       "milliseconds_per_sample": 1000 * median_seconds / selected["total"]}


def summary_rows(condition_rows):
    full = median_row(condition_rows["full"])
    rows = []
    methods = (
        ("full", "Full DistilBERT"),
        ("per_sample", "Interaction router, per-sample dispatch"),
    )
    if "bucketed" in condition_rows:
        methods += (("bucketed", "Interaction router, route-bucketed dispatch"),)
    for method, label in methods:
        row = median_row(condition_rows[method])
        rows.append({
            "method": label,
            "accuracy": row["accuracy"],
            "correct": row["correct"],
            "total": row["total"],
            "seconds": row["seconds"],
            "samples_per_second": row["samples_per_second"],
            "milliseconds_per_sample": row["milliseconds_per_sample"],
            "speedup_vs_full": full["seconds"] / row["seconds"],
        })
    return rows


def write_report(path, args, latency_rows, throughput_rows, batch_one_routes, batch_n_routes):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DistilBERT Interaction Route-Value Inference Benchmark\n\n")
        handle.write("This report times inference only. The model, checkpoint, cached SST-2 dataset, "
                     "and tokenized tensors are prepared before timing. Timed loops include device transfer, "
                     "layer execution, adapter execution, route-value scoring, and classification.\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Checkpoint: `{args.checkpoint}`\n")
        handle.write("- Dataset: cached GLUE/SST-2 validation\n")
        handle.write(f"- Evaluated samples: `{latency_rows[0]['total']}`\n")
        handle.write(f"- Timed repeats: `{args.repeats}`; warm-up passes: `{args.warmup}`\n")
        handle.write("- Target: `4 / 6` layers; interaction router scores six legal complete routes.\n\n")
        handle.write(f"- PyTorch: `{torch.__version__}`; CUDA runtime: `{torch.version.cuda}`\n")
        if torch.cuda.is_available():
            handle.write(f"- GPU: `{torch.cuda.get_device_name(torch.cuda.current_device())}`\n\n")
        else:
            handle.write("- Device: `CPU`\n\n")
        handle.write("## Batch-1 Latency\n\n")
        handle.write("| Method | Accuracy | Time (s) | ms/sample | Speedup vs Full |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in latency_rows:
            handle.write(f"| {row['method']} | {row['accuracy']:.4f} | {row['seconds']:.4f} | "
                         f"{row['milliseconds_per_sample']:.3f} | {row['speedup_vs_full']:.4f}x |\n")
        handle.write("\n## Batch-N Throughput\n\n")
        handle.write(f"Batch size: `{args.throughput_batch_size}`\n\n")
        handle.write("| Method | Accuracy | Time (s) | Samples/s | Speedup vs Full |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in throughput_rows:
            handle.write(f"| {row['method']} | {row['accuracy']:.4f} | {row['seconds']:.4f} | "
                         f"{row['samples_per_second']:.2f} | {row['speedup_vs_full']:.4f}x |\n")
        handle.write("\n## Route Distribution\n\n")
        handle.write("Batch-1 dynamic route counts:\n\n")
        for route, count in batch_one_routes.most_common():
            handle.write(f"- `{list(route)}`: {count}\n")
        handle.write("\nBatch-N route-bucketed counts:\n\n")
        for route, count in batch_n_routes.most_common():
            handle.write(f"- `{list(route)}`: {count}\n")
        handle.write("\n## I/O Boundary\n\n")
        handle.write("- Parquet I/O: not applicable. This DistilBERT SST-2 experiment reads Hugging Face cached "
                     "Arrow data, not ImageNet parquet shards.\n")
        handle.write("- Model loading, dataset cache discovery/loading, and tokenization are excluded from both tables. "
                     "They must not be interpreted as inference latency or throughput.\n")
        handle.write("- Batch-1 latency and batch-N throughput are separate metrics and are not interchangeable.\n")


def parse_args():
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(default_dir / "checkpoints_4of6_coverage3" / "interaction_route_value_router.pt"),
    )
    parser.add_argument("--throughput_batch_size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument(
        "--output_dir",
        default=str(default_dir / "results_4of6_coverage3" / "inference_benchmark"),
    )
    return parser.parse_args()


def main(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components = load_components(checkpoint_path)
    _, tokenizer, _, _, _, _, _ = components
    dataset = impl.bert_impl.load_sst2(tokenizer)["validation"]
    if args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    batch_one = run_condition(1, args, components, dataset)
    batch_n = run_condition(args.throughput_batch_size, args, components, dataset)
    latency_rows = summary_rows(batch_one)
    throughput_rows = summary_rows(batch_n)
    write_rows = []
    for metric, rows in (("batch_1_latency", latency_rows), ("batch_n_throughput", throughput_rows)):
        for row in rows:
            write_rows.append({"metric": metric} | row)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=write_rows[0].keys())
        writer.writeheader()
        writer.writerows(write_rows)
    report = output_dir / "interaction_route_value_inference.md"
    write_report(
        report,
        args,
        latency_rows,
        throughput_rows,
        median_row(batch_one["per_sample"])["routes"],
        median_row(batch_n["bucketed"])["routes"],
    )
    print(f"Saved benchmark report: {report}", flush=True)


if __name__ == "__main__":
    main(parse_args())
