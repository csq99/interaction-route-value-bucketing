"""Inference benchmark for a shared multi-budget interaction router."""

import argparse
import csv
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train_interaction_multibudget_bert as train

impl = train.impl


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_components(path):
    checkpoint = torch.load(path, map_location=impl.config.DEVICE, weights_only=False)
    args = checkpoint["args"]
    impl.config.MODEL_ID = impl.resolve_model_id(args["model_id"])
    model, tokenizer = impl.load_model_and_tokenizer()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    num_layers = len(model.distilbert.transformer.layer)
    candidates = checkpoint["candidate_layers"]
    positions = {layer: index for index, layer in enumerate(candidates)}
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(model.config.hidden_size, args["adapter_bottleneck_dim"], num_layers, dtype).to(impl.config.DEVICE)
    router = impl.InteractionRouteValueRouter(model.config.hidden_size, len(candidates), args["router_hidden_dim"], dtype).to(impl.config.DEVICE)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    router.load_state_dict(checkpoint["router_state_dict"])
    adapter.eval()
    router.eval()
    return model, tokenizer, adapter, router, candidates, positions, checkpoint["target_layers"]


def target_spec(candidates, positions, target_layers, dtype):
    routes = impl.complete_routes(6, target_layers)
    mask = impl.pair_masks(candidates, routes, impl.config.DEVICE, dtype)[1]
    route_positions = train.route_positions(routes, positions)
    return routes, mask, route_positions


def choose_routes(model, router, routes, pair_mask, route_positions, input_ids, attention_mask):
    hidden, layer_mask = impl.prepare_layer0(model, input_ids, attention_mask)
    unary, pairwise = router(hidden)
    chosen = impl.score_routes(unary, pairwise, route_positions, pair_mask).argmax(dim=1)
    return hidden, layer_mask, [routes[i] for i in chosen.detach().cpu().tolist()]


def dynamic(model, adapter, router, routes, pair_mask, route_positions, input_ids, attention_mask, bucketed):
    hidden, layer_mask, selected_routes = choose_routes(
        model, router, routes, pair_mask, route_positions, input_ids, attention_mask,
    )
    if bucketed:
        states = impl.execute_grouped_routes(model, adapter, hidden, layer_mask, selected_routes)
    else:
        states = torch.cat([
            impl.execute_route(model, adapter, hidden[index:index + 1], layer_mask[index:index + 1], route)
            for index, route in enumerate(selected_routes)
        ], dim=0)
    return impl.logits_from_hidden(model, states), selected_routes


def evaluate(method, loader, components, spec):
    model, _, adapter, router, _, _, _ = components
    routes, mask, positions = spec
    correct = total = 0
    predictions = []
    route_counts = Counter()
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(impl.config.DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(impl.config.DEVICE, non_blocking=True)
            labels = batch["labels"].to(impl.config.DEVICE, non_blocking=True)
            if method == "full":
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                logits, selected = dynamic(
                    model, adapter, router, routes, mask, positions, input_ids, attention_mask,
                    bucketed=method == "bucketed",
                )
                route_counts.update(tuple(route) for route in selected)
            predicted = logits.argmax(dim=-1)
            correct += int(predicted.eq(labels).sum())
            total += labels.numel()
            predictions.extend(predicted.detach().cpu().tolist())
    sync()
    seconds = time.perf_counter() - start
    return {"accuracy": correct / total, "correct": correct, "total": total, "seconds": seconds,
            "samples_per_second": total / seconds, "milliseconds_per_sample": 1000 * seconds / total,
            "predictions": predictions, "routes": route_counts}


def median(rows):
    seconds = statistics.median(row["seconds"] for row in rows)
    source = min(rows, key=lambda row: abs(row["seconds"] - seconds))
    return source | {"seconds": seconds, "samples_per_second": source["total"] / seconds,
                     "milliseconds_per_sample": 1000 * seconds / source["total"]}


def condition(batch_size, args, components, spec, dataset):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    methods = ["full", "per_sample"] if batch_size == 1 else ["full", "per_sample", "bucketed"]
    for _ in range(args.warmup):
        for method in methods:
            evaluate(method, loader, components, spec)
    rows = {method: [] for method in methods}
    for repeat in range(1, args.repeats + 1):
        for method in methods:
            row = evaluate(method, loader, components, spec)
            rows[method].append(row)
            print(f"batch={batch_size} repeat={repeat} method={method} accuracy={row['accuracy']:.4f} time={row['seconds']:.4f}s", flush=True)
    if "bucketed" in rows:
        for index in range(args.repeats):
            if rows["per_sample"][index]["predictions"] != rows["bucketed"][index]["predictions"]:
                raise RuntimeError("Route bucketing changed predictions.")
    return rows


def summarized(metric, target_layers, rows):
    full = median(rows["full"])
    labels = {"full": "Full DistilBERT", "per_sample": "Interaction router, per-sample dispatch",
              "bucketed": "Interaction router, route-bucketed dispatch"}
    output = []
    for method, values in rows.items():
        row = median(values)
        output.append({"metric": metric, "target_layers": target_layers, "theoretical_speedup": 6 / target_layers,
                       "method": labels[method], "accuracy": row["accuracy"], "correct": row["correct"],
                       "total": row["total"], "seconds": row["seconds"],
                       "samples_per_second": row["samples_per_second"],
                       "milliseconds_per_sample": row["milliseconds_per_sample"],
                       "speedup_vs_full": full["seconds"] / row["seconds"]})
    return output


def write_report(path, args, output):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Shared Multi-Budget Interaction Router: Inference Benchmark\n\n")
        handle.write("Timing includes tensor transfer, one-shot route scoring, adapter execution, selected encoder layers, "
                     "and classification. Model loading, dataset loading, and tokenization are outside timed loops.\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Checkpoint: `{args.checkpoint}`\n")
        handle.write(f"- Repeats: `{args.repeats}`; warm-up: `{args.warmup}`\n")
        handle.write(f"- PyTorch: `{torch.__version__}`; CUDA: `{torch.version.cuda}`\n")
        handle.write(f"- GPU: `{torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}`\n\n")
        for metric, title in (("batch_1_latency", "Batch-1 Latency"), ("batch_n_throughput", "Batch-N Throughput")):
            handle.write(f"## {title}\n\n")
            if metric == "batch_1_latency":
                handle.write("| Budget | Method | Accuracy | ms/sample | Speedup vs full |\n|---:|---|---:|---:|---:|\n")
                for row in (item for item in output if item["metric"] == metric):
                    handle.write(f"| {row['target_layers']}/6 | {row['method']} | {row['accuracy']:.4f} | {row['milliseconds_per_sample']:.3f} | {row['speedup_vs_full']:.4f}x |\n")
            else:
                handle.write(f"Batch size: `{args.throughput_batch_size}`\n\n")
                handle.write("| Budget | Method | Accuracy | Samples/s | Speedup vs full |\n|---:|---|---:|---:|---:|\n")
                for row in (item for item in output if item["metric"] == metric):
                    handle.write(f"| {row['target_layers']}/6 | {row['method']} | {row['accuracy']:.4f} | {row['samples_per_second']:.2f} | {row['speedup_vs_full']:.4f}x |\n")
            handle.write("\n")
        handle.write("## I/O Boundary\n\n")
        handle.write("- Parquet I/O: not applicable. SST-2 uses cached Hugging Face Arrow data, not parquet.\n")
        handle.write("- Batch-1 latency and batch-N throughput are separate metrics. Route bucketing is measured only for batch-N.\n")


def main(args):
    components = load_components(args.checkpoint)
    model, tokenizer, _, _, candidates, positions, saved_targets = components
    targets = args.target_layers or saved_targets
    targets = sorted(set(targets))
    if any(target not in saved_targets for target in targets):
        raise ValueError(f"Requested target not in checkpoint training budgets: {saved_targets}")
    dataset = impl.bert_impl.load_sst2(tokenizer)["validation"]
    if args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dtype = next(model.parameters()).dtype
    result_rows = []
    for target in targets:
        print(f"=== Benchmarking {target}/6 ===", flush=True)
        spec = target_spec(candidates, positions, target, dtype)
        latency = condition(1, args, components, spec, dataset)
        throughput = condition(args.throughput_batch_size, args, components, spec, dataset)
        result_rows += summarized("batch_1_latency", target, latency)
        result_rows += summarized("batch_n_throughput", target, throughput)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    report = output_dir / "interaction_multibudget_inference.md"
    write_report(report, args, result_rows)
    print(f"Saved benchmark report: {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target_layers", type=int, nargs="+")
    parser.add_argument("--throughput_batch_size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "inference_benchmark"))
    main(parser.parse_args())
