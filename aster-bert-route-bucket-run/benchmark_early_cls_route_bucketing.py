"""Benchmark batch route bucketing for the existing Early-CLS+BCE BERT router.

The router and adapter weights are loaded from the completed BERT experiment.
Only the execution schedule changes: samples either run one by one after route
selection or are grouped by their identical dynamically predicted routes.
"""

import argparse
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader

SOURCE_DIR = Path(__file__).resolve().parents[1] / "aster-bert"
sys.path.insert(0, str(SOURCE_DIR))

import config_bert as config
from components import DynamicAdapter
from model_utils_bert import load_model_and_tokenizer
from train_early_cls_topk_contribution_router_bert import (
    EarlyClsMiddleContributionPredictor,
    layer_hidden_state,
    load_sst2,
    logits_from_hidden,
    select_middle_topk_routes,
)


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def route_key(route):
    return tuple(route)


def prepare_routes(model, predictor, input_ids, attention_mask, candidate_layers, target_layers):
    """Run the shared prefix once and make one route decision per sample."""
    hidden = model.distilbert.embeddings(input_ids=input_ids)
    hidden = layer_hidden_state(
        model.distilbert.transformer.layer[0](hidden, attn_mask=attention_mask)
    )
    scores = predictor(hidden[:, 0, :])
    routes = select_middle_topk_routes(
        scores,
        target_layers,
        len(model.distilbert.transformer.layer),
        candidate_layers,
    )
    return hidden, routes


def execute_route(model, adapter, hidden, attention_mask, route):
    """Execute one route for a whole tensor of samples with the same route."""
    state = hidden
    previous_layer = 0
    for layer_idx in route[1:]:
        if layer_idx > previous_layer + 1:
            state = adapter(state, previous_layer, layer_idx)
        state = layer_hidden_state(
            model.distilbert.transformer.layer[layer_idx](state, attn_mask=attention_mask)
        )
        previous_layer = layer_idx
    return state


def run_per_sample(model, predictor, adapter, input_ids, attention_mask, candidate_layers, target_layers):
    hidden, routes = prepare_routes(
        model, predictor, input_ids, attention_mask, candidate_layers, target_layers
    )
    final_states = []
    for sample_idx, route in enumerate(routes):
        final_states.append(
            execute_route(
                model,
                adapter,
                hidden[sample_idx : sample_idx + 1],
                attention_mask[sample_idx : sample_idx + 1],
                route,
            )
        )
    return torch.cat(final_states, dim=0), routes


def run_route_bucketed(model, predictor, adapter, input_ids, attention_mask, candidate_layers, target_layers):
    hidden, routes = prepare_routes(
        model, predictor, input_ids, attention_mask, candidate_layers, target_layers
    )
    route_groups = defaultdict(list)
    for sample_idx, route in enumerate(routes):
        route_groups[route_key(route)].append(sample_idx)

    final_states = torch.empty_like(hidden)
    for route, indices in route_groups.items():
        index_tensor = torch.tensor(indices, device=hidden.device, dtype=torch.long)
        grouped_state = execute_route(
            model,
            adapter,
            hidden.index_select(0, index_tensor),
            attention_mask.index_select(0, index_tensor),
            list(route),
        )
        final_states.index_copy_(0, index_tensor, grouped_state)
    return final_states, routes


def run_full(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


def evaluate(method, loader, model, predictor, adapter, candidate_layers, target_layers):
    correct = 0
    total = 0
    route_counter = Counter()
    sync_if_cuda()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(config.DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(config.DEVICE, non_blocking=True)
            labels = batch["labels"].to(config.DEVICE, non_blocking=True)
            if method == "full":
                logits = run_full(model, input_ids, attention_mask)
            elif method == "per_sample":
                final_state, routes = run_per_sample(
                    model, predictor, adapter, input_ids, attention_mask, candidate_layers, target_layers
                )
                route_counter.update(route_key(route) for route in routes)
                logits = logits_from_hidden(model, final_state)
            elif method == "bucketed":
                final_state, routes = run_route_bucketed(
                    model, predictor, adapter, input_ids, attention_mask, candidate_layers, target_layers
                )
                route_counter.update(route_key(route) for route in routes)
                logits = logits_from_hidden(model, final_state)
            else:
                raise ValueError(f"Unknown method: {method}")
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.numel()
    sync_if_cuda()
    elapsed = time.perf_counter() - start
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "seconds": elapsed,
        "routes": route_counter,
    }


def format_route(route):
    return "[" + ", ".join(str(layer) for layer in route) + "]"


def write_report(path, args, results):
    full_median = statistics.median(row["seconds"] for row in results["full"])
    per_sample_median = statistics.median(row["seconds"] for row in results["per_sample"])
    bucketed_median = statistics.median(row["seconds"] for row in results["bucketed"])
    latest = results["bucketed"][-1]
    routes = latest["routes"]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Early-CLS Dynamic Route Bucketing BERT Benchmark\n\n")
        handle.write(
            "This benchmark reuses one completed Early-CLS+BCE predictor and adapter checkpoint. "
            "It changes only inference scheduling: per-sample route execution versus grouping "
            "samples with identical dynamically predicted routes inside each batch.\n\n"
        )
        handle.write("## Setup\n\n")
        handle.write(f"- Checkpoint: `{args.checkpoint}`\n")
        handle.write(f"- Target executed layers: `{args.target_executed_layers}` / 6\n")
        handle.write(f"- Eval batch size: `{args.batch_size}`\n")
        handle.write(f"- Timed repeats: `{args.repeats}`\n")
        handle.write(f"- Warm-up passes: `{args.warmup}`\n\n")
        handle.write("## Median Results\n\n")
        handle.write("| Method | Accuracy | Time (s) | Samples/s | Speedup vs Full |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for name, label, median_seconds in [
            ("full", "Full baseline", full_median),
            ("per_sample", "Dynamic router, per-sample dispatch", per_sample_median),
            ("bucketed", "Dynamic router, route-bucketed dispatch", bucketed_median),
        ]:
            row = results[name][-1]
            handle.write(
                f"| {label} | {row['accuracy']:.4f} | {median_seconds:.4f} | "
                f"{row['total'] / median_seconds:.2f} | {full_median / median_seconds:.4f}x |\n"
            )
        handle.write("\n## Dynamic Route Distribution\n\n")
        handle.write(f"- Unique routes in final bucketed pass: `{len(routes)}`\n")
        for route, count in routes.most_common():
            handle.write(f"- `{format_route(route)}`: {count}\n")
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Per-sample and bucketed modes use exactly the same checkpoint and top-k route rule. "
            "Any output mismatch is a correctness failure; timing differences isolate execution scheduling. "
            "Batch size 1 cannot benefit from grouping, while larger batches can only group samples "
            "whose routes the router independently selected as identical.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark dynamic BERT route bucketing.")
    parser.add_argument(
        "--checkpoint",
        default=str(
            SOURCE_DIR
            / "checkpoints_distilbert_sst2"
            / "early_cls_topk_contribution_router_cls4_bce_local"
            / "early_cls_topk_contribution_router_bert_checkpoint.pt"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_executed_layers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--output", default="route_bucket_benchmark_bce_cls4_bs32.md")
    return parser.parse_args()


def main(args):
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    model, tokenizer = load_model_and_tokenizer()
    tokenized = load_sst2(tokenizer)
    eval_dataset = tokenized["validation"]
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    candidate_layers = list(range(1, len(model.distilbert.transformer.layer) - 1))
    dtype = next(model.parameters()).dtype
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE, weights_only=False)
    predictor = EarlyClsMiddleContributionPredictor(
        model.config.hidden_size,
        checkpoint["args"]["predictor_hidden_dim"],
        candidate_layers,
        dtype,
    ).to(config.DEVICE)
    adapter = DynamicAdapter(
        model.config.hidden_size,
        config.ADAPTER_BOTTLENECK_DIM,
        len(model.distilbert.transformer.layer),
        dtype,
    ).to(config.DEVICE)
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    model.eval()
    predictor.eval()
    adapter.eval()

    methods = ["full", "per_sample", "bucketed"]
    for _ in range(args.warmup):
        for method in methods:
            evaluate(method, loader, model, predictor, adapter, candidate_layers, args.target_executed_layers)

    results = {method: [] for method in methods}
    for repeat_idx in range(args.repeats):
        for method in methods:
            row = evaluate(method, loader, model, predictor, adapter, candidate_layers, args.target_executed_layers)
            results[method].append(row)
            print(
                f"repeat={repeat_idx + 1} method={method} accuracy={row['accuracy']:.4f} "
                f"time={row['seconds']:.4f}s routes={len(row['routes'])}"
            )

    per_sample_predictions = results["per_sample"][-1]["correct"]
    bucketed_predictions = results["bucketed"][-1]["correct"]
    if per_sample_predictions != bucketed_predictions:
        raise RuntimeError(
            "Route bucketing changed model outputs: per-sample and bucketed correct counts differ."
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    write_report(output_path, args, results)
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main(parse_args())
