"""Compare saved interaction routers on one fixed test subset and adapter."""

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aster-deit-interaction-multibudget-run"))
import train_interaction_multibudget_deit as shared

impl = shared.impl


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def route_name(route, candidates, layer_count):
    return str([0] + [candidates[index] for index in route] + [layer_count - 1])


def load_router(path, hidden_size, candidate_count, hidden_dim, dtype):
    checkpoint = torch.load(path, map_location=impl.config.DEVICE, weights_only=False)
    if sorted(checkpoint["target_layers"]) != [4, 6, 7]:
        raise ValueError(f"Unexpected budgets in {path}")
    router = impl.InteractionRouteValueRouter(hidden_size, candidate_count, hidden_dim, dtype)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.to(impl.config.DEVICE).eval()
    return router


@torch.inference_mode()
def evaluate(args):
    impl.set_seed(args.seed)
    impl.config.MODEL_ID = args.model_id
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    adapter_ckpt = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    if int(adapter_ckpt["epoch"]) != 15:
        raise ValueError("This comparison expects the fixed epoch-15 Adapter.")
    reference = torch.load(args.router_epoch1, map_location="cpu", weights_only=False)
    saved_args = reference["args"]
    model, transform = impl.load_model_and_processor()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model)
    candidates = reference["candidate_layers"]
    if candidates != list(range(1, len(layers) - 1)):
        raise ValueError("Candidate layers do not match this model.")
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(model.config.hidden_size, saved_args["adapter_bottleneck_dim"],
                                  len(layers), dtype)
    adapter.load_state_dict(adapter_ckpt["adapter_state_dict"])
    adapter.to(impl.config.DEVICE).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    router_specs = [
        (1, args.router_epoch1),
        (2, args.router_epoch2),
        (3, args.router_epoch3),
    ]
    routers = [(epoch, load_router(path, model.config.hidden_size, len(candidates),
                                    saved_args["router_hidden_dim"], dtype))
               for epoch, path in router_specs]
    targets = [4, 6, 7]
    routes = {target: impl.all_middle_routes(len(candidates), target - 2, impl.config.DEVICE)
              for target in targets}
    masks = {target: impl.pair_masks(len(candidates), routes[target], dtype)[1]
             for target in targets}
    loader, count = shared.make_loader(args.eval_csv, transform, args.batch_size,
                                       args.max_eval_samples, args.num_workers, False)

    # Keep only per-budget/method counters in memory; images are processed once.
    states = {}
    for epoch, _ in router_specs:
        for target in targets:
            for method in ("interaction", "unary_only"):
                states[(epoch, target, method)] = {
                    "correct": 0, "total": 0, "usage": Counter(),
                    "margin_sum": 0.0, "entropy_sum": 0.0,
                }

    for images, labels in loader:
        images = images.to(impl.config.DEVICE)
        labels = labels.to(impl.config.DEVICE)
        _, teacher_hidden = shared.teacher_batch(model, images)
        hidden0 = teacher_hidden[1]
        for target in targets:
            budget_routes = routes[target]
            pair_mask = masks[target]
            for epoch, router in routers:
                unary, pairwise = router(hidden0)
                score_sets = {
                    "interaction": impl.score_routes(unary, pairwise, budget_routes, pair_mask),
                    "unary_only": unary[:, budget_routes].sum(dim=-1),
                }
                for method, scores in score_sets.items():
                    chosen_indices = scores.argmax(dim=1)
                    chosen_routes = budget_routes.index_select(0, chosen_indices)
                    hidden = shared.execute_grouped_routes(layers, adapter, hidden0,
                                                           chosen_routes, candidates)
                    predicted = impl.logits_from_hidden(model, hidden).argmax(-1)
                    stat = states[(epoch, target, method)]
                    stat["correct"] += int(predicted.eq(labels).sum())
                    stat["total"] += labels.numel()
                    for route in chosen_routes.detach().cpu().tolist():
                        key = tuple(route)
                        stat["usage"][key] += 1
                    if scores.size(1) > 1:
                        top2 = scores.topk(2, dim=1).values
                        stat["margin_sum"] += float((top2[:, 0] - top2[:, 1]).mean()) * labels.numel()
                    probabilities = torch.softmax(scores.float(), dim=1).clamp_min(1e-12)
                    entropy = -(probabilities * probabilities.log()).sum(dim=1)
                    norm_entropy = entropy / math.log(scores.size(1))
                    stat["entropy_sum"] += float(norm_entropy.mean()) * labels.numel()

    summary_rows, usage_rows = [], []
    for (epoch, target, method), stat in states.items():
        total = stat["total"]
        usage = stat["usage"]
        summary_rows.append({
            "router_epoch": epoch, "adapter_epoch": 15, "target_layers": target,
            "method": method, "samples": total, "accuracy": stat["correct"] / total,
            "selected_route_count": len(usage),
            "max_route_share": max(usage.values()) / total,
            "mean_top1_top2_score_margin": stat["margin_sum"] / total,
            "mean_normalized_score_entropy": stat["entropy_sum"] / total,
        })
        for route, route_count in usage.most_common():
            usage_rows.append({
                "router_epoch": epoch, "adapter_epoch": 15, "target_layers": target,
                "method": method, "route": route_name(route, candidates, len(layers)),
                "count": route_count, "share": route_count / total,
            })
    write_csv(output / "router_epoch_comparison.csv", summary_rows)
    write_csv(output / "route_usage.csv", usage_rows)
    print(f"Evaluated {count} test samples; results saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--router_epoch1", required=True)
    parser.add_argument("--router_epoch2", required=True)
    parser.add_argument("--router_epoch3", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    evaluate(parser.parse_args())
