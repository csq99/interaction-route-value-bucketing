"""Repeat the route-selection diagnosis with persistent random generators."""

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


def route_text(route, candidates, layer_count):
    return str([0] + [candidates[index] for index in route] + [layer_count - 1])


def choose_indices(scores, method, generator):
    if method == "argmax":
        return scores.argmax(dim=1)
    if method == "uniform_all":
        return torch.randint(scores.size(1), (scores.size(0),), device=scores.device, generator=generator)
    if method.startswith("topk_uniform_"):
        k = min(int(method.rsplit("_", 1)[1]), scores.size(1))
        order = scores.argsort(dim=1, descending=True)[:, :k]
        picks = torch.randint(k, (scores.size(0),), device=scores.device, generator=generator)
        return order.gather(1, picks[:, None]).squeeze(1)
    if method.startswith("zsoftmax_"):
        temperature = float(method.split("_", 1)[1])
        centered = scores.float() - scores.float().mean(dim=1, keepdim=True)
        scale = scores.float().std(dim=1, keepdim=True).clamp_min(1e-6)
        probabilities = torch.softmax((centered / scale) / temperature, dim=1)
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    raise ValueError(f"Unknown selection method: {method}")


@torch.inference_mode()
def main(args):
    impl.set_seed(args.seed)
    impl.config.MODEL_ID = args.model_id
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    router_ckpt = torch.load(args.router_checkpoint, map_location=impl.config.DEVICE, weights_only=False)
    adapter_ckpt = torch.load(args.adapter_checkpoint, map_location=impl.config.DEVICE, weights_only=False)
    if int(adapter_ckpt["epoch"]) != 15:
        raise ValueError("This experiment expects the epoch-15 Adapter checkpoint.")
    saved_args = router_ckpt["args"]
    model, transform = impl.load_model_and_processor()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model)
    candidates = router_ckpt["candidate_layers"]
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(
        model.config.hidden_size, saved_args["adapter_bottleneck_dim"], len(layers), dtype,
    ).to(impl.config.DEVICE)
    adapter.load_state_dict(adapter_ckpt["adapter_state_dict"])
    adapter.eval()
    router = impl.InteractionRouteValueRouter(
        model.config.hidden_size, len(candidates), saved_args["router_hidden_dim"], dtype,
    ).to(impl.config.DEVICE)
    router.load_state_dict(router_ckpt["router_state_dict"])
    router.eval()

    targets = [4, 6, 7]
    routes = {
        target: impl.all_middle_routes(len(candidates), target - 2, impl.config.DEVICE)
        for target in targets
    }
    masks = {
        target: impl.pair_masks(len(candidates), routes[target], dtype)[1]
        for target in targets
    }
    loader, sample_count = shared.make_loader(
        args.eval_csv, transform, args.batch_size, args.max_eval_samples,
        args.num_workers, False,
    )
    methods = [
        "argmax", "zsoftmax_0.25", "zsoftmax_0.5", "zsoftmax_1.0",
        "topk_uniform_5", "uniform_all",
    ]
    repeats = {method: (1 if method == "argmax" else args.repeats) for method in methods}
    states = {
        (target, method, repeat): {
            "correct": 0, "total": 0, "usage": Counter(),
            "margin_sum": 0.0, "entropy_sum": 0.0,
        }
        for target in targets for method in methods
        for repeat in range(repeats[method])
    }
    # A generator is created once per complete evaluation stream, not once per batch.
    generators = {}
    for target in targets:
        for method in methods:
            if method == "argmax":
                continue
            for repeat in range(repeats[method]):
                generator = torch.Generator(device=impl.config.DEVICE)
                generator.manual_seed(
                    args.seed + 1000 * target + 100 * repeat + sum(map(ord, method))
                )
                generators[(target, method, repeat)] = generator

    for images, labels in loader:
        images = images.to(impl.config.DEVICE)
        labels = labels.to(impl.config.DEVICE)
        _, teacher_hidden = shared.teacher_batch(model, images)
        hidden0 = teacher_hidden[1]
        unary, pairwise = router(hidden0)
        for target in targets:
            scores = impl.score_routes(unary, pairwise, routes[target], masks[target])
            probabilities = torch.softmax(scores.float(), dim=1).clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=1) / math.log(scores.size(1))
            top2 = scores.topk(2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            for method in methods:
                for repeat in range(repeats[method]):
                    generator = generators.get((target, method, repeat))
                    chosen_indices = choose_indices(scores, method, generator)
                    chosen_routes = routes[target].index_select(0, chosen_indices)
                    hidden = shared.execute_grouped_routes(
                        layers, adapter, hidden0, chosen_routes, candidates,
                    )
                    predicted = impl.logits_from_hidden(model, hidden).argmax(-1)
                    stat = states[(target, method, repeat)]
                    stat["correct"] += int(predicted.eq(labels).sum())
                    stat["total"] += labels.numel()
                    stat["margin_sum"] += float(margin.sum())
                    stat["entropy_sum"] += float(entropy.sum())
                    for route in chosen_routes.detach().cpu().tolist():
                        stat["usage"][tuple(route)] += 1

    summary, usage = [], []
    for (target, method, repeat), stat in states.items():
        total = stat["total"]
        counts = stat["usage"]
        summary.append({
            "adapter_epoch": 15,
            "router_epoch": int(router_ckpt.get("epoch", 1)),
            "target_layers": target,
            "method": method,
            "repeat": repeat,
            "samples": total,
            "accuracy": stat["correct"] / total,
            "selected_route_count": len(counts),
            "max_route_share": max(counts.values()) / total,
            "mean_top1_top2_score_margin": stat["margin_sum"] / total,
            "mean_normalized_score_entropy": stat["entropy_sum"] / total,
        })
        for route, count in counts.most_common():
            usage.append({
                "adapter_epoch": 15,
                "router_epoch": int(router_ckpt.get("epoch", 1)),
                "target_layers": target,
                "method": method,
                "repeat": repeat,
                "route": route_text(route, candidates, len(layers)),
                "count": count,
                "share": count / total,
            })
    write_csv(output / "selection_rule_summary.csv", summary)
    write_csv(output / "selection_rule_route_usage.csv", usage)
    print(f"Evaluated {sample_count} samples; saved fixed-RNG diagnosis to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
