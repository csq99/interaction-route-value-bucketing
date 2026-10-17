"""Fixed-route quality, unary-vs-interaction, and oracle diagnosis for one budget/checkpoint."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

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


def route_text(route, candidate_layers, layer_count):
    return str([0] + [candidate_layers[index] for index in route.tolist()] + [layer_count - 1])


def one_based_rank(order, chosen):
    return order.eq(chosen.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1


def load_components(args):
    router_checkpoint = torch.load(args.router_checkpoint, map_location=impl.config.DEVICE, weights_only=False)
    saved_args = router_checkpoint["args"]
    impl.config.MODEL_ID = args.model_id
    model, transform = impl.load_model_and_processor()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model)
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(
        model.config.hidden_size, saved_args["adapter_bottleneck_dim"], len(layers), dtype
    ).to(impl.config.DEVICE)
    adapter_checkpoint = torch.load(args.adapter_checkpoint, map_location=impl.config.DEVICE, weights_only=False)
    adapter.load_state_dict(adapter_checkpoint["adapter_state_dict"])
    adapter.eval()
    router = impl.InteractionRouteValueRouter(
        model.config.hidden_size, len(router_checkpoint["candidate_layers"]),
        saved_args["router_hidden_dim"], dtype,
    ).to(impl.config.DEVICE)
    router.load_state_dict(router_checkpoint["router_state_dict"])
    router.eval()
    return router_checkpoint, model, transform, layers, adapter, router, dtype


@torch.inference_mode()
def diagnose(args, checkpoint, model, layers, adapter, router, loader, routes, pair_mask, candidates):
    route_count = len(routes)
    total = full_correct = loss_oracle_correct = any_correct = 0
    objective_sums = torch.zeros(route_count, dtype=torch.float64)
    correct_sums = torch.zeros(route_count, dtype=torch.long)
    modes = ("interaction", "unary_only") if args.compare_router_scores else ()
    mode_stats = {name: {
        "correct": 0, "objective": 0.0, "oracle_match": 0, "oracle_score_rank": 0.0,
        "choice_loss_rank": 0.0, "corr": 0.0, "usage": Counter(),
    } for name in modes}
    oracle_usage = Counter()

    for images, labels in tqdm(loader, desc=f"routes {args.adapter_epoch}e {args.target_layers}/12"):
        images = images.to(impl.config.DEVICE)
        labels = labels.to(impl.config.DEVICE)
        teacher_logits, teacher_hidden = shared.teacher_batch(model, images)
        hidden0 = teacher_hidden[1]
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        losses, correct = [], []
        for route in routes:
            hidden = shared.execute_route(layers, adapter, hidden0, route, candidates)
            logits = impl.logits_from_hidden(model, hidden)
            ce = F.cross_entropy(logits, labels, reduction="none")
            kd = F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs, reduction="none").sum(dim=-1)
            losses.append(ce + args.route_kd_weight * kd)
            correct.append(logits.argmax(-1).eq(labels))
        losses = torch.stack(losses, dim=1)
        correct = torch.stack(correct, dim=1)
        oracle = losses.argmin(dim=1)
        row = torch.arange(labels.numel(), device=labels.device)
        total += labels.numel()
        full_correct += int(teacher_logits.argmax(-1).eq(labels).sum())
        loss_oracle_correct += int(correct[row, oracle].sum())
        any_correct += int(correct.any(dim=1).sum())
        objective_sums += losses.double().sum(dim=0).cpu()
        correct_sums += correct.sum(dim=0).cpu()
        oracle_usage.update(oracle.cpu().tolist())

        if modes:
            unary, pairwise = router(hidden0)
            score_map = {
                "interaction": impl.score_routes(unary, pairwise, routes, pair_mask),
                "unary_only": unary[:, routes].sum(dim=-1),
            }
            for name, scores in score_map.items():
                chosen = scores.argmax(dim=1)
                stats = mode_stats[name]
                stats["correct"] += int(correct[row, chosen].sum())
                stats["objective"] += float(losses[row, chosen].sum())
                stats["oracle_match"] += int(chosen.eq(oracle).sum())
                stats["oracle_score_rank"] += float(one_based_rank(scores.argsort(1, descending=True), oracle).sum())
                stats["choice_loss_rank"] += float(one_based_rank(losses.argsort(1), chosen).sum())
                centered_scores = scores - scores.mean(1, keepdim=True)
                centered_values = -losses + losses.mean(1, keepdim=True)
                denominator = centered_scores.square().sum(1).sqrt() * centered_values.square().sum(1).sqrt()
                stats["corr"] += float(
                    (centered_scores * centered_values).sum(1).div(denominator.clamp_min(1e-12)).sum()
                )
                stats["usage"].update(chosen.cpu().tolist())

    accuracies = correct_sums.double() / total
    objectives = objective_sums / total
    route_rows = []
    for index, route in enumerate(routes.cpu()):
        route_rows.append({
            "adapter_epoch": args.adapter_epoch, "target_layers": args.target_layers,
            "route_index": index, "route": route_text(route, candidates, len(layers)),
            "fixed_accuracy": float(accuracies[index]), "mean_objective": float(objectives[index]),
            "loss_oracle_selected_count": oracle_usage[index],
        })
    best_accuracy_index = int(accuracies.argmax())
    best_objective_index = int(objectives.argmin())
    summary = {
        "adapter_epoch": args.adapter_epoch, "target_layers": args.target_layers,
        "samples": total, "legal_routes": route_count, "full_accuracy": full_correct / total,
        "fixed_accuracy_mean": float(accuracies.mean()),
        "fixed_accuracy_std": float(accuracies.std(unbiased=False)),
        "fixed_accuracy_min": float(accuracies.min()), "fixed_accuracy_max": float(accuracies.max()),
        "best_fixed_accuracy_route": route_rows[best_accuracy_index]["route"],
        "best_objective": float(objectives[best_objective_index]),
        "best_objective_route": route_rows[best_objective_index]["route"],
        "best_objective_route_accuracy": float(accuracies[best_objective_index]),
        "loss_oracle_accuracy": loss_oracle_correct / total,
        "any_correct_oracle_accuracy": any_correct / total,
    }
    return summary, route_rows, mode_stats


def main(args):
    impl.set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, model, transform, layers, adapter, router, dtype = load_components(args)
    candidates = checkpoint["candidate_layers"]
    routes = impl.all_middle_routes(len(candidates), args.target_layers - 2, impl.config.DEVICE)
    pair_mask = impl.pair_masks(len(candidates), routes, dtype)[1]
    loader, sample_count = shared.make_loader(
        args.eval_csv, transform, args.batch_size, args.max_eval_samples, args.num_workers, False
    )

    summary, route_rows, mode_stats = diagnose(
        args, checkpoint, model, layers, adapter, router, loader, routes, pair_mask, candidates
    )
    mode_rows = []
    for name, stats in mode_stats.items():
        usage = stats["usage"]
        mode_rows.append({
            "adapter_epoch": args.adapter_epoch, "target_layers": args.target_layers, "score_mode": name,
            "samples": summary["samples"], "accuracy": stats["correct"] / summary["samples"],
            "mean_objective": stats["objective"] / summary["samples"],
            "loss_oracle_match_share": stats["oracle_match"] / summary["samples"],
            "loss_oracle_mean_score_rank": stats["oracle_score_rank"] / summary["samples"],
            "router_choice_mean_loss_rank": stats["choice_loss_rank"] / summary["samples"],
            "mean_score_vs_negative_loss_pearson": stats["corr"] / summary["samples"],
            "selected_route_count": len(usage),
            "max_route_share": max(usage.values()) / summary["samples"],
        })
    write_csv(output / "fixed_route_summary.csv", [summary])
    write_csv(output / "route_quality.csv", route_rows)
    write_csv(output / "router_score_ablation.csv", mode_rows)
    print(f"Saved route diagnosis to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--adapter_epoch", type=int, required=True)
    parser.add_argument("--target_layers", type=int, required=True)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--route_kd_weight", type=float, default=1.0)
    parser.add_argument("--compare_router_scores", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
