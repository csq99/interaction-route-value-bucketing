"""Exhaustive route-quality diagnosis for a trained shared DeiT interaction checkpoint."""

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


def route_text(route, candidate_layers, layer_count):
    return str([0] + [candidate_layers[index] for index in route.tolist()] + [layer_count - 1])


def rank_of_index(order, indices):
    """Return one-based ranks for indices in every row of an argsort result."""
    return order.eq(indices.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1


@torch.no_grad()
def diagnose_target(model, layers, adapter, router, loader, routes, pair_mask, candidate_layers, target, route_kd_weight):
    route_count = len(routes)
    total = full_correct = router_correct = loss_oracle_correct = correct_oracle_correct = 0
    router_objective_sum = loss_oracle_objective_sum = 0.0
    oracle_match_sum = oracle_score_rank_sum = router_loss_rank_sum = score_corr_sum = 0.0
    route_objective_sums = torch.zeros(route_count, dtype=torch.float64)
    route_correct_sums = torch.zeros(route_count, dtype=torch.long)
    router_usage, oracle_usage = Counter(), Counter()

    for images, labels in tqdm(loader, desc=f"Exhaustive oracle diagnosis ({target}/{len(layers)})"):
        images = images.to(impl.config.DEVICE)
        labels = labels.to(impl.config.DEVICE)
        teacher_logits, teacher_hidden = shared.teacher_batch(model, images)
        hidden0 = teacher_hidden[1]
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        unary, pairwise = router(hidden0)
        scores = impl.score_routes(unary, pairwise, routes, pair_mask)

        losses, correct = [], []
        for route in routes:
            logits = impl.logits_from_hidden(model, shared.execute_route(layers, adapter, hidden0, route, candidate_layers))
            ce = F.cross_entropy(logits, labels, reduction="none")
            kd = F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs, reduction="none").sum(dim=-1)
            losses.append(ce + route_kd_weight * kd)
            correct.append(logits.argmax(-1).eq(labels))
        losses = torch.stack(losses, dim=1)
        correct = torch.stack(correct, dim=1)

        router_indices = scores.argmax(dim=1)
        loss_oracle_indices = losses.argmin(dim=1)
        batch = labels.numel()
        row = torch.arange(batch, device=labels.device)
        full_correct += int(teacher_logits.argmax(-1).eq(labels).sum())
        router_correct += int(correct[row, router_indices].sum())
        loss_oracle_correct += int(correct[row, loss_oracle_indices].sum())
        correct_oracle_correct += int(correct.any(dim=1).sum())
        total += batch
        router_objective_sum += float(losses[row, router_indices].sum())
        loss_oracle_objective_sum += float(losses[row, loss_oracle_indices].sum())
        oracle_match_sum += float(router_indices.eq(loss_oracle_indices).sum())

        score_order = scores.argsort(dim=1, descending=True)
        loss_order = losses.argsort(dim=1)
        oracle_score_rank_sum += float(rank_of_index(score_order, loss_oracle_indices).sum())
        router_loss_rank_sum += float(rank_of_index(loss_order, router_indices).sum())
        centered_scores = scores - scores.mean(dim=1, keepdim=True)
        centered_negative_losses = -losses + losses.mean(dim=1, keepdim=True)
        denominator = centered_scores.square().sum(dim=1).sqrt() * centered_negative_losses.square().sum(dim=1).sqrt()
        score_corr_sum += float((centered_scores * centered_negative_losses).sum(dim=1).div(denominator.clamp_min(1e-12)).sum())

        route_objective_sums += losses.detach().double().sum(dim=0).cpu()
        route_correct_sums += correct.detach().sum(dim=0).cpu()
        router_usage.update(router_indices.detach().cpu().tolist())
        oracle_usage.update(loss_oracle_indices.detach().cpu().tolist())

    best_fixed_objective_index = int(route_objective_sums.argmin())
    best_fixed_accuracy_index = int(route_correct_sums.argmax())
    route_rows = []
    for index, route in enumerate(routes.detach().cpu()):
        route_rows.append({
            "target_layers": target,
            "route": route_text(route, candidate_layers, len(layers)),
            "mean_objective": float(route_objective_sums[index] / total),
            "fixed_accuracy": float(route_correct_sums[index] / total),
            "router_selected_count": router_usage[index],
            "loss_oracle_selected_count": oracle_usage[index],
        })
    summary = {
        "target_layers": target,
        "legal_routes": route_count,
        "samples": total,
        "full_accuracy": full_correct / total,
        "router_accuracy": router_correct / total,
        "loss_oracle_accuracy": loss_oracle_correct / total,
        "any_correct_oracle_accuracy": correct_oracle_correct / total,
        "router_mean_objective": router_objective_sum / total,
        "loss_oracle_mean_objective": loss_oracle_objective_sum / total,
        "router_matches_loss_oracle_share": oracle_match_sum / total,
        "loss_oracle_mean_router_rank": oracle_score_rank_sum / total,
        "router_choice_mean_loss_rank": router_loss_rank_sum / total,
        "mean_score_vs_negative_loss_pearson": score_corr_sum / total,
        "best_fixed_objective_route": route_rows[best_fixed_objective_index]["route"],
        "best_fixed_objective_accuracy": float(route_correct_sums[best_fixed_objective_index] / total),
        "best_fixed_objective": float(route_objective_sums[best_fixed_objective_index] / total),
        "best_fixed_accuracy_route": route_rows[best_fixed_accuracy_index]["route"],
        "best_fixed_accuracy": float(route_correct_sums[best_fixed_accuracy_index] / total),
        "router_selected_route_count": len(router_usage),
        "router_max_route_share": max(router_usage.values()) / total,
    }
    return summary, route_rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    impl.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=impl.config.DEVICE, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_targets = sorted(checkpoint["target_layers"])
    requested_targets = sorted(set(args.target_layers))
    if requested_targets != checkpoint_targets:
        raise ValueError(f"Requested targets {requested_targets} do not match checkpoint targets {checkpoint_targets}.")

    impl.config.MODEL_ID = args.model_id
    model, transform = impl.load_model_and_processor()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model)
    candidate_layers = checkpoint["candidate_layers"]
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(model.config.hidden_size, checkpoint_args["adapter_bottleneck_dim"], len(layers), dtype).to(impl.config.DEVICE)
    router = impl.InteractionRouteValueRouter(model.config.hidden_size, len(candidate_layers), checkpoint_args["router_hidden_dim"], dtype).to(impl.config.DEVICE)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    router.load_state_dict(checkpoint["router_state_dict"])
    adapter.eval()
    router.eval()

    loader, sample_count = shared.make_loader(args.eval_csv, transform, args.batch_size, args.max_eval_samples, args.num_workers, False)
    summaries, route_rows = [], []
    for target in requested_targets:
        routes = impl.all_middle_routes(len(candidate_layers), target - 2, impl.config.DEVICE)
        pair_mask = impl.pair_masks(len(candidate_layers), routes, dtype)[1]
        summary, rows = diagnose_target(model, layers, adapter, router, loader, routes, pair_mask,
                                        candidate_layers, target, args.route_kd_weight)
        summaries.append(summary)
        route_rows.extend(rows)

    write_csv(output_dir / "oracle_diagnostic_summary.csv", summaries)
    write_csv(output_dir / "route_quality.csv", route_rows)
    report = output_dir / "interaction_router_oracle_diagnostic.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# DeiT Interaction Router vs Oracle Diagnostic\n\n")
        handle.write(f"- Checkpoint: `{args.checkpoint}`\n")
        handle.write(f"- Fixed evaluation subset: first `{sample_count}` rows of `{args.eval_csv}`\n")
        handle.write(f"- Objective: CE + `{args.route_kd_weight}` * logit-KD, evaluated over every legal route per budget.\n")
        handle.write("- `loss oracle` chooses the minimum objective route per image. `any-correct oracle` chooses whether any legal route predicts the correct label.\n\n")
        handle.write("| Budget | Router acc | Loss-oracle acc | Any-correct oracle acc | Best fixed acc | Router/oracle match | Oracle mean router rank | Router mean loss rank | Score/-loss corr | Router max share |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            handle.write(f"| {row['target_layers']}/12 | {row['router_accuracy']:.4f} | {row['loss_oracle_accuracy']:.4f} | {row['any_correct_oracle_accuracy']:.4f} | {row['best_fixed_accuracy']:.4f} | {row['router_matches_loss_oracle_share']:.4f} | {row['loss_oracle_mean_router_rank']:.2f} | {row['router_choice_mean_loss_rank']:.2f} | {row['mean_score_vs_negative_loss_pearson']:.4f} | {row['router_max_route_share']:.4f} |\n")
            handle.write(f"\n{row['target_layers']}/12 best fixed by objective: `{row['best_fixed_objective_route']}`; best fixed by accuracy: `{row['best_fixed_accuracy_route']}`.\n")
    print(f"Saved diagnostic report: {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[4, 6, 7])
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--route_kd_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
