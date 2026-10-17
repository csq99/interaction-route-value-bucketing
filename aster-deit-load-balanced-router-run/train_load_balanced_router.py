"""Train quality-aware, load-balanced interaction routers with a frozen Adapter."""

import argparse
import csv
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
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


def variant_name(weight):
    return f"load_{weight:g}".replace(".", "p")


def load_balance_loss(scores, target_distribution, temperature, prior_smoothing):
    """Match a sharp top-1 proxy to a quality-aware, mildly smoothed prior."""
    normalized = scores.float() - scores.float().mean(dim=1, keepdim=True)
    normalized = normalized / scores.float().std(dim=1, keepdim=True).clamp_min(1e-6)
    top1_proxy = F.softmax(normalized / temperature, dim=1)
    predicted_load = top1_proxy.mean(dim=0).clamp_min(1e-8)

    quality_prior = target_distribution.detach().float().mean(dim=0)
    uniform = torch.full_like(quality_prior, 1.0 / quality_prior.numel())
    prior = ((1.0 - prior_smoothing) * quality_prior + prior_smoothing * uniform).clamp_min(1e-8)
    prior = prior / prior.sum()
    return F.kl_div(predicted_load.log(), prior, reduction="sum"), predicted_load, prior


@torch.inference_mode()
def evaluate_router(model, layers, adapter, router, loader, routes_by_target,
                    masks_by_target, candidates, method):
    rows, usage_rows = [], []
    for target in sorted(routes_by_target):
        model.eval()
        adapter.eval()
        router.eval()
        full_correct = dynamic_correct = total = 0
        margin_sum = entropy_sum = 0.0
        route_counter, route_correct = Counter(), Counter()
        routes = routes_by_target[target]
        for images, labels in tqdm(loader, desc=f"Evaluate {method} ({target}/12)"):
            images = images.to(impl.config.DEVICE)
            labels = labels.to(impl.config.DEVICE)
            teacher_logits, teacher_hidden = shared.teacher_batch(model, images)
            hidden0 = teacher_hidden[1]
            unary, pairwise = router(hidden0)
            scores = impl.score_routes(unary, pairwise, routes, masks_by_target[target])
            chosen_indices = scores.argmax(dim=1)
            chosen_routes = routes.index_select(0, chosen_indices)
            hidden = shared.execute_grouped_routes(
                layers, adapter, hidden0, chosen_routes, candidates,
            )
            predicted = impl.logits_from_hidden(model, hidden).argmax(-1)
            full_predicted = teacher_logits.argmax(-1)
            dynamic_correct += int(predicted.eq(labels).sum())
            full_correct += int(full_predicted.eq(labels).sum())
            total += labels.numel()
            top2 = scores.topk(2, dim=1).values
            margin_sum += float((top2[:, 0] - top2[:, 1]).sum())
            probabilities = torch.softmax(scores.float(), dim=1).clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=1)
            entropy_sum += float((entropy / math.log(scores.size(1))).sum())
            for index, route in enumerate(chosen_routes.detach().cpu().tolist()):
                key = tuple([0] + [candidates[item] for item in route] + [len(layers) - 1])
                route_counter[key] += 1
                route_correct[key] += int(predicted[index].eq(labels[index]))

        rows.append({
            "method": method,
            "target_layers": target,
            "samples": total,
            "full_accuracy": full_correct / total,
            "dynamic_accuracy": dynamic_correct / total,
            "selected_route_count": len(route_counter),
            "max_route_share": max(route_counter.values()) / total,
            "mean_top1_top2_score_margin": margin_sum / total,
            "mean_normalized_score_entropy": entropy_sum / total,
        })
        for route, count in route_counter.most_common():
            usage_rows.append({
                "method": method,
                "target_layers": target,
                "route": str(list(route)),
                "count": count,
                "share": count / total,
                "conditional_accuracy": route_correct[route] / count,
            })
    return rows, usage_rows


def main(args):
    if not 0.0 <= args.load_prior_smoothing <= 1.0:
        raise ValueError("--load_prior_smoothing must be in [0, 1].")
    if args.top1_proxy_temperature <= 0.0:
        raise ValueError("--top1_proxy_temperature must be positive.")
    if len(set(args.load_balance_weights)) != len(args.load_balance_weights):
        raise ValueError("--load_balance_weights must not contain duplicates.")
    if any(weight < 0.0 for weight in args.load_balance_weights):
        raise ValueError("--load_balance_weights must be non-negative.")
    impl.set_seed(args.seed)
    impl.config.MODEL_ID = args.model_id
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"Checkpoint directory is not empty: {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reference = torch.load(args.reference_router_checkpoint, map_location="cpu", weights_only=False)
    adapter_checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    if int(adapter_checkpoint.get("epoch", -1)) != 15:
        raise ValueError("Expected an epoch-15 Adapter checkpoint.")
    if sorted(reference["target_layers"]) != sorted(args.target_layers):
        raise ValueError("Target budgets differ from the reference Router checkpoint.")
    saved_args = reference["args"]
    reference_adapter = reference.get("adapter_state_dict")
    if reference_adapter is None:
        raise ValueError("Reference Router checkpoint does not contain its Adapter state.")
    if set(reference_adapter) != set(adapter_checkpoint["adapter_state_dict"]):
        raise ValueError("Adapter parameter names differ from the reference checkpoint.")
    for key, value in adapter_checkpoint["adapter_state_dict"].items():
        if not torch.equal(value.cpu(), reference_adapter[key].cpu()):
            raise ValueError(f"Adapter differs from the reference checkpoint: {key}")

    model, transform = impl.load_model_and_processor()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model)
    candidates = list(range(1, len(layers) - 1))
    if candidates != reference["candidate_layers"]:
        raise ValueError("Candidate layers differ from the reference Router checkpoint.")
    dtype = next(model.parameters()).dtype
    targets = sorted(args.target_layers)
    routes_by_target = {
        target: impl.all_middle_routes(len(candidates), target - 2, impl.config.DEVICE)
        for target in targets
    }
    masks_by_target = {
        target: impl.pair_masks(len(candidates), routes, dtype)[1]
        for target, routes in routes_by_target.items()
    }
    train_loader, train_count = shared.make_loader(
        args.train_csv, transform, args.batch_size, args.max_train_samples,
        args.num_workers, True,
    )
    eval_loader, eval_count = shared.make_loader(
        args.eval_csv, transform, args.eval_batch_size, args.max_eval_samples,
        args.num_workers, False,
    )

    adapter = impl.DynamicAdapter(
        model.config.hidden_size, saved_args["adapter_bottleneck_dim"], len(layers), dtype,
    ).to(impl.config.DEVICE)
    adapter.load_state_dict(adapter_checkpoint["adapter_state_dict"])
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    reference_router = impl.InteractionRouteValueRouter(
        model.config.hidden_size, len(candidates), saved_args["router_hidden_dim"], dtype,
    ).to(impl.config.DEVICE)
    reference_router.load_state_dict(reference["router_state_dict"])
    reference_router.eval()

    # Every variant starts from exactly the same random initialization.
    template = impl.InteractionRouteValueRouter(
        model.config.hidden_size, len(candidates), saved_args["router_hidden_dim"], dtype,
    ).to(impl.config.DEVICE)
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    del template
    routers, optimizers = {}, {}
    for weight in args.load_balance_weights:
        name = variant_name(weight)
        router = impl.InteractionRouteValueRouter(
            model.config.hidden_size, len(candidates), saved_args["router_hidden_dim"], dtype,
        ).to(impl.config.DEVICE)
        router.load_state_dict(initial_state)
        router.train()
        routers[name] = router
        optimizers[name] = AdamW(router.parameters(), lr=args.router_learning_rate)

    metric_totals = {
        name: {"listwise": 0.0, "load": 0.0, "total": 0.0, "steps": 0}
        for name in routers
    }
    train_start = time.perf_counter()
    progress = tqdm(train_loader, desc="Load-balanced interaction Router epoch 1")
    for step, (images, labels) in enumerate(progress, 1):
        images = images.to(impl.config.DEVICE)
        labels = labels.to(impl.config.DEVICE)
        teacher_logits, teacher_hidden = shared.teacher_batch(model, images)
        hidden0 = teacher_hidden[1]
        target = random.choice(targets)
        all_routes = routes_by_target[target]
        count = min(args.routes_per_batch, len(all_routes))
        sampled_indices = torch.randperm(len(all_routes), device=impl.config.DEVICE)[:count]
        sampled_routes = all_routes.index_select(0, sampled_indices)
        route_losses = shared.route_losses(
            model, layers, adapter, hidden0, teacher_logits, labels,
            sampled_routes, candidates, args.route_kd_weight,
        )
        target_distribution = F.softmax(
            -route_losses / args.route_label_temperature, dim=-1,
        ).detach()
        sampled_mask = masks_by_target[target].index_select(0, sampled_indices)

        for weight in args.load_balance_weights:
            name = variant_name(weight)
            router = routers[name]
            optimizer = optimizers[name]
            unary, pairwise = router(hidden0)
            scores = impl.score_routes(unary, pairwise, sampled_routes, sampled_mask)
            listwise = -(target_distribution * F.log_softmax(scores, dim=-1)).sum(-1).mean()
            balance, predicted_load, prior = load_balance_loss(
                scores, target_distribution, args.top1_proxy_temperature,
                args.load_prior_smoothing,
            )
            total_loss = listwise + weight * balance
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), args.clip_grad_norm)
            optimizer.step()
            totals = metric_totals[name]
            totals["listwise"] += float(listwise.detach())
            totals["load"] += float(balance.detach())
            totals["total"] += float(total_loss.detach())
            totals["steps"] += 1

        if step % args.log_interval == 0:
            last_name = variant_name(args.load_balance_weights[-1])
            last = metric_totals[last_name]
            progress.set_postfix(
                target=f"{target}/{len(layers)}",
                loss=f"{last['total'] / last['steps']:.4f}",
                proxy_max=f"{float(predicted_load.max()):.3f}",
                prior_max=f"{float(prior.max()):.3f}",
            )
    train_seconds = time.perf_counter() - train_start

    history_rows = []
    for weight in args.load_balance_weights:
        name = variant_name(weight)
        router = routers[name]
        router.eval()
        totals = metric_totals[name]
        history_rows.append({
            "method": name,
            "epoch": 1,
            "load_balance_weight": weight,
            "listwise_loss": totals["listwise"] / totals["steps"],
            "load_balance_loss": totals["load"] / totals["steps"],
            "total_loss": totals["total"] / totals["steps"],
            "train_seconds_shared_route_loss_pass": train_seconds,
        })
        checkpoint_args = dict(vars(args))
        checkpoint_args["adapter_bottleneck_dim"] = saved_args["adapter_bottleneck_dim"]
        checkpoint_args["router_hidden_dim"] = saved_args["router_hidden_dim"]
        torch.save({
            "adapter_state_dict": adapter.state_dict(),
            "router_state_dict": router.state_dict(),
            "optimizer_state_dict": optimizers[name].state_dict(),
            "epoch": 1,
            "method": name,
            "load_balance_weight": weight,
            "load_prior_smoothing": args.load_prior_smoothing,
            "top1_proxy_temperature": args.top1_proxy_temperature,
            "target_layers": targets,
            "candidate_layers": candidates,
            "args": checkpoint_args,
        }, checkpoint_dir / f"{name}_router_epoch_01.pt")

    write_csv(output_dir / "training_history.csv", history_rows)

    accuracy_rows, usage_rows = [], []
    baseline_rows, baseline_usage = evaluate_router(
        model, layers, adapter, reference_router, eval_loader, routes_by_target,
        masks_by_target, candidates, "reference_epoch1_argmax",
    )
    accuracy_rows.extend(baseline_rows)
    usage_rows.extend(baseline_usage)
    for weight in args.load_balance_weights:
        name = variant_name(weight)
        rows, route_rows = evaluate_router(
            model, layers, adapter, routers[name], eval_loader, routes_by_target,
            masks_by_target, candidates, name,
        )
        accuracy_rows.extend(rows)
        usage_rows.extend(route_rows)

    write_csv(output_dir / "accuracy_summary.csv", accuracy_rows)
    write_csv(output_dir / "route_distribution.csv", usage_rows)
    metadata = [{
        "train_samples": train_count,
        "eval_samples": eval_count,
        "adapter_epoch": 15,
        "router_epochs": 1,
        "routes_per_batch": args.routes_per_batch,
        "load_balance_weights": str(args.load_balance_weights),
        "load_prior_smoothing": args.load_prior_smoothing,
        "top1_proxy_temperature": args.top1_proxy_temperature,
        "train_seconds_shared_route_loss_pass": train_seconds,
        "reference_router_checkpoint": args.reference_router_checkpoint,
        "adapter_checkpoint": args.adapter_checkpoint,
    }]
    write_csv(output_dir / "run_metadata.csv", metadata)
    print(f"Saved Router v2 results to {output_dir}", flush=True)
    print(f"Saved Router v2 checkpoints to {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--reference_router_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[4, 6, 7])
    parser.add_argument("--max_train_samples", type=int, default=40000)
    parser.add_argument("--max_eval_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--routes_per_batch", type=int, default=8)
    parser.add_argument("--router_learning_rate", type=float, default=1e-4)
    parser.add_argument("--route_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_label_temperature", type=float, default=1.0)
    parser.add_argument("--load_balance_weights", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0])
    parser.add_argument("--load_prior_smoothing", type=float, default=0.25)
    parser.add_argument("--top1_proxy_temperature", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
