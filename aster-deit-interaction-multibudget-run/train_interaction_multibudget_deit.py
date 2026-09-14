"""Shared random-route adapter and interaction router across DeiT budgets."""

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

SOURCE_DIR = Path(__file__).resolve().parents[1] / "aster-deit-interaction-route-value-run"
sys.path.insert(0, str(SOURCE_DIR))
import train_interaction_route_value_deit as impl


def make_loader(csv_path, transform, batch_size, max_samples, num_workers, shuffle):
    dataset = impl.ImageNetParquetSplitDataset(
        csv_path, transform=transform, max_samples=None if max_samples <= 0 else max_samples,
    )
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers, "pin_memory": False}
    if num_workers:
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs), len(dataset)


@torch.no_grad()
def teacher_batch(model, images):
    output = model(images, output_hidden_states=True)
    return output.logits.detach(), tuple(state.detach() for state in output.hidden_states)


def choose_budget_route(routes_by_target):
    target = random.choice(sorted(routes_by_target))
    index = torch.randint(len(routes_by_target[target]), (1,), device=impl.config.DEVICE).item()
    return target, routes_by_target[target][index]


def execute_route(layers, adapter, hidden0, middle_route, candidate_layers):
    state = hidden0
    previous = 0
    for candidate_index in middle_route.tolist():
        layer_index = candidate_layers[candidate_index]
        if layer_index > previous + 1:
            state = adapter(state, previous, layer_index)
        state = impl.layer_hidden_state(layers[layer_index](state))
        previous = layer_index
    final_index = len(layers) - 1
    if final_index > previous + 1:
        state = adapter(state, previous, final_index)
    return impl.layer_hidden_state(layers[final_index](state))


def execute_grouped_routes(layers, adapter, hidden0, middle_routes, candidate_layers):
    groups = defaultdict(list)
    for index, route in enumerate(middle_routes.detach().cpu().tolist()):
        groups[tuple(route)].append(index)
    output = torch.empty_like(hidden0)
    for route, indices in groups.items():
        index_tensor = torch.tensor(indices, device=hidden0.device, dtype=torch.long)
        result = execute_route(
            layers, adapter, hidden0.index_select(0, index_tensor),
            torch.tensor(route, device=hidden0.device, dtype=torch.long), candidate_layers,
        )
        output.index_copy_(0, index_tensor, result)
    return output


def route_training_loss(model, layers, adapter, teacher_logits, teacher_hidden, labels,
                        middle_route, candidate_layers, args):
    """CE + logit KD plus KD at each subsequently executed real layer."""
    state = teacher_hidden[1]
    previous = 0
    loss = torch.zeros((), device=state.device, dtype=state.dtype)
    for candidate_index in middle_route.tolist():
        layer_index = candidate_layers[candidate_index]
        if layer_index > previous + 1:
            state = adapter(state, previous, layer_index)
        state = impl.layer_hidden_state(layers[layer_index](state))
        loss = loss + args.hidden_kd_weight * F.mse_loss(state, teacher_hidden[layer_index + 1])
        previous = layer_index
    final_index = len(layers) - 1
    if final_index > previous + 1:
        state = adapter(state, previous, final_index)
    state = impl.layer_hidden_state(layers[final_index](state))
    loss = loss + args.hidden_kd_weight * F.mse_loss(state, teacher_hidden[final_index + 1])
    logits = impl.logits_from_hidden(model, state)
    ce = F.cross_entropy(logits, labels)
    kd = F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher_logits, -1), reduction="batchmean")
    return loss + args.adapter_ce_weight * ce + args.adapter_logit_kd_weight * kd


def route_losses(model, layers, adapter, hidden0, teacher_logits, labels, routes, candidate_layers, kd_weight):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, -1)
        losses = []
        for route in routes:
            logits = impl.logits_from_hidden(model, execute_route(layers, adapter, hidden0, route, candidate_layers))
            ce = F.cross_entropy(logits, labels, reduction="none")
            kd = F.kl_div(F.log_softmax(logits, -1), teacher_probs, reduction="none").sum(-1)
            losses.append(ce + kd_weight * kd)
    return torch.stack(losses, dim=1)


def adapter_epoch(model, layers, adapter, loader, routes_by_target, candidate_layers, args, optimizer, epoch):
    model.eval(); adapter.train(); total = 0.0
    progress = tqdm(loader, desc=f"Shared random-route adapter epoch {epoch}")
    for step, (images, labels) in enumerate(progress, 1):
        images, labels = images.to(impl.config.DEVICE), labels.to(impl.config.DEVICE)
        teacher_logits, teacher_hidden = teacher_batch(model, images)
        target, route = choose_budget_route(routes_by_target)
        loss = route_training_loss(model, layers, adapter, teacher_logits, teacher_hidden, labels,
                                   route, candidate_layers, args)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad_norm)
        optimizer.step(); total += float(loss.detach())
        if step % args.log_interval == 0:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", target=f"{target}/{len(layers)}")
    return total / max(1, len(loader))


def router_epoch(model, layers, adapter, router, loader, routes_by_target, masks_by_target,
                 candidate_layers, args, optimizer, epoch):
    model.eval(); adapter.eval(); router.train(); total = 0.0
    progress = tqdm(loader, desc=f"Shared interaction router epoch {epoch}")
    for step, (images, labels) in enumerate(progress, 1):
        images, labels = images.to(impl.config.DEVICE), labels.to(impl.config.DEVICE)
        teacher_logits, teacher_hidden = teacher_batch(model, images)
        hidden0 = teacher_hidden[1]
        target = random.choice(sorted(routes_by_target))
        routes = routes_by_target[target]
        count = min(args.routes_per_batch, len(routes))
        sampled_indices = torch.randperm(len(routes), device=impl.config.DEVICE)[:count]
        sampled_routes = routes.index_select(0, sampled_indices)
        losses = route_losses(model, layers, adapter, hidden0, teacher_logits, labels, sampled_routes,
                              candidate_layers, args.route_kd_weight)
        unary, pairwise = router(hidden0)
        scores = impl.score_routes(unary, pairwise, sampled_routes,
                                   masks_by_target[target].index_select(0, sampled_indices))
        target_distribution = F.softmax(-losses / args.route_label_temperature, dim=-1)
        loss = -(target_distribution * F.log_softmax(scores, dim=-1)).sum(-1).mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), args.clip_grad_norm)
        optimizer.step(); total += float(loss.detach())
        if step % args.log_interval == 0:
            progress.set_postfix(listwise=f"{float(loss.detach()):.4f}", target=f"{target}/{len(layers)}")
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_target(model, layers, adapter, router, loader, routes, pair_mask, candidate_layers):
    model.eval(); adapter.eval(); router.eval()
    full_correct = dynamic_correct = total = 0
    route_counter, route_correct = Counter(), Counter()
    for images, labels in tqdm(loader, desc=f"Dynamic evaluation ({routes.size(1) + 2}/12)"):
        images, labels = images.to(impl.config.DEVICE), labels.to(impl.config.DEVICE)
        teacher_logits, teacher_hidden = teacher_batch(model, images)
        hidden0 = teacher_hidden[1]
        unary, pairwise = router(hidden0)
        route_indices = impl.score_routes(unary, pairwise, routes, pair_mask).argmax(dim=1)
        selected = routes.index_select(0, route_indices)
        state = execute_grouped_routes(layers, adapter, hidden0, selected, candidate_layers)
        predicted, full_predicted = impl.logits_from_hidden(model, state).argmax(-1), teacher_logits.argmax(-1)
        full_correct += int(full_predicted.eq(labels).sum())
        dynamic_correct += int(predicted.eq(labels).sum())
        total += labels.numel()
        for index, route in enumerate(selected.detach().cpu().tolist()):
            key = tuple([0] + [candidate_layers[item] for item in route] + [len(layers) - 1])
            route_counter[key] += 1
            route_correct[key] += int(predicted[index].eq(labels[index]))
    return {"full_accuracy": full_correct / total, "dynamic_accuracy": dynamic_correct / total,
            "full_correct": full_correct, "dynamic_correct": dynamic_correct, "total": total,
            "routes": route_counter, "route_accuracy": {key: route_correct[key] / count for key, count in route_counter.items()}}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main(args):
    impl.set_seed(args.seed)
    impl.config.MODEL_ID = args.model_id
    output_dir, checkpoint_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model, transform = impl.load_model_and_processor(); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model); num_layers = len(layers)
    targets = sorted(set(args.target_layers))
    if any(target < 3 or target > num_layers for target in targets):
        raise ValueError(f"target_layers must be in [3, {num_layers}]")
    candidate_layers = list(range(1, num_layers - 1))
    dtype = next(model.parameters()).dtype
    routes_by_target = {target: impl.all_middle_routes(len(candidate_layers), target - 2, impl.config.DEVICE) for target in targets}
    masks_by_target = {target: impl.pair_masks(len(candidate_layers), routes, dtype)[1] for target, routes in routes_by_target.items()}
    train_loader, train_count = make_loader(args.train_csv, transform, args.batch_size, args.max_train_samples, args.num_workers, True)
    eval_loader, eval_count = make_loader(args.eval_csv, transform, args.eval_batch_size, args.max_eval_samples, args.num_workers, False)
    if args.init_adapter_checkpoint and args.resume_adapter_checkpoint:
        raise ValueError("Use either --init_adapter_checkpoint or --resume_adapter_checkpoint, not both.")
    adapter = impl.DynamicAdapter(model.config.hidden_size, args.adapter_bottleneck_dim, num_layers, dtype).to(impl.config.DEVICE)
    if args.init_adapter_checkpoint:
        state = torch.load(args.init_adapter_checkpoint, map_location=impl.config.DEVICE, weights_only=False)
        adapter.load_state_dict(state["adapter_state_dict"])
    router = impl.InteractionRouteValueRouter(model.config.hidden_size, len(candidate_layers), args.router_hidden_dim, dtype).to(impl.config.DEVICE)

    adapter_optimizer = AdamW(adapter.parameters(), lr=args.adapter_learning_rate)
    start_epoch = 0
    if args.resume_adapter_checkpoint:
        state = torch.load(args.resume_adapter_checkpoint, map_location=impl.config.DEVICE, weights_only=False)
        adapter.load_state_dict(state["adapter_state_dict"])
        if "optimizer_state_dict" not in state:
            raise ValueError("Resume checkpoint lacks optimizer_state_dict; use it only as --init_adapter_checkpoint.")
        adapter_optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"])
        if args.adapter_epochs <= start_epoch:
            raise ValueError("--adapter_epochs must exceed the resume checkpoint epoch.")
    adapter_history = []
    adapter_start = time.perf_counter()
    if start_epoch == 0:
        epochs = range(0, args.adapter_epochs + 1)
    else:
        epochs = range(start_epoch + 1, args.adapter_epochs + 1)
    for epoch in epochs:
        loss = 0.0 if epoch == 0 else adapter_epoch(model, layers, adapter, train_loader, routes_by_target,
                                                     candidate_layers, args, adapter_optimizer, epoch)
        adapter_history.append({"epoch": epoch, "train_loss": loss})
        if epoch in args.save_epochs:
            torch.save({"epoch": epoch, "adapter_state_dict": adapter.state_dict(),
                        "optimizer_state_dict": adapter_optimizer.state_dict(), "args": vars(args)},
                       checkpoint_dir / f"shared_adapter_epoch_{epoch:03d}.pt")
        print(f"[adapter] epoch={epoch} loss={loss:.6f}", flush=True)
    adapter_seconds = time.perf_counter() - adapter_start

    router_optimizer = AdamW(router.parameters(), lr=args.router_learning_rate)
    router_history = []
    router_start = time.perf_counter()
    for epoch in range(1, args.router_epochs + 1):
        loss = router_epoch(model, layers, adapter, router, train_loader, routes_by_target, masks_by_target,
                            candidate_layers, args, router_optimizer, epoch)
        router_history.append({"epoch": epoch, "train_loss": loss})
        print(f"[router] epoch={epoch} loss={loss:.6f}", flush=True)
    router_seconds = time.perf_counter() - router_start

    summary, route_rows = [], []
    for target in targets:
        result = evaluate_target(model, layers, adapter, router, eval_loader, routes_by_target[target],
                                 masks_by_target[target], candidate_layers)
        max_share = max(result["routes"].values()) / result["total"]
        summary.append({"target_layers": target, "theoretical_speedup": num_layers / target,
                        "legal_routes": len(routes_by_target[target]), "full_accuracy": result["full_accuracy"],
                        "dynamic_accuracy": result["dynamic_accuracy"], "selected_route_count": len(result["routes"]),
                        "max_route_share": max_share})
        for route, count in result["routes"].most_common():
            route_rows.append({"target_layers": target, "route": str(list(route)), "count": count,
                               "share": count / result["total"], "conditional_accuracy": result["route_accuracy"][route]})
    checkpoint_path = checkpoint_dir / "shared_interaction_router.pt"
    torch.save({"adapter_state_dict": adapter.state_dict(), "router_state_dict": router.state_dict(),
                "target_layers": targets, "candidate_layers": candidate_layers, "args": vars(args)}, checkpoint_path)
    write_csv(output_dir / "adapter_history.csv", adapter_history)
    write_csv(output_dir / "router_history.csv", router_history)
    write_csv(output_dir / "accuracy_summary.csv", summary)
    write_csv(output_dir / "dynamic_route_distribution.csv", route_rows)
    report = output_dir / "interaction_multibudget_deit.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Shared Multi-Budget Interaction Route-Value Router for DeiT\n\n")
        handle.write("A single adapter and a single unary-plus-pairwise router are trained once across the listed budgets. "
                     "Each batch samples a budget uniformly, then a random legal complete route.\n\n## Setup\n\n")
        handle.write(f"- Train/eval samples: `{train_count}` / `{eval_count}`\n")
        handle.write(f"- Targets: `{targets}` / `{num_layers}`; legal-route counts: `{ {target: len(routes_by_target[target]) for target in targets} }`\n")
        handle.write(f"- Adapter epochs: `{args.adapter_epochs}`; router epochs: `{args.router_epochs}`\n")
        handle.write(f"- Initial adapter: `{args.init_adapter_checkpoint or 'random initialization'}`\n")
        handle.write(f"- Resumed adapter: `{args.resume_adapter_checkpoint or 'no'}`\n")
        handle.write(f"- Adapter wall time: `{adapter_seconds:.2f}` seconds; router wall time: `{router_seconds:.2f}` seconds\n\n")
        handle.write("## Dynamic Accuracy and Route Diversity\n\n| Budget | Theory | Legal routes | Full acc | Dynamic acc | Selected routes | Max route share |\n|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            handle.write(f"| {row['target_layers']}/12 | {row['theoretical_speedup']:.2f}x | {row['legal_routes']} | {row['full_accuracy']:.4f} | {row['dynamic_accuracy']:.4f} | {row['selected_route_count']} | {row['max_route_share']:.4f} |\n")
        handle.write(f"\n- Checkpoint: `{checkpoint_path}`\n")
    print(f"Saved checkpoint: {checkpoint_path}\nSaved report: {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True); parser.add_argument("--train_csv", required=True); parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[4, 6, 7])
    parser.add_argument("--max_train_samples", type=int, default=2048); parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16); parser.add_argument("--eval_batch_size", type=int, default=16); parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--adapter_epochs", type=int, default=10); parser.add_argument("--router_epochs", type=int, default=1); parser.add_argument("--routes_per_batch", type=int, default=8)
    parser.add_argument("--router_hidden_dim", type=int, default=256); parser.add_argument("--adapter_bottleneck_dim", type=int, default=128)
    parser.add_argument("--adapter_learning_rate", type=float, default=5e-5); parser.add_argument("--router_learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_kd_weight", type=float, default=1.0); parser.add_argument("--adapter_ce_weight", type=float, default=10.0); parser.add_argument("--adapter_logit_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_kd_weight", type=float, default=1.0); parser.add_argument("--route_label_temperature", type=float, default=1.0); parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--init_adapter_checkpoint"); parser.add_argument("--resume_adapter_checkpoint")
    parser.add_argument("--save_epochs", type=int, nargs="+", default=[0, 1, 3, 5, 10]); parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", required=True); parser.add_argument("--output_dir", required=True); parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
