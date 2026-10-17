"""Joint Adapter + interaction route-value Router training for DeiT.

The backbone is frozen. Each step updates the Adapter with one uniformly
sampled complete route and updates the Router with detached route-quality
labels computed on a fresh candidate set. Random route coverage prevents the
Adapter from receiving gradients only through the Router's current favorite.
"""

import argparse
import csv
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
        csv_path, transform=transform,
        max_samples=None if max_samples <= 0 else max_samples,
    )
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers,
              "pin_memory": False}
    if num_workers:
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs), len(dataset)


def teacher_batch(model, images):
    with torch.no_grad():
        output = model(images, output_hidden_states=True)
    return output.logits.detach(), tuple(state.detach() for state in output.hidden_states)


def choose_route(routes_by_target):
    target = random.choice(sorted(routes_by_target))
    routes = routes_by_target[target]
    index = torch.randint(len(routes), (1,), device=impl.config.DEVICE).item()
    return target, routes[index]


def execute_route(layers, adapter, hidden0, route, candidates):
    state = hidden0
    previous = 0
    for candidate_index in route.tolist():
        layer_index = candidates[candidate_index]
        if layer_index > previous + 1:
            state = adapter(state, previous, layer_index)
        state = impl.layer_hidden_state(layers[layer_index](state))
        previous = layer_index
    final_index = len(layers) - 1
    if final_index > previous + 1:
        state = adapter(state, previous, final_index)
    return impl.layer_hidden_state(layers[final_index](state))


def execute_grouped_routes(layers, adapter, hidden0, routes, candidates):
    groups = defaultdict(list)
    for index, route in enumerate(routes.detach().cpu().tolist()):
        groups[tuple(route)].append(index)
    output = torch.empty_like(hidden0)
    for route, indices in groups.items():
        index_tensor = torch.tensor(indices, device=hidden0.device)
        state = execute_route(
            layers, adapter, hidden0.index_select(0, index_tensor),
            torch.tensor(route, device=hidden0.device), candidates,
        )
        output.index_copy_(0, index_tensor, state)
    return output


def adapter_loss(model, layers, adapter, teacher_logits, teacher_hidden, labels,
                 route, candidates, args):
    """CE + final logit KD + KD after every subsequently executed real layer."""
    state = teacher_hidden[1]
    previous = 0
    hidden_kd = torch.zeros((), device=state.device, dtype=state.dtype)
    for candidate_index in route.tolist():
        layer_index = candidates[candidate_index]
        if layer_index > previous + 1:
            state = adapter(state, previous, layer_index)
        state = impl.layer_hidden_state(layers[layer_index](state))
        hidden_kd = hidden_kd + F.mse_loss(state, teacher_hidden[layer_index + 1])
        previous = layer_index
    final_index = len(layers) - 1
    if final_index > previous + 1:
        state = adapter(state, previous, final_index)
    state = impl.layer_hidden_state(layers[final_index](state))
    logits = impl.logits_from_hidden(model, state)
    ce = F.cross_entropy(logits, labels)
    logit_kd = F.kl_div(
        F.log_softmax(logits, dim=-1), F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    )
    return (args.hidden_kd_weight * hidden_kd
            + args.adapter_ce_weight * ce
            + args.adapter_logit_kd_weight * logit_kd), state


def route_quality(model, layers, adapter, hidden0, teacher_logits, labels,
                  routes, candidates, kd_weight):
    """Return detached CE+logit-KD quality labels for a candidate route set."""
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        values = []
        for route in routes:
            logits = impl.logits_from_hidden(
                model, execute_route(layers, adapter, hidden0, route, candidates),
            )
            ce = F.cross_entropy(logits, labels, reduction="none")
            kd = F.kl_div(
                F.log_softmax(logits, dim=-1), teacher_probs,
                reduction="none",
            ).sum(dim=-1)
            values.append(ce + kd_weight * kd)
        return torch.stack(values, dim=1)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train_epoch(model, layers, adapter, router, loader, routes_by_target,
                masks_by_target, candidates, args, adapter_optimizer,
                router_optimizer, epoch):
    model.eval(); adapter.train(); router.train()
    total_adapter = total_router = total_steps = 0.0
    progress = tqdm(loader, desc=f"Joint interaction epoch {epoch}")
    for step, (images, labels) in enumerate(progress, 1):
        images = images.to(impl.config.DEVICE)
        labels = labels.to(impl.config.DEVICE)
        teacher_logits, teacher_hidden = teacher_batch(model, images)
        hidden0 = teacher_hidden[1]

        # Adapter coverage path. This is independent of the Router decision.
        _, coverage_route = choose_route(routes_by_target)
        adapter_total, _ = adapter_loss(
            model, layers, adapter, teacher_logits, teacher_hidden, labels,
            coverage_route, candidates, args,
        )

        # Router supervision uses the current Adapter, but its quality labels
        # are detached so the Router loss cannot update the Adapter implicitly.
        target = random.choice(sorted(routes_by_target))
        all_routes = routes_by_target[target]
        count = min(args.routes_per_batch, len(all_routes))
        sampled_indices = torch.randperm(len(all_routes), device=impl.config.DEVICE)[:count]
        sampled_routes = all_routes.index_select(0, sampled_indices)
        sampled_mask = masks_by_target[target].index_select(0, sampled_indices)
        qualities = route_quality(
            model, layers, adapter, hidden0, teacher_logits, labels,
            sampled_routes, candidates, args.route_kd_weight,
        )
        target_distribution = F.softmax(
            -qualities / args.route_label_temperature, dim=-1,
        )
        unary, pairwise = router(hidden0)
        scores = impl.score_routes(unary, pairwise, sampled_routes, sampled_mask)
        router_total = -(
            target_distribution * F.log_softmax(scores, dim=-1)
        ).sum(dim=-1).mean()

        adapter_optimizer.zero_grad()
        adapter_total.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad_norm)
        adapter_optimizer.step()

        router_optimizer.zero_grad()
        router_total.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), args.clip_grad_norm)
        router_optimizer.step()

        total_adapter += float(adapter_total.detach())
        total_router += float(router_total.detach())
        total_steps += 1
        if step % args.log_interval == 0:
            progress.set_postfix(
                adapter=f"{total_adapter / total_steps:.4f}",
                router=f"{total_router / total_steps:.4f}",
            )
    return total_adapter / max(1.0, total_steps), total_router / max(1.0, total_steps)


@torch.no_grad()
def evaluate(model, layers, adapter, router, loader, routes_by_target,
             masks_by_target, candidates):
    model.eval(); adapter.eval(); router.eval()
    rows, usage_rows = [], []
    for target in sorted(routes_by_target):
        routes = routes_by_target[target]
        pair_mask = masks_by_target[target]
        full_correct = dynamic_correct = total = 0
        route_counter, route_correct = Counter(), Counter()
        for images, labels in tqdm(loader, desc=f"Joint evaluation ({target}/12)"):
            images = images.to(impl.config.DEVICE)
            labels = labels.to(impl.config.DEVICE)
            teacher_logits, teacher_hidden = teacher_batch(model, images)
            hidden0 = teacher_hidden[1]
            unary, pairwise = router(hidden0)
            selected = routes.index_select(
                0, impl.score_routes(unary, pairwise, routes, pair_mask).argmax(dim=1),
            )
            hidden = execute_grouped_routes(layers, adapter, hidden0, selected, candidates)
            predicted = impl.logits_from_hidden(model, hidden).argmax(dim=-1)
            full_predicted = teacher_logits.argmax(dim=-1)
            dynamic_correct += int(predicted.eq(labels).sum())
            full_correct += int(full_predicted.eq(labels).sum())
            total += labels.numel()
            for index, route in enumerate(selected.detach().cpu().tolist()):
                key = tuple([0] + [candidates[item] for item in route] + [len(layers) - 1])
                route_counter[key] += 1
                route_correct[key] += int(predicted[index].eq(labels[index]))
        rows.append({
            "target_layers": target, "samples": total,
            "full_accuracy": full_correct / total,
            "dynamic_accuracy": dynamic_correct / total,
            "selected_route_count": len(route_counter),
            "max_route_share": max(route_counter.values()) / total,
        })
        for route, count in route_counter.most_common():
            usage_rows.append({
                "target_layers": target, "route": str(list(route)), "count": count,
                "share": count / total,
                "conditional_accuracy": route_correct[route] / count,
            })
    return rows, usage_rows


def main(args):
    if args.resume_checkpoint and (args.init_adapter_checkpoint or args.init_router_checkpoint):
        raise ValueError("Use --resume_checkpoint alone, or initialize from separate checkpoints.")
    if bool(args.init_adapter_checkpoint) != bool(args.init_router_checkpoint):
        raise ValueError("Provide both --init_adapter_checkpoint and --init_router_checkpoint.")
    if args.epochs < 1 or args.routes_per_batch < 2:
        raise ValueError("--epochs must be positive and --routes_per_batch must be at least 2.")
    impl.set_seed(args.seed)
    impl.config.MODEL_ID = args.model_id
    output_dir, checkpoint_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model, transform = impl.load_model_and_processor()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model)
    num_layers = len(layers)
    candidates = list(range(1, num_layers - 1))
    targets = sorted(set(args.target_layers))
    routes_by_target = {
        target: impl.all_middle_routes(len(candidates), target - 2, impl.config.DEVICE)
        for target in targets
    }
    masks_by_target = {
        target: impl.pair_masks(len(candidates), routes, next(model.parameters()).dtype)[1]
        for target, routes in routes_by_target.items()
    }
    train_loader, train_count = make_loader(
        args.train_csv, transform, args.batch_size, args.max_train_samples, args.num_workers, True,
    )
    eval_loader, eval_count = make_loader(
        args.eval_csv, transform, args.eval_batch_size, args.max_eval_samples, args.num_workers, False,
    )
    dtype = next(model.parameters()).dtype
    adapter = impl.DynamicAdapter(model.config.hidden_size, args.adapter_bottleneck_dim,
                                  num_layers, dtype).to(impl.config.DEVICE)
    router = impl.InteractionRouteValueRouter(model.config.hidden_size, len(candidates),
                                               args.router_hidden_dim, dtype).to(impl.config.DEVICE)
    if args.init_adapter_checkpoint:
        state = torch.load(args.init_adapter_checkpoint, map_location=impl.config.DEVICE,
                           weights_only=False)
        adapter.load_state_dict(state["adapter_state_dict"])
    if args.init_router_checkpoint:
        state = torch.load(args.init_router_checkpoint, map_location=impl.config.DEVICE,
                           weights_only=False)
        router.load_state_dict(state["router_state_dict"])
    adapter_optimizer = AdamW(adapter.parameters(), lr=args.adapter_learning_rate)
    router_optimizer = AdamW(router.parameters(), lr=args.router_learning_rate)
    start_epoch = 0
    if args.resume_checkpoint:
        state = torch.load(args.resume_checkpoint, map_location=impl.config.DEVICE,
                           weights_only=False)
        adapter.load_state_dict(state["adapter_state_dict"])
        router.load_state_dict(state["router_state_dict"])
        adapter_optimizer.load_state_dict(state["adapter_optimizer_state_dict"])
        router_optimizer.load_state_dict(state["router_optimizer_state_dict"])
        start_epoch = int(state["epoch"])
        if args.epochs <= start_epoch:
            raise ValueError("--epochs is the desired total epoch; it must exceed resume epoch.")
    history = []
    accuracy_rows, usage_rows = [], []

    def evaluate_and_save(epoch):
        rows, usage = evaluate(model, layers, adapter, router, eval_loader,
                               routes_by_target, masks_by_target, candidates)
        accuracy_rows.extend({"epoch": epoch, **row} for row in rows)
        usage_rows.extend({"epoch": epoch, **row} for row in usage)
        write_csv(output_dir / "accuracy_curve.csv", accuracy_rows)
        write_csv(output_dir / "route_distribution_curve.csv", usage_rows)

    evaluate_and_save(start_epoch)
    start = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        adapter_loss_value, router_loss_value = train_epoch(
            model, layers, adapter, router, train_loader, routes_by_target,
            masks_by_target, candidates, args, adapter_optimizer, router_optimizer, epoch,
        )
        history.append({"epoch": epoch, "adapter_loss": adapter_loss_value,
                       "router_loss": router_loss_value})
        checkpoint = {
            "epoch": epoch, "adapter_state_dict": adapter.state_dict(),
            "router_state_dict": router.state_dict(),
            "adapter_optimizer_state_dict": adapter_optimizer.state_dict(),
            "router_optimizer_state_dict": router_optimizer.state_dict(),
            "target_layers": targets, "candidate_layers": candidates,
            "args": vars(args),
        }
        torch.save(checkpoint, checkpoint_dir / f"joint_epoch_{epoch:03d}.pt")
        print(f"[joint] epoch={epoch} adapter_loss={adapter_loss_value:.6f} "
              f"router_loss={router_loss_value:.6f}", flush=True)
        evaluate_and_save(epoch)
    elapsed = time.perf_counter() - start
    write_csv(output_dir / "training_history.csv", history)
    write_csv(output_dir / "run_metadata.csv", [{
        "train_samples": train_count, "eval_samples": eval_count,
        "epochs": args.epochs, "target_layers": str(targets),
        "wall_time_seconds": elapsed,
        "init_adapter_checkpoint": args.init_adapter_checkpoint or "random",
        "init_router_checkpoint": args.init_router_checkpoint or "random",
    }])
    torch.save({
        "epoch": args.epochs, "adapter_state_dict": adapter.state_dict(),
        "router_state_dict": router.state_dict(),
        "adapter_optimizer_state_dict": adapter_optimizer.state_dict(),
        "router_optimizer_state_dict": router_optimizer.state_dict(),
        "target_layers": targets, "candidate_layers": candidates,
        "args": vars(args),
    }, checkpoint_dir / "joint_final.pt")
    print(f"Saved joint results to {output_dir}", flush=True)
    print(f"Saved joint checkpoints to {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[4, 6, 7])
    parser.add_argument("--max_train_samples", type=int, default=40000)
    parser.add_argument("--max_eval_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--routes_per_batch", type=int, default=8)
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_bottleneck_dim", type=int, default=128)
    parser.add_argument("--adapter_learning_rate", type=float, default=5e-5)
    parser.add_argument("--router_learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_kd_weight", type=float, default=1.0)
    parser.add_argument("--adapter_ce_weight", type=float, default=10.0)
    parser.add_argument("--adapter_logit_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_label_temperature", type=float, default=1.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--init_adapter_checkpoint")
    parser.add_argument("--init_router_checkpoint")
    parser.add_argument("--resume_checkpoint")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
