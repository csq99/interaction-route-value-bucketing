"""One shared random-route adapter and interaction router across BERT budgets."""

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import sys

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / "aster-bert-interaction-route-value-run"
sys.path.insert(0, str(SOURCE_DIR))
import train_interaction_route_value_bert as impl


def choose_route(routes_by_target):
    target_layers = random.choice(sorted(routes_by_target))
    return target_layers, random.choice(routes_by_target[target_layers])


def route_positions(routes, positions):
    return torch.tensor(
        [[positions[layer] for layer in route[1:-1]] for route in routes],
        device=impl.config.DEVICE,
        dtype=torch.long,
    )


def train_adapter_epoch(model, adapter, loader, routes_by_target, args, optimizer, epoch):
    model.eval()
    adapter.train()
    total_loss = 0.0
    progress = impl.tqdm(loader, desc=f"Shared random-route adapter epoch {epoch}")
    for step, batch in enumerate(progress, 1):
        input_ids = batch["input_ids"].to(impl.config.DEVICE)
        attention_mask = batch["attention_mask"].to(impl.config.DEVICE)
        labels = batch["labels"].to(impl.config.DEVICE)
        teacher_logits, _ = impl.teacher_batch(model, input_ids, attention_mask)
        hidden0, layer_attention_mask = impl.prepare_layer0(model, input_ids, attention_mask)
        target_layers, route = choose_route(routes_by_target)
        logits = impl.logits_from_hidden(
            model, impl.execute_route(model, adapter, hidden0, layer_attention_mask, route),
        )
        ce = F.cross_entropy(logits, labels)
        kd = F.kl_div(
            F.log_softmax(logits, dim=-1), F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
        loss = args.adapter_ce_weight * ce + args.adapter_kd_weight * kd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach())
        if step % args.log_interval == 0:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", target=f"{target_layers}/6", route=str(route))
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
    steps = min(len(loader), args.max_train_batches) if args.max_train_batches > 0 else len(loader)
    return total_loss / max(1, steps)


def train_router_epoch(model, adapter, router, loader, routes_by_target, positions,
                       masks_by_target, args, optimizer, epoch):
    model.eval()
    adapter.eval()
    router.train()
    total_loss = 0.0
    progress = impl.tqdm(loader, desc=f"Shared interaction router epoch {epoch}")
    for step, batch in enumerate(progress, 1):
        input_ids = batch["input_ids"].to(impl.config.DEVICE)
        attention_mask = batch["attention_mask"].to(impl.config.DEVICE)
        labels = batch["labels"].to(impl.config.DEVICE)
        teacher_logits, _ = impl.teacher_batch(model, input_ids, attention_mask)
        hidden0, layer_attention_mask = impl.prepare_layer0(model, input_ids, attention_mask)
        target_layers = random.choice(sorted(routes_by_target))
        legal_routes = routes_by_target[target_layers]
        count = min(args.routes_per_batch, len(legal_routes))
        selected = torch.randperm(len(legal_routes), device=impl.config.DEVICE)[:count]
        sampled_routes = [legal_routes[index] for index in selected.detach().cpu().tolist()]
        sampled_losses = impl.route_loss_matrix(
            model, adapter, hidden0, layer_attention_mask, teacher_logits, labels,
            sampled_routes, args.route_kd_weight,
        )
        unary, pairwise = router(hidden0)
        scores = impl.score_routes(
            unary, pairwise, route_positions(sampled_routes, positions),
            masks_by_target[target_layers].index_select(0, selected),
        )
        target_distribution = F.softmax(-sampled_losses / args.route_label_temperature, dim=-1)
        loss = -(target_distribution * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), args.clip_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach())
        if step % args.log_interval == 0:
            progress.set_postfix(listwise=f"{float(loss):.4f}", target=f"{target_layers}/6")
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
    steps = min(len(loader), args.max_train_batches) if args.max_train_batches > 0 else len(loader)
    return total_loss / max(1, steps)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all_targets(model, adapter, router, loader, routes_by_target, positions, masks_by_target):
    rows = []
    for target_layers, legal_routes in routes_by_target.items():
        fixed_rows = impl.evaluate_fixed_routes(model, adapter, loader, legal_routes)
        result = impl.evaluate_router(
            model, adapter, router, loader, legal_routes, positions, masks_by_target[target_layers],
        )
        best = max(fixed_rows, key=lambda row: row["accuracy"])
        route_count = len(result["routes"])
        max_share = max(result["routes"].values()) / result["total"]
        rows.append({
            "target_layers": target_layers,
            "theoretical_speedup": 6 / target_layers,
            "legal_routes": len(legal_routes),
            "full_accuracy": result["full_accuracy"],
            "best_fixed_accuracy": best["accuracy"],
            "best_fixed_route": best["route"],
            "router_accuracy": result["router_accuracy"],
            "selected_route_count": route_count,
            "max_route_share": max_share,
        })
    return rows


def main(args):
    impl.set_seed(args.seed)
    impl.config.MODEL_ID = impl.resolve_model_id(args.model_id)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = impl.load_model_and_tokenizer()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    datasets = impl.bert_impl.load_sst2(tokenizer)
    train_ds, eval_ds = datasets["train"], datasets["validation"]
    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    train_loader = impl.make_loader(train_ds, args.batch_size, True, args.seed)
    eval_loader = impl.make_loader(eval_ds, args.eval_batch_size, False, args.seed)

    num_layers = len(model.distilbert.transformer.layer)
    targets = sorted(set(args.target_layers))
    if any(target < 2 or target > num_layers for target in targets):
        raise ValueError(f"target_layers must be in [2, {num_layers}]")
    candidate_layers = list(range(1, num_layers - 1))
    positions = {layer: index for index, layer in enumerate(candidate_layers)}
    dtype = next(model.parameters()).dtype
    routes_by_target = {target: impl.complete_routes(num_layers, target) for target in targets}
    masks_by_target = {
        target: impl.pair_masks(candidate_layers, routes, impl.config.DEVICE, dtype)[1]
        for target, routes in routes_by_target.items()
    }
    adapter = impl.DynamicAdapter(
        model.config.hidden_size, args.adapter_bottleneck_dim, num_layers, dtype,
    ).to(impl.config.DEVICE)
    router = impl.InteractionRouteValueRouter(
        model.config.hidden_size, len(candidate_layers), args.router_hidden_dim, dtype,
    ).to(impl.config.DEVICE)

    adapter_optimizer = AdamW(adapter.parameters(), lr=args.adapter_learning_rate)
    adapter_history = []
    start = time.perf_counter()
    for epoch in range(args.adapter_epochs + 1):
        loss = 0.0 if epoch == 0 else train_adapter_epoch(
            model, adapter, train_loader, routes_by_target, args, adapter_optimizer, epoch,
        )
        if epoch in args.eval_epochs:
            torch.save({"epoch": epoch, "adapter_state_dict": adapter.state_dict(), "args": vars(args)},
                       checkpoint_dir / f"shared_adapter_epoch_{epoch:03d}.pt")
            print(f"[adapter] epoch={epoch} loss={loss:.6f}", flush=True)
        adapter_history.append({"epoch": epoch, "train_loss": loss})
    adapter_seconds = time.perf_counter() - start

    router_optimizer = AdamW(router.parameters(), lr=args.router_learning_rate)
    router_history = []
    start = time.perf_counter()
    for epoch in range(1, args.router_epochs + 1):
        loss = train_router_epoch(
            model, adapter, router, train_loader, routes_by_target, positions,
            masks_by_target, args, router_optimizer, epoch,
        )
        router_history.append({"epoch": epoch, "train_loss": loss})
        print(f"[router] epoch={epoch} loss={loss:.6f}", flush=True)
    router_seconds = time.perf_counter() - start

    results = evaluate_all_targets(model, adapter, router, eval_loader, routes_by_target, positions, masks_by_target)
    checkpoint_path = checkpoint_dir / "shared_interaction_router.pt"
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "router_state_dict": router.state_dict(),
        "target_layers": targets,
        "candidate_layers": candidate_layers,
        "args": vars(args),
    }, checkpoint_path)
    write_csv(output_dir / "adapter_history.csv", adapter_history)
    write_csv(output_dir / "router_history.csv", router_history)
    write_csv(output_dir / "accuracy_summary.csv", results)
    report = output_dir / "interaction_multibudget_bert.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Shared Multi-Budget Interaction Route-Value Router\n\n")
        handle.write("One adapter and one interaction router are trained once across all listed budgets. "
                     "Each training batch samples a budget uniformly, then samples a legal complete route within it.\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Train/eval samples: `{len(train_ds)}` / `{len(eval_ds)}`\n")
        handle.write(f"- Target layers: `{targets}` / `{num_layers}`\n")
        handle.write(f"- Shared adapter epochs: `{args.adapter_epochs}`; router epochs: `{args.router_epochs}`\n")
        handle.write(f"- Adapter wall time: `{adapter_seconds:.2f}` seconds; router wall time: `{router_seconds:.2f}` seconds\n\n")
        handle.write("## Accuracy and route diversity\n\n")
        handle.write("| Layers | Theory | Legal routes | Full acc | Best fixed | Dynamic acc | Selected routes | Max route share |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            handle.write(f"| {row['target_layers']}/6 | {row['theoretical_speedup']:.2f}x | {row['legal_routes']} | "
                         f"{row['full_accuracy']:.4f} | {row['best_fixed_accuracy']:.4f} | "
                         f"{row['router_accuracy']:.4f} | {row['selected_route_count']} | "
                         f"{row['max_route_share']:.4f} |\n")
        handle.write(f"\n- Checkpoint: `{checkpoint_path}`\n")
    print(f"Saved checkpoint: {checkpoint_path}\nSaved report: {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--target_layers", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--adapter_epochs", type=int, default=3)
    parser.add_argument("--router_epochs", type=int, default=1)
    parser.add_argument("--routes_per_batch", type=int, default=3)
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_bottleneck_dim", type=int, default=128)
    parser.add_argument("--adapter_learning_rate", type=float, default=5e-5)
    parser.add_argument("--router_learning_rate", type=float, default=1e-4)
    parser.add_argument("--adapter_ce_weight", type=float, default=10.0)
    parser.add_argument("--adapter_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_label_temperature", type=float, default=1.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_epochs", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--checkpoint_dir", default=str(HERE / "checkpoints_shared_multibudget"))
    parser.add_argument("--output_dir", default=str(HERE / "results_shared_multibudget"))
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
