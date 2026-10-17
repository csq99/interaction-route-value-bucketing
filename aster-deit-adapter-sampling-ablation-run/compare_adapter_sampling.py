"""Ablation: random full-route rollout vs balanced local-edge adapter training.

Both branches use the same frozen DeiT, data split, initial adapter checkpoint,
losses and fixed-route evaluation.  No router is trained in this experiment.
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

INTERACTION_DIR = Path(__file__).resolve().parents[1] / "aster-deit-interaction-route-value-run"
sys.path.insert(0, str(INTERACTION_DIR))
import train_interaction_route_value_deit as interaction_impl

config = interaction_impl.config
DynamicAdapter = interaction_impl.DynamicAdapter
ImageNetParquetSplitDataset = interaction_impl.ImageNetParquetSplitDataset
get_encoder_layers = interaction_impl.get_encoder_layers
load_model_and_processor = interaction_impl.load_model_and_processor
layer_hidden_state = interaction_impl.layer_hidden_state
logits_from_hidden = interaction_impl.logits_from_hidden


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(csv_path, transform, batch_size, max_samples, num_workers, shuffle):
    dataset = ImageNetParquetSplitDataset(
        csv_path, transform=transform,
        max_samples=None if max_samples <= 0 else max_samples,
    )
    kwargs = {"batch_size": batch_size, "shuffle": shuffle,
              "num_workers": num_workers, "pin_memory": False}
    if num_workers:
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def teacher_batch(model, images):
    out = model(images, output_hidden_states=True)
    return out.logits.detach(), tuple(h.detach() for h in out.hidden_states)


def run_route(model, layers, adapter, teacher_hidden, route):
    """Execute a complete route, returning student states after real layers."""
    state = teacher_hidden[0]
    previous = -1
    states = []
    for layer_index in route:
        if layer_index > previous + 1 and previous >= 0:
            state = adapter(state, previous, layer_index)
        state = layer_hidden_state(layers[layer_index](state))
        states.append((layer_index, state))
        previous = layer_index
    return states


def route_training_loss(model, layers, adapter, teacher_logits, teacher_hidden,
                        labels, route, args):
    states = run_route(model, layers, adapter, teacher_hidden, route)
    loss = torch.zeros((), device=labels.device, dtype=teacher_hidden[0].dtype)
    for layer_index, student_state in states:
        # hidden_states[k+1] is the output of encoder layer k.
        loss = loss + args.hidden_kd_weight * F.mse_loss(
            student_state, teacher_hidden[layer_index + 1])
    logits = logits_from_hidden(model, states[-1][1])
    ce = F.cross_entropy(logits, labels)
    kd = F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher_logits, -1), reduction="batchmean")
    return loss + args.ce_weight * ce + args.logit_kd_weight * kd


def edge_training_loss(model, layers, adapter, teacher_logits, teacher_hidden,
                       labels, curr, nxt, args):
    """Train one skip edge in a realistic prefix-to-tail context."""
    # Teacher-forced prefix state is the exact input at curr for this edge.
    state = teacher_hidden[curr]
    state = layer_hidden_state(layers[curr](state))
    adapted = adapter(state, curr, nxt)
    state = layer_hidden_state(layers[nxt](adapted))
    loss = args.hidden_kd_weight * F.mse_loss(state, teacher_hidden[nxt + 1])
    for layer_index in range(nxt + 1, len(layers)):
        state = layer_hidden_state(layers[layer_index](state))
        loss = loss + args.hidden_kd_weight * F.mse_loss(state, teacher_hidden[layer_index + 1])
    logits = logits_from_hidden(model, state)
    ce = F.cross_entropy(logits, labels)
    kd = F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher_logits, -1), reduction="batchmean")
    return loss + args.ce_weight * ce + args.logit_kd_weight * kd


def train_epoch(model, layers, adapter, loader, routes, edges, mode, args, optimizer, epoch):
    model.eval(); adapter.train(); running = 0.0
    progress = tqdm(loader, desc=f"{mode} epoch {epoch}")
    for step, (images, labels) in enumerate(progress, 1):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        teacher_logits, teacher_hidden = teacher_batch(model, images)
        if mode == "random_route":
            route = random.choice(routes)
            loss = route_training_loss(model, layers, adapter, teacher_logits,
                                       teacher_hidden, labels, route, args)
        else:
            # Deterministic shuffled edge order per epoch gives balanced coverage.
            start = ((step - 1) * args.edges_per_batch) % len(edges)
            selected = [edges[(start + j) % len(edges)] for j in range(args.edges_per_batch)]
            loss = torch.zeros((), device=config.DEVICE, dtype=teacher_hidden[0].dtype)
            for curr, nxt in selected:
                loss = loss + edge_training_loss(model, layers, adapter, teacher_logits,
                                                 teacher_hidden, labels, curr, nxt, args) / len(selected)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad_norm)
        optimizer.step(); running += float(loss.detach())
        if step % args.log_interval == 0:
            progress.set_postfix(loss=f"{float(loss):.4f}")
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate_fixed_routes(model, layers, adapter, loader, routes):
    model.eval(); adapter.eval(); stats = defaultdict(lambda: {"correct": 0, "total": 0, "ce": 0.0, "kd": 0.0})
    for images, labels in tqdm(loader, desc="Fixed-route evaluation"):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        teacher_logits, teacher_hidden = teacher_batch(model, images)
        for route in routes:
            states = run_route(model, layers, adapter, teacher_hidden, route)
            logits = logits_from_hidden(model, states[-1][1]); key = tuple(route); s = stats[key]
            s["correct"] += int(logits.argmax(-1).eq(labels).sum()); s["total"] += labels.numel()
            s["ce"] += float(F.cross_entropy(logits, labels, reduction="sum"))
            s["kd"] += float(F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher_logits, -1), reduction="none").sum())
    return [{"route": str(list(route)), "accuracy": s["correct"] / s["total"],
             "correct": s["correct"], "total": s["total"],
             "mean_ce": s["ce"] / s["total"], "mean_kd": s["kd"] / s["total"]}
            for route, s in stats.items()]


def build_adapter(model, args):
    dtype = next(model.parameters()).dtype
    adapter = DynamicAdapter(model.config.hidden_size, args.adapter_bottleneck_dim,
                             len(get_encoder_layers(model)), dtype).to(config.DEVICE)
    if args.init_adapter_checkpoint:
        ckpt = torch.load(args.init_adapter_checkpoint, map_location=config.DEVICE)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
    return adapter


def run_branch(mode, args, routes, edges, transform):
    set_seed(args.seed)
    model, _ = load_model_and_processor(); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    layers = get_encoder_layers(model)
    adapter = build_adapter(model, args)
    optimizer = AdamW(adapter.parameters(), lr=args.learning_rate)
    train_loader = make_loader(args.train_csv, transform, args.batch_size, args.max_train_samples, args.num_workers, True)
    eval_loader = make_loader(args.eval_csv, transform, args.eval_batch_size, args.max_eval_samples, args.num_workers, False)
    branch_dir = Path(args.output_dir) / mode; branch_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir) / mode; ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    eval_epochs = set(args.eval_epochs)
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            loss = train_epoch(model, layers, adapter, train_loader, routes, edges, mode, args, optimizer, epoch)
        else:
            loss = 0.0
        torch.save({"epoch": epoch, "adapter_state_dict": adapter.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "args": vars(args)},
                   ckpt_dir / f"adapter_epoch_{epoch:03d}.pt")
        if epoch in eval_epochs:
            rows = evaluate_fixed_routes(model, layers, adapter, eval_loader, routes)
            best = max(r["accuracy"] for r in rows)
            history.append((epoch, loss, best))
            if epoch == args.epochs:
                final_rows = rows
            print(f"[{mode}] epoch={epoch} loss={loss:.6f} best_fixed_accuracy={best:.4f}", flush=True)
    report = branch_dir / "report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write(f"# Adapter sampling ablation: `{mode}`\n\n")
        f.write(f"- Init checkpoint: `{args.init_adapter_checkpoint or 'random'}`\n")
        f.write(f"- Train/eval samples: `{args.max_train_samples}` / `{args.max_eval_samples}`\n")
        f.write(f"- Epochs: `{args.epochs}`; batch: `{args.batch_size}`\n\n")
        f.write("| Epoch | Train loss | Best fixed-route accuracy |\n|---:|---:|---:|\n")
        for e, l, b in history: f.write(f"| {e} | {l:.6f} | {b:.4f} |\n")
        f.write("\nFinal fixed-route details are in `final_routes.csv`.\n")
    if "final_rows" not in locals():
        final_rows = evaluate_fixed_routes(model, layers, adapter, eval_loader, routes)
    with (branch_dir / "final_routes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_rows[0].keys()); writer.writeheader(); writer.writerows(final_rows)
    with (branch_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["epoch", "train_loss", "best_fixed_route_accuracy"]); writer.writerows(history)


def main(args):
    config.MODEL_ID = args.model_id
    model, transform = load_model_and_processor(); layers = get_encoder_layers(model); del model
    # 2x target: first and final layers plus four middle layers.
    candidate_layers = list(range(1, len(layers) - 1)); middle_count = args.target_layers - 2
    import itertools
    routes = [[0] + [candidate_layers[i] for i in combo] + [len(layers) - 1]
              for combo in itertools.combinations(range(len(candidate_layers)), middle_count)]
    edges = [(c, n) for c in range(len(layers) - 2) for n in range(c + 2, len(layers))]
    for mode in ("random_route", "balanced_local_edge"):
        run_branch(mode, args, routes, edges, transform)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True); p.add_argument("--train_csv", required=True); p.add_argument("--eval_csv", required=True)
    p.add_argument("--target_layers", type=int, default=6); p.add_argument("--epochs", type=int, default=10); p.add_argument("--eval_epochs", type=int, nargs="+", default=[0, 1, 3, 5, 10])
    p.add_argument("--max_train_samples", type=int, default=2048); p.add_argument("--max_eval_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16); p.add_argument("--eval_batch_size", type=int, default=16); p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--edges_per_batch", type=int, default=4); p.add_argument("--adapter_bottleneck_dim", type=int, default=128); p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--hidden_kd_weight", type=float, default=1.0); p.add_argument("--ce_weight", type=float, default=10.0); p.add_argument("--logit_kd_weight", type=float, default=1.0); p.add_argument("--clip_grad_norm", type=float, default=1.0); p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--init_adapter_checkpoint", required=True); p.add_argument("--checkpoint_dir", default="./checkpoints_adapter_sampling_ablation"); p.add_argument("--output_dir", default="./results_adapter_sampling_ablation"); p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
