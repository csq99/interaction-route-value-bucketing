import argparse
import csv
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from components import DynamicAdapter
from deit_trainer import ImageNetParquetDataset, ImageNetParquetSplitDataset
from model_utils import get_encoder_layers, load_model_and_processor, run_embeddings


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def layer_hidden_state(layer_output):
    return layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output


def logits_from_hidden(model, hidden_state):
    return model.classifier(hidden_state[:, 0])


def target_layers_from_speedup(num_layers, speedup):
    return max(2, min(num_layers, int(round(num_layers / speedup))))


def make_dataloaders(image_transform, args):
    train_max = None if args.max_train_samples is not None and args.max_train_samples <= 0 else args.max_train_samples
    eval_max = None if args.max_eval_samples is not None and args.max_eval_samples <= 0 else args.max_eval_samples
    if args.train_csv:
        train_dataset = ImageNetParquetSplitDataset(args.train_csv, transform=image_transform, max_samples=train_max)
    else:
        train_dataset = ImageNetParquetDataset(
            args.imagenet_parquet_dir,
            split=args.train_split,
            transform=image_transform,
            max_samples=train_max,
        )
    if args.eval_csv:
        eval_dataset = ImageNetParquetSplitDataset(args.eval_csv, transform=image_transform, max_samples=eval_max)
    else:
        eval_dataset = ImageNetParquetDataset(
            args.imagenet_parquet_dir,
            split=args.eval_split,
            transform=image_transform,
            max_samples=eval_max,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader, len(train_dataset), len(eval_dataset)


def cls_cosine_delta(h_in, h_out):
    return 1.0 - F.cosine_similarity(h_in[:, 0, :], h_out[:, 0, :], dim=-1, eps=1e-8)


def topk_multilabel_targets(labels, k):
    targets = torch.zeros_like(labels)
    if k <= 0:
        return targets
    top_pos = torch.topk(labels, k=k, dim=1).indices
    targets.scatter_(1, top_pos, 1.0)
    return targets


class EarlyClsMiddlePredictor(nn.Module):
    def __init__(self, hidden_dim, predictor_hidden_dim, candidate_layers, dtype):
        super().__init__()
        layer_emb_dim = max(16, hidden_dim // 4)
        self.candidate_layers = list(candidate_layers)
        self.layer_embedding = nn.Embedding(
            len(self.candidate_layers),
            layer_emb_dim,
            device=config.DEVICE,
            dtype=dtype,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + layer_emb_dim, predictor_hidden_dim, device=config.DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim // 2, device=config.DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim // 2, 1, device=config.DEVICE, dtype=dtype),
        )

    def forward_logits(self, early_cls):
        batch_size = early_cls.shape[0]
        layer_pos = torch.arange(len(self.candidate_layers), device=early_cls.device)
        layer_emb = self.layer_embedding(layer_pos)
        cls_expand = early_cls.unsqueeze(1).expand(batch_size, len(self.candidate_layers), early_cls.shape[-1])
        layer_expand = layer_emb.unsqueeze(0).expand(batch_size, len(self.candidate_layers), layer_emb.shape[-1])
        x = torch.cat([cls_expand, layer_expand], dim=-1)
        return self.mlp(x).squeeze(-1)

    def forward(self, early_cls):
        return self.forward_logits(early_cls)


@torch.no_grad()
def compute_early_cls_and_labels(model, encoder_layers, images, candidate_layers):
    embeddings = run_embeddings(model, images)
    layer0_out = layer_hidden_state(encoder_layers[0](embeddings))
    hidden = embeddings
    labels = []
    for layer_idx, layer in enumerate(encoder_layers):
        next_hidden = layer_hidden_state(layer(hidden))
        if layer_idx in candidate_layers:
            labels.append(cls_cosine_delta(hidden, next_hidden))
        hidden = next_hidden
    return layer0_out[:, 0, :].detach(), torch.stack(labels, dim=1).detach()


def select_routes(scores, target_executed_layers, num_layers, candidate_layers):
    middle_budget = target_executed_layers - 2
    if middle_budget < 0:
        raise ValueError("target_executed_layers must include first and final layers.")
    if middle_budget > len(candidate_layers):
        raise ValueError("target_executed_layers asks for more middle layers than candidates.")

    routes = []
    for i in range(scores.shape[0]):
        selected = {0, num_layers - 1}
        if middle_budget > 0:
            top_pos = torch.topk(scores[i], k=middle_budget).indices.detach().cpu().tolist()
            selected.update(candidate_layers[pos] for pos in top_pos)
        routes.append(sorted(selected))
    return routes


def train_predictor(model, encoder_layers, predictor, train_loader, target_executed_layers, candidate_layers, args):
    optimizer = AdamW(predictor.parameters(), lr=args.predictor_learning_rate)
    model.eval()
    predictor.train()
    for p in model.parameters():
        p.requires_grad_(False)

    middle_budget = target_executed_layers - 2
    sync_if_cuda()
    start = time.perf_counter()
    for epoch in range(args.predictor_epochs):
        pbar = tqdm(train_loader, desc=f"DeiT early-CLS BCE predictor {target_executed_layers}/{len(encoder_layers)} epoch {epoch + 1}")
        optimizer.zero_grad()
        for step, (images, _) in enumerate(pbar):
            if args.max_predictor_batches is not None and step >= args.max_predictor_batches:
                break
            images = images.to(config.DEVICE, non_blocking=True)
            early_cls, labels = compute_early_cls_and_labels(model, encoder_layers, images, candidate_layers)
            logits = predictor.forward_logits(early_cls)
            targets = topk_multilabel_targets(labels, middle_budget)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            (loss / args.gradient_accumulation_steps).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % args.log_interval == 0:
                pbar.set_postfix({"bce": f"{loss.item():.4f}"})

    optimizer.step()
    optimizer.zero_grad()
    sync_if_cuda()
    return time.perf_counter() - start, optimizer


def run_router(model, encoder_layers, predictor, adapter, images, target_executed_layers, candidate_layers, collect_routes=False):
    batch_size = images.shape[0]
    num_layers = len(encoder_layers)
    hidden_state = run_embeddings(model, images)
    hidden_state = layer_hidden_state(encoder_layers[0](hidden_state))
    scores = predictor(hidden_state[:, 0, :])
    routes = select_routes(scores, target_executed_layers, num_layers, candidate_layers)

    final_states = hidden_state.clone()
    executed_counts = torch.ones(batch_size, dtype=torch.long, device=config.DEVICE)
    skipped_counts = torch.zeros(batch_size, dtype=torch.long, device=config.DEVICE)
    kd_pairs = [(1, torch.arange(batch_size, device=config.DEVICE), hidden_state)]

    for i, route in enumerate(routes):
        sample_state = hidden_state[i:i + 1]
        prev_layer = 0
        for layer_idx in route[1:]:
            if layer_idx > prev_layer + 1:
                sample_state = adapter(sample_state, prev_layer, layer_idx)
                skipped_counts[i] += layer_idx - prev_layer - 1
            sample_state = layer_hidden_state(encoder_layers[layer_idx](sample_state))
            executed_counts[i] += 1
            kd_pairs.append((layer_idx + 1, torch.tensor([i], device=config.DEVICE), sample_state))
            prev_layer = layer_idx
        final_states[i:i + 1] = sample_state

    return final_states, kd_pairs, executed_counts, skipped_counts, routes if collect_routes else None


def train_adapter(model, encoder_layers, predictor, adapter, train_loader, target_executed_layers, candidate_layers, args):
    optimizer = AdamW(adapter.parameters(), lr=args.adapter_learning_rate)
    model.eval()
    predictor.eval()
    adapter.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)

    sync_if_cuda()
    start = time.perf_counter()
    for epoch in range(args.adapter_epochs):
        pbar = tqdm(train_loader, desc=f"DeiT early-CLS adapter {target_executed_layers}/{len(encoder_layers)} epoch {epoch + 1}")
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(pbar):
            if args.max_adapter_batches is not None and step >= args.max_adapter_batches:
                break
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            with torch.no_grad():
                teacher_outputs = model(images, output_hidden_states=True)
                teacher_hidden_states = teacher_outputs.hidden_states

            final_hidden, kd_pairs, _, _, _ = run_router(
                model, encoder_layers, predictor, adapter, images, target_executed_layers, candidate_layers
            )
            logits = logits_from_hidden(model, final_hidden)
            loss_ce = F.cross_entropy(logits, labels)
            kd_losses = []
            for teacher_idx, group_indices, student_state in kd_pairs:
                if teacher_idx >= len(teacher_hidden_states):
                    continue
                teacher_state = teacher_hidden_states[teacher_idx].detach().index_select(0, group_indices)
                kd_losses.append(F.mse_loss(student_state[:, 0, :], teacher_state[:, 0, :]))
            loss_kd = torch.stack(kd_losses).mean() if kd_losses else torch.tensor(0.0, device=config.DEVICE)
            loss = args.ce_loss_weight * loss_ce + args.kd_loss_weight * loss_kd
            (loss / args.gradient_accumulation_steps).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % args.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "ce": f"{loss_ce.item():.4f}", "kd": f"{loss_kd.item():.4f}"})

    optimizer.step()
    optimizer.zero_grad()
    sync_if_cuda()
    return time.perf_counter() - start, optimizer


@torch.inference_mode()
def evaluate_full(model, eval_loader):
    correct = 0
    samples = 0
    sync_if_cuda()
    start = time.perf_counter()
    for images, labels in tqdm(eval_loader, desc="Full DeiT baseline"):
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        pred = torch.argmax(model(images).logits, dim=-1)
        correct += (pred == labels).sum().item()
        samples += labels.size(0)
    sync_if_cuda()
    return correct, samples, time.perf_counter() - start


@torch.inference_mode()
def evaluate_router(model, encoder_layers, predictor, adapter, eval_loader, target_executed_layers, candidate_layers):
    correct = 0
    samples = 0
    total_executed = 0
    total_skipped = 0
    route_counter = Counter()
    route_correct = defaultdict(int)
    route_total = defaultdict(int)

    sync_if_cuda()
    start = time.perf_counter()
    for images, labels in tqdm(eval_loader, desc=f"Early-CLS top-k BCE target {target_executed_layers}/{len(encoder_layers)} eval"):
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        final_hidden, _, executed_counts, skipped_counts, routes = run_router(
            model, encoder_layers, predictor, adapter, images, target_executed_layers, candidate_layers, collect_routes=True
        )
        pred = torch.argmax(logits_from_hidden(model, final_hidden), dim=-1)
        is_correct = pred == labels
        correct += is_correct.sum().item()
        samples += labels.size(0)
        total_executed += executed_counts.sum().item()
        total_skipped += skipped_counts.sum().item()
        for i, route in enumerate(routes):
            key = tuple(route)
            route_counter[key] += 1
            route_total[key] += 1
            route_correct[key] += int(is_correct[i].item())

    sync_if_cuda()
    route_accuracy = {route: route_correct[route] / route_total[route] for route in route_total}
    return correct, samples, time.perf_counter() - start, total_executed / samples, total_skipped / samples, route_counter, route_accuracy


def write_reports(md_path, csv_path, args, num_layers, train_count, eval_count, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "target_speedup", "target_executed_layers", "full_accuracy", "router_accuracy",
            "full_correct", "router_correct", "total", "full_time_s", "router_time_s",
            "full_ms_per_sample", "router_ms_per_sample", "avg_executed_layers",
            "avg_skipped_layers", "theoretical_speedup", "actual_speedup",
            "predictor_train_time_s", "adapter_train_time_s", "checkpoint_path",
            "top_route", "top_route_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Early-CLS Top-K BCE Contribution Router DeiT/ImageNet Code-Split Experiment\n\n")
        f.write(
            "This experiment first executes DeiT layer 0, uses the layer-0 CLS representation to predict "
            "multi-label top-k selections over middle layers, then executes the selected route with ASTER DynamicAdapter compensation.\n\n"
        )
        f.write("## Setup\n\n")
        f.write("- Predictor label: true top-k middle layers from per-layer `cls_cosine_delta`\n")
        f.write("- Predictor loss: multi-label `BCEWithLogitsLoss`\n")
        f.write("- Forced layers: first and final encoder layer\n")
        f.write(f"- Model path/id: `{args.model_id}`\n")
        f.write(f"- ImageNet parquet dir: `{args.imagenet_parquet_dir}`\n")
        f.write(f"- Train CSV: `{args.train_csv}`\n")
        f.write(f"- Eval CSV: `{args.eval_csv}`\n")
        f.write(f"- Train samples: `{train_count}`\n")
        f.write(f"- Eval samples: `{eval_count}`\n")
        f.write(f"- Predictor epochs per target: `{args.predictor_epochs}`\n")
        f.write(f"- Adapter epochs per target: `{args.adapter_epochs}`\n")
        f.write(f"- Batch size: `{args.batch_size}`\n")
        f.write(f"- Eval batch size: `{args.eval_batch_size}`\n")
        f.write(f"- Encoder layers: `{num_layers}`\n\n")

        f.write("## Results\n\n")
        f.write("| Target | Executed Layers | Full Acc | Router Acc | Full Time (s) | Router Time (s) | Theoretical Speedup | Actual Speedup | Route Top-1 |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['target_speedup']}x | {row['target_executed_layers']} / {num_layers} | "
                f"{float(row['full_accuracy']):.4f} | {float(row['router_accuracy']):.4f} | "
                f"{float(row['full_time_s']):.4f} | {float(row['router_time_s']):.4f} | "
                f"{float(row['theoretical_speedup']):.4f}x | {float(row['actual_speedup']):.4f}x | "
                f"`{row['top_route']}` ({row['top_route_count']}) |\n"
            )
        f.write("\n## Per-Target Details\n\n")
        for row in rows:
            f.write(f"### Target {row['target_speedup']}x\n\n")
            f.write(f"- Checkpoint: `{row['checkpoint_path']}`\n")
            f.write(f"- Predictor training wall time: `{float(row['predictor_train_time_s']):.2f}` seconds\n")
            f.write(f"- Adapter training wall time: `{float(row['adapter_train_time_s']):.2f}` seconds\n")
            f.write(f"- Full baseline: `{row['full_correct']} / {row['total']}` correct, `{float(row['full_time_s']):.4f}` s\n")
            f.write(f"- Router: `{row['router_correct']} / {row['total']}` correct, `{float(row['router_time_s']):.4f}` s\n")
            f.write(f"- Average executed layers: `{float(row['avg_executed_layers']):.4f}` / `{num_layers}`\n")
            f.write(f"- Average skipped layers: `{float(row['avg_skipped_layers']):.4f}` / `{num_layers}`\n")
            f.write("- Route distribution:\n")
            for route, count, acc in row["route_details"]:
                f.write(f"  - `{route}`: {count}, accuracy `{acc:.4f}`\n")
            f.write("\n")


def main(args):
    set_seed(args.seed)
    if args.model_id:
        config.MODEL_ID = args.model_id
    if args.imagenet_parquet_dir:
        config.IMAGENET_PARQUET_DIR = args.imagenet_parquet_dir
    args.model_id = config.MODEL_ID
    args.imagenet_parquet_dir = config.IMAGENET_PARQUET_DIR
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("--- Early-CLS top-k BCE DeiT code-split benchmark ---")
    model, image_transform = load_model_and_processor()
    model.eval()
    encoder_layers = get_encoder_layers(model)
    num_layers = len(encoder_layers)
    hidden_dim = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    print(f"Base model dtype: {dtype}, hidden_dim: {hidden_dim}, num_layers: {num_layers}")

    train_loader, eval_loader, train_count, eval_count = make_dataloaders(image_transform, args)
    full_correct, full_total, full_time = evaluate_full(model, eval_loader)

    rows = []
    for speedup_text in args.speedups.split(","):
        target_speedup = float(speedup_text.strip())
        target_executed_layers = target_layers_from_speedup(num_layers, target_speedup)
        candidate_layers = list(range(1, num_layers - 1))
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"early_cls_topk_bce_deit_target_{target_speedup:g}x")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "early_cls_topk_bce_deit_checkpoint.pt")

        predictor = EarlyClsMiddlePredictor(hidden_dim, args.predictor_hidden_dim, candidate_layers, dtype).to(config.DEVICE)
        adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, dtype).to(config.DEVICE)

        if args.skip_training and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
            predictor.load_state_dict(checkpoint["predictor_state_dict"])
            adapter.load_state_dict(checkpoint["adapter_state_dict"])
            predictor_train_time = 0.0
            adapter_train_time = 0.0
            print(f"Loaded checkpoint for target {target_speedup:g}x from: {checkpoint_path}")
        else:
            predictor_train_time, predictor_optimizer = train_predictor(
                model, encoder_layers, predictor, train_loader, target_executed_layers, candidate_layers, args
            )
            adapter_train_time, adapter_optimizer = train_adapter(
                model, encoder_layers, predictor, adapter, train_loader, target_executed_layers, candidate_layers, args
            )
            torch.save(
                {
                    "predictor_state_dict": predictor.state_dict(),
                    "adapter_state_dict": adapter.state_dict(),
                    "predictor_optimizer_state_dict": predictor_optimizer.state_dict(),
                    "adapter_optimizer_state_dict": adapter_optimizer.state_dict(),
                    "target_speedup": target_speedup,
                    "target_executed_layers": target_executed_layers,
                    "candidate_layers": candidate_layers,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint for target {target_speedup:g}x to: {checkpoint_path}")

        predictor.eval()
        adapter.eval()
        (
            router_correct,
            router_total,
            router_time,
            avg_executed,
            avg_skipped,
            route_counter,
            route_accuracy,
        ) = evaluate_router(model, encoder_layers, predictor, adapter, eval_loader, target_executed_layers, candidate_layers)
        top_route, top_route_count = route_counter.most_common(1)[0]
        route_details = [
            (list(route), count, route_accuracy[route])
            for route, count in route_counter.most_common(20)
        ]
        rows.append({
            "target_speedup": f"{target_speedup:g}",
            "target_executed_layers": target_executed_layers,
            "full_accuracy": full_correct / full_total,
            "router_accuracy": router_correct / router_total,
            "full_correct": full_correct,
            "router_correct": router_correct,
            "total": router_total,
            "full_time_s": full_time,
            "router_time_s": router_time,
            "full_ms_per_sample": full_time * 1000 / full_total,
            "router_ms_per_sample": router_time * 1000 / router_total,
            "avg_executed_layers": avg_executed,
            "avg_skipped_layers": avg_skipped,
            "theoretical_speedup": num_layers / avg_executed,
            "actual_speedup": full_time / router_time,
            "predictor_train_time_s": predictor_train_time,
            "adapter_train_time_s": adapter_train_time,
            "checkpoint_path": checkpoint_path,
            "top_route": list(top_route),
            "top_route_count": top_route_count,
            "route_details": route_details,
        })

    md_path = os.path.join(args.output_dir, args.output_name)
    csv_path = os.path.join(args.output_dir, args.csv_name)
    write_reports(md_path, csv_path, args, num_layers, train_count, eval_count, rows)
    print(f"Saved markdown report to: {md_path}")
    print(f"Saved csv report to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Early-CLS top-k BCE contribution router on ASTER-DeiT code split.")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--imagenet_parquet_dir", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--eval_csv", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--speedups", type=str, default="1.71,2.0,3.0")
    parser.add_argument("--predictor_epochs", type=int, default=1)
    parser.add_argument("--adapter_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--predictor_hidden_dim", type=int, default=256)
    parser.add_argument("--predictor_learning_rate", type=float, default=1e-4)
    parser.add_argument("--adapter_learning_rate", type=float, default=5e-5)
    parser.add_argument("--ce_loss_weight", type=float, default=10.0)
    parser.add_argument("--kd_loss_weight", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_predictor_batches", type=int, default=None)
    parser.add_argument("--max_adapter_batches", type=int, default=None)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    parser.add_argument("--output_name", type=str, default="early_cls_topk_bce_deit_code_split_speedups_1epoch.md")
    parser.add_argument("--csv_name", type=str, default="early_cls_topk_bce_deit_code_split_speedups_1epoch.csv")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
