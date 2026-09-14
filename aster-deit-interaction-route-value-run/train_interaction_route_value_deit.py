import argparse
import csv
import itertools
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

SOURCE_DIR = Path(__file__).resolve().parents[1] / "aster-deit-early-cls-bce-run"
sys.path.insert(0, str(SOURCE_DIR))

import config
from components import DynamicAdapter
from model_utils import get_encoder_layers, load_model_and_processor, run_embeddings
from train_early_cls_topk_bce_deit import ImageNetParquetSplitDataset, layer_hidden_state, logits_from_hidden


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(csv_path, transform, batch_size, max_samples, num_workers):
    limit = None if max_samples <= 0 else max_samples
    dataset = ImageNetParquetSplitDataset(csv_path, transform=transform, max_samples=limit)
    kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": num_workers, "pin_memory": False}
    if num_workers:
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs), len(dataset)


def all_middle_routes(candidate_count, middle_count, device):
    return torch.tensor(list(itertools.combinations(range(candidate_count), middle_count)), device=device, dtype=torch.long)


def pair_masks(candidate_count, routes, dtype):
    pairs = list(itertools.combinations(range(candidate_count), 2))
    mask = torch.zeros((routes.size(0), len(pairs)), device=routes.device, dtype=dtype)
    for route_index, route in enumerate(routes.detach().cpu().tolist()):
        selected = set(route)
        for pair_index, pair in enumerate(pairs):
            if pair[0] in selected and pair[1] in selected:
                mask[route_index, pair_index] = 1
    return pairs, mask


class InteractionRouteValueRouter(nn.Module):
    """One forward pass produces unary and pairwise values for every middle layer."""

    def __init__(self, hidden_dim, candidate_count, hidden_size, dtype):
        super().__init__()
        self.candidate_count = candidate_count
        self.pair_count = candidate_count * (candidate_count - 1) // 2
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_size, device=config.DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, device=config.DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, candidate_count + self.pair_count, device=config.DEVICE, dtype=dtype),
        )

    def forward(self, hidden):
        patches = hidden[:, 1:, :]
        summary = torch.cat((hidden[:, 0, :], patches.mean(dim=1), patches.std(dim=1, unbiased=False)), dim=-1)
        output = self.network(summary)
        return output[:, :self.candidate_count], output[:, self.candidate_count:]


def score_routes(unary, pairwise, routes, pair_mask):
    unary_score = unary[:, routes].sum(dim=-1)
    pair_score = pairwise @ pair_mask.t()
    return unary_score + pair_score


def full_teacher_and_hidden0(model, encoder_layers, images):
    with torch.no_grad():
        teacher_logits = model(images).logits.detach()
        hidden0 = layer_hidden_state(encoder_layers[0](run_embeddings(model, images))).detach()
    return teacher_logits, hidden0


def execute_shared_route(encoder_layers, adapter, hidden0, middle_route, candidate_layers):
    state = hidden0
    previous = 0
    full_route = [candidate_layers[index] for index in middle_route.tolist()] + [len(encoder_layers) - 1]
    for layer_index in full_route:
        if layer_index > previous + 1:
            state = adapter(state, previous, layer_index)
        state = layer_hidden_state(encoder_layers[layer_index](state))
        previous = layer_index
    return state


def execute_grouped_routes(encoder_layers, adapter, hidden0, middle_routes, candidate_layers):
    groups = defaultdict(list)
    for index, route in enumerate(middle_routes.detach().cpu().tolist()):
        groups[tuple(route)].append(index)
    final_states = torch.empty_like(hidden0)
    for route, indices in groups.items():
        index_tensor = torch.tensor(indices, device=hidden0.device, dtype=torch.long)
        state = execute_shared_route(encoder_layers, adapter, hidden0.index_select(0, index_tensor),
                                     torch.tensor(route, device=hidden0.device), candidate_layers)
        final_states.index_copy_(0, index_tensor, state)
    return final_states


def route_losses(model, encoder_layers, adapter, hidden0, teacher_logits, labels, routes, candidate_layers, kd_weight):
    losses = []
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        for route in routes:
            logits = logits_from_hidden(model, execute_shared_route(encoder_layers, adapter, hidden0, route, candidate_layers))
            ce = F.cross_entropy(logits, labels, reduction="none")
            kd = F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs, reduction="none").sum(dim=-1)
            losses.append(ce + kd_weight * kd)
    return torch.stack(losses, dim=1)


def coverage_train_epoch(model, encoder_layers, adapter, loader, routes, candidate_layers, args, optimizer, epoch):
    model.eval()
    adapter.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Coverage adapter epoch {epoch}")
    for step, (images, labels) in enumerate(progress):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        teacher_logits, hidden0 = full_teacher_and_hidden0(model, encoder_layers, images)
        route = routes[torch.randint(len(routes), (1,), device=config.DEVICE).item()]
        logits = logits_from_hidden(model, execute_shared_route(encoder_layers, adapter, hidden0, route, candidate_layers))
        ce = F.cross_entropy(logits, labels)
        kd = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="batchmean")
        loss = args.adapter_ce_weight * ce + args.adapter_kd_weight * kd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        if (step + 1) % args.log_interval == 0:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / max(1, len(loader))


def route_value_train_epoch(model, encoder_layers, adapter, router, loader, routes, pair_mask, candidate_layers, args, optimizer, epoch):
    model.eval()
    adapter.eval()
    router.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Route-value router epoch {epoch}")
    for step, (images, labels) in enumerate(progress):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        teacher_logits, hidden0 = full_teacher_and_hidden0(model, encoder_layers, images)
        sampled_indices = torch.randperm(len(routes), device=config.DEVICE)[:args.routes_per_batch]
        sampled_routes = routes.index_select(0, sampled_indices)
        sampled_losses = route_losses(model, encoder_layers, adapter, hidden0, teacher_logits, labels,
                                      sampled_routes, candidate_layers, args.route_kd_weight)
        unary, pairwise = router(hidden0)
        predicted_scores = score_routes(unary, pairwise, sampled_routes, pair_mask.index_select(0, sampled_indices))
        target_distribution = F.softmax(-sampled_losses / args.route_label_temperature, dim=-1)
        listwise_loss = -(target_distribution * F.log_softmax(predicted_scores, dim=-1)).sum(dim=-1).mean()
        loss = listwise_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), args.clip_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        if (step + 1) % args.log_interval == 0:
            progress.set_postfix({"listwise": f"{loss.item():.4f}"})
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, encoder_layers, adapter, router, loader, routes, pair_mask, candidate_layers):
    model.eval()
    adapter.eval()
    router.eval()
    full_correct = router_correct = total = 0
    route_counter = Counter()
    route_correct = Counter()
    route_total = Counter()
    start = time.perf_counter()
    for images, labels in tqdm(loader, desc="Evaluating interaction router"):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        teacher_logits, hidden0 = full_teacher_and_hidden0(model, encoder_layers, images)
        unary, pairwise = router(hidden0)
        route_indices = score_routes(unary, pairwise, routes, pair_mask).argmax(dim=1)
        middle_routes = routes.index_select(0, route_indices)
        final_hidden = execute_grouped_routes(encoder_layers, adapter, hidden0, middle_routes, candidate_layers)
        predicted = logits_from_hidden(model, final_hidden).argmax(dim=-1)
        full_predicted = teacher_logits.argmax(dim=-1)
        full_correct += int(full_predicted.eq(labels).sum().item())
        router_correct += int(predicted.eq(labels).sum().item())
        total += labels.size(0)
        for index, route in enumerate(middle_routes.detach().cpu().tolist()):
            key = tuple([0] + [candidate_layers[item] for item in route] + [len(encoder_layers) - 1])
            route_counter[key] += 1
            route_total[key] += 1
            route_correct[key] += int(predicted[index].eq(labels[index]).item())
    elapsed = time.perf_counter() - start
    return {
        "full_accuracy": full_correct / total,
        "router_accuracy": router_correct / total,
        "full_correct": full_correct,
        "router_correct": router_correct,
        "total": total,
        "router_eval_time_s": elapsed,
        "routes": route_counter,
        "route_accuracy": {route: route_correct[route] / route_total[route] for route in route_counter},
    }


def write_report(path, args, result, coverage_loss, route_loss, route_count):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# One-Shot Interaction-Aware Route Value Router DeiT Pilot\n\n")
        handle.write("The router scores all legal layer subsets in one forward pass using unary and pairwise layer interactions. ")
        handle.write("Route-level labels are CE+KD losses measured on sampled adapter-executed paths.\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Target executed layers: `{args.target_layers}` / `12`\n")
        handle.write(f"- Train samples: `{args.max_train_samples}`\n- Eval samples: `{result['total']}`\n")
        handle.write(f"- Legal middle-layer routes: `{route_count}`\n- Sampled paths per training batch: `{args.routes_per_batch}`\n")
        handle.write(f"- Coverage adapter epochs: `{args.coverage_adapter_epochs}`\n- Router epochs: `{args.router_epochs}`\n\n")
        handle.write("## Results\n\n| Method | Accuracy | Correct / Total |\n|---|---:|---:|\n")
        handle.write(f"| Full DeiT | {result['full_accuracy']:.4f} | {result['full_correct']} / {result['total']} |\n")
        handle.write(f"| Interaction route router | {result['router_accuracy']:.4f} | {result['router_correct']} / {result['total']} |\n\n")
        handle.write("## Training\n\n")
        handle.write(f"- Final coverage-adapter loss: `{coverage_loss:.6f}`\n")
        handle.write(f"- Final route-value listwise loss: `{route_loss:.6f}`\n\n")
        handle.write("## Route Distribution\n\n")
        for route, count in result["routes"].most_common():
            handle.write(f"- `{list(route)}`: {count}, accuracy={result['route_accuracy'][route]:.4f}\n")


def main(args):
    set_seed(args.seed)
    if args.target_layers < 3 or args.target_layers > 12:
        raise ValueError("target_layers must be in [3, 12]")
    config.MODEL_ID = args.model_id
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model, transform = load_model_and_processor()
    model.eval()
    encoder_layers = get_encoder_layers(model)
    num_layers = len(encoder_layers)
    candidate_layers = list(range(1, num_layers - 1))
    middle_count = args.target_layers - 2
    routes = all_middle_routes(len(candidate_layers), middle_count, config.DEVICE)
    _, pair_mask = pair_masks(len(candidate_layers), routes, next(model.parameters()).dtype)
    print(f"Target {args.target_layers}/{num_layers}; evaluating {len(routes)} legal routes", flush=True)

    train_loader, train_count = make_loader(args.train_csv, transform, args.batch_size, args.max_train_samples, args.num_workers)
    eval_loader, _ = make_loader(args.eval_csv, transform, args.eval_batch_size, args.max_eval_samples, args.num_workers)
    dtype = next(model.parameters()).dtype
    adapter = DynamicAdapter(model.config.hidden_size, args.adapter_bottleneck_dim, num_layers, dtype).to(config.DEVICE)
    router = InteractionRouteValueRouter(model.config.hidden_size, len(candidate_layers), args.router_hidden_dim, dtype).to(config.DEVICE)

    adapter_optimizer = AdamW(adapter.parameters(), lr=args.adapter_learning_rate)
    coverage_loss = 0.0
    for epoch in range(1, args.coverage_adapter_epochs + 1):
        coverage_loss = coverage_train_epoch(model, encoder_layers, adapter, train_loader, routes, candidate_layers,
                                             args, adapter_optimizer, epoch)
        print(f"Coverage adapter epoch {epoch}: loss={coverage_loss:.6f}", flush=True)
    coverage_path = os.path.join(args.checkpoint_dir, "coverage_adapter.pt")
    torch.save({"adapter_state_dict": adapter.state_dict(), "target_layers": args.target_layers}, coverage_path)

    router_optimizer = AdamW(router.parameters(), lr=args.router_learning_rate)
    route_loss = 0.0
    for epoch in range(1, args.router_epochs + 1):
        route_loss = route_value_train_epoch(model, encoder_layers, adapter, router, train_loader, routes, pair_mask,
                                             candidate_layers, args, router_optimizer, epoch)
        print(f"Route-value router epoch {epoch}: listwise_loss={route_loss:.6f}", flush=True)

    result = evaluate(model, encoder_layers, adapter, router, eval_loader, routes, pair_mask, candidate_layers)
    checkpoint_path = os.path.join(args.checkpoint_dir, "interaction_route_value_router.pt")
    torch.save({
        "adapter_state_dict": adapter.state_dict(), "router_state_dict": router.state_dict(),
        "target_layers": args.target_layers, "candidate_layers": candidate_layers, "args": vars(args),
    }, checkpoint_path)
    report_path = os.path.join(args.output_dir, args.output_name)
    write_report(report_path, args, result, coverage_loss, route_loss, len(routes))
    with open(os.path.join(args.output_dir, args.csv_name), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["full_accuracy", "router_accuracy", "full_correct", "router_correct", "total", "router_eval_time_s", "route_count"])
        writer.writeheader()
        writer.writerow({key: result[key] for key in ("full_accuracy", "router_accuracy", "full_correct", "router_correct", "total", "router_eval_time_s")} | {"route_count": len(routes)})
    print(f"Saved checkpoint: {checkpoint_path}\nSaved report: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction-aware one-shot route value router for DeiT.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--target_layers", type=int, default=6)
    parser.add_argument("--max_train_samples", type=int, default=2048)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--coverage_adapter_epochs", type=int, default=1)
    parser.add_argument("--router_epochs", type=int, default=1)
    parser.add_argument("--routes_per_batch", type=int, default=8)
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_bottleneck_dim", type=int, default=128)
    parser.add_argument("--adapter_learning_rate", type=float, default=5e-5)
    parser.add_argument("--router_learning_rate", type=float, default=1e-4)
    parser.add_argument("--adapter_ce_weight", type=float, default=10.0)
    parser.add_argument("--adapter_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_label_temperature", type=float, default=1.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default="./checkpoints_deit/interaction_route_value")
    parser.add_argument("--output_dir", default="./comparison_results")
    parser.add_argument("--output_name", default="interaction_route_value_deit_pilot.md")
    parser.add_argument("--csv_name", default="interaction_route_value_deit_pilot.csv")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
