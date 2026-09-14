"""DistilBERT interaction-aware route-value router with random-route adapter coverage."""

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
from transformers.masking_utils import create_bidirectional_mask


ROOT = Path(__file__).resolve().parents[1]
BERT_DIR = ROOT / "aster-bert"
sys.path.insert(0, str(BERT_DIR))

# Keep the experiment offline and use the cache already used by the local aster
# environment. The project cache takes precedence when it exists.
project_hf_home = ROOT / "hf_cache_bert_sst2"
user_hf_home = Path.home() / ".cache" / "huggingface"
hf_home = project_hf_home if project_hf_home.exists() else user_hf_home
os.environ.setdefault("HF_HOME", str(hf_home))
os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import config_bert as config
import train_early_cls_topk_contribution_router_bert as bert_impl
from components import DynamicAdapter
from model_utils_bert import load_model_and_tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def layer_hidden_state(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def logits_from_hidden(model, hidden):
    cls = hidden[:, 0]
    if hasattr(model, "pre_classifier"):
        cls = model.pre_classifier(cls)
    return model.classifier(cls)


def complete_routes(num_layers, target_layers):
    middle_count = target_layers - 2
    candidates = list(range(1, num_layers - 1))
    return [[0] + list(combo) + [num_layers - 1]
            for combo in itertools.combinations(candidates, middle_count)]


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def pair_masks(candidate_layers, routes, device, dtype):
    pairs = list(itertools.combinations(candidate_layers, 2))
    mask = torch.zeros((len(routes), len(pairs)), device=device, dtype=dtype)
    for route_index, route in enumerate(routes):
        selected = set(route[1:-1])
        for pair_index, pair in enumerate(pairs):
            if pair[0] in selected and pair[1] in selected:
                mask[route_index, pair_index] = 1
    return pairs, mask


class InteractionRouteValueRouter(nn.Module):
    """One forward pass predicts unary and pairwise values for all middle layers."""

    def __init__(self, hidden_dim, candidate_count, hidden_size, dtype):
        super().__init__()
        self.candidate_count = candidate_count
        self.pair_count = candidate_count * (candidate_count - 1) // 2
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_size, device=config.DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, device=config.DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, candidate_count + self.pair_count,
                      device=config.DEVICE, dtype=dtype),
        )

    def forward(self, hidden):
        tokens = hidden[:, 1:, :]
        summary = torch.cat((
            hidden[:, 0, :],
            tokens.mean(dim=1),
            tokens.std(dim=1, unbiased=False),
        ), dim=-1)
        output = self.network(summary)
        return output[:, :self.candidate_count], output[:, self.candidate_count:]


def score_routes(unary, pairwise, route_positions, pair_mask):
    unary_score = unary[:, route_positions].sum(dim=-1)
    pair_score = pairwise @ pair_mask.t()
    return unary_score + pair_score


def teacher_batch(model, input_ids, attention_mask):
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    return output.logits.detach(), tuple(hidden.detach() for hidden in output.hidden_states)


def prepare_layer0(model, input_ids, attention_mask):
    """Match DistilBERT.forward's attention-mask preparation for manual layers."""
    embeddings = model.distilbert.embeddings(input_ids=input_ids)
    layer_attention_mask = create_bidirectional_mask(
        config=model.distilbert.config,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
    )
    hidden0 = layer_hidden_state(model.distilbert.transformer.layer[0](
        embeddings, attention_mask=layer_attention_mask,
    ))
    return hidden0, layer_attention_mask


def execute_route(model, adapter, hidden0, attention_mask, route):
    """Execute layer 0 output through one complete route ending at layer 5."""
    state = hidden0
    previous = 0
    for layer_index in route[1:]:
        if layer_index > previous + 1:
            state = adapter(state, previous, layer_index)
        state = layer_hidden_state(model.distilbert.transformer.layer[layer_index](
            state, attention_mask=attention_mask))
        previous = layer_index
    return state


def execute_grouped_routes(model, adapter, hidden0, attention_mask, routes):
    groups = defaultdict(list)
    for index, route in enumerate(routes):
        groups[tuple(route)].append(index)
    final_states = torch.empty_like(hidden0)
    for route, indices in groups.items():
        index_tensor = torch.tensor(indices, device=hidden0.device, dtype=torch.long)
        state = execute_route(
            model,
            adapter,
            hidden0.index_select(0, index_tensor),
            attention_mask.index_select(0, index_tensor),
            list(route),
        )
        final_states.index_copy_(0, index_tensor, state)
    return final_states


def route_loss_matrix(model, adapter, hidden0, attention_mask, teacher_logits,
                      labels, routes, kd_weight):
    """Return per-sample CE+logit-KD loss for every sampled complete route."""
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        losses = []
        for route in routes:
            logits = logits_from_hidden(
                model, execute_route(model, adapter, hidden0, attention_mask, route))
            ce = F.cross_entropy(logits, labels, reduction="none")
            kd = F.kl_div(
                F.log_softmax(logits, dim=-1),
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)
            losses.append(ce + kd_weight * kd)
        return torch.stack(losses, dim=1)


def evaluate_fixed_routes(model, adapter, loader, legal_routes):
    model.eval()
    adapter.eval()
    stats = {tuple(route): [0, 0] for route in legal_routes}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating fixed routes"):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            hidden0, layer_attention_mask = prepare_layer0(model, input_ids, attention_mask)
            for route in legal_routes:
                logits = logits_from_hidden(
                    model, execute_route(model, adapter, hidden0, layer_attention_mask, route))
                correct, total = stats[tuple(route)]
                stats[tuple(route)] = [correct + int(logits.argmax(-1).eq(labels).sum()),
                                       total + labels.numel()]
    rows = []
    for route, (correct, total) in stats.items():
        rows.append({
            "route": str(list(route)),
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
        })
    return rows


def coverage_train_epoch(model, adapter, loader, legal_routes, args, optimizer, epoch):
    model.eval()
    adapter.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Random-route adapter epoch {epoch}")
    for step, batch in enumerate(progress, 1):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        teacher_logits, _ = teacher_batch(model, input_ids, attention_mask)
        hidden0, layer_attention_mask = prepare_layer0(model, input_ids, attention_mask)
        route = random.choice(legal_routes)
        logits = logits_from_hidden(
            model, execute_route(model, adapter, hidden0, layer_attention_mask, route))
        ce = F.cross_entropy(logits, labels)
        kd = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
        loss = args.adapter_ce_weight * ce + args.adapter_kd_weight * kd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach())
        if step % args.log_interval == 0:
            progress.set_postfix(loss=f"{float(loss):.4f}", route=str(route))
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
    steps = min(len(loader), args.max_train_batches) if args.max_train_batches > 0 else len(loader)
    return total_loss / max(1, steps)


def route_value_train_epoch(model, adapter, router, loader, routes, route_positions,
                            pair_mask, args, optimizer, epoch):
    model.eval()
    adapter.eval()
    router.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Route-value router epoch {epoch}")
    for step, batch in enumerate(progress, 1):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        teacher_logits, _ = teacher_batch(model, input_ids, attention_mask)
        hidden0, layer_attention_mask = prepare_layer0(model, input_ids, attention_mask)
        sample_count = min(args.routes_per_batch, len(routes))
        sampled_indices = torch.randperm(len(routes), device=config.DEVICE)[:sample_count]
        sampled_routes = [routes[index] for index in sampled_indices.detach().cpu().tolist()]
        sampled_losses = route_loss_matrix(
            model, adapter, hidden0, layer_attention_mask, teacher_logits, labels,
            sampled_routes, args.route_kd_weight,
        )
        unary, pairwise = router(hidden0)
        sampled_mask = pair_mask.index_select(0, sampled_indices)
        sampled_positions = torch.tensor(
            [[route_positions[layer] for layer in route[1:-1]] for route in sampled_routes],
            device=config.DEVICE,
            dtype=torch.long,
        )
        predicted_scores = score_routes(unary, pairwise, sampled_positions, sampled_mask)
        target_distribution = F.softmax(
            -sampled_losses / args.route_label_temperature, dim=-1)
        loss = -(target_distribution * F.log_softmax(predicted_scores, dim=-1)).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), args.clip_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach())
        if step % args.log_interval == 0:
            progress.set_postfix(listwise=f"{float(loss):.4f}")
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
    steps = min(len(loader), args.max_train_batches) if args.max_train_batches > 0 else len(loader)
    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate_router(model, adapter, router, loader, legal_routes, route_positions, pair_mask):
    model.eval()
    adapter.eval()
    router.eval()
    full_correct = router_correct = total = 0
    route_counter = Counter()
    route_correct = Counter()
    route_total = Counter()
    fixed_stats = {tuple(route): [0, 0] for route in legal_routes}
    start = time.perf_counter()
    for batch in tqdm(loader, desc="Evaluating interaction router"):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        teacher_logits, _ = teacher_batch(model, input_ids, attention_mask)
        hidden0, layer_attention_mask = prepare_layer0(model, input_ids, attention_mask)
        unary, pairwise = router(hidden0)
        all_positions = torch.tensor(
            [[route_positions[layer] for layer in route[1:-1]] for route in legal_routes],
            device=config.DEVICE,
            dtype=torch.long,
        )
        selected = score_routes(unary, pairwise, all_positions, pair_mask).argmax(dim=1)
        selected_routes = [legal_routes[index] for index in selected.detach().cpu().tolist()]
        final_hidden = execute_grouped_routes(
            model, adapter, hidden0, layer_attention_mask, selected_routes)
        predicted = logits_from_hidden(model, final_hidden).argmax(dim=-1)
        full_predicted = teacher_logits.argmax(dim=-1)
        full_correct += int(full_predicted.eq(labels).sum())
        router_correct += int(predicted.eq(labels).sum())
        total += labels.numel()
        for index, route in enumerate(selected_routes):
            key = tuple(route)
            route_counter[key] += 1
            route_correct[key] += int(predicted[index].eq(labels[index]))
            route_total[key] += 1
        for route in legal_routes:
            fixed_logits = logits_from_hidden(
                model, execute_route(model, adapter, hidden0, layer_attention_mask, route))
            fixed_predicted = fixed_logits.argmax(dim=-1)
            correct, count = fixed_stats[tuple(route)]
            fixed_stats[tuple(route)] = [
                correct + int(fixed_predicted.eq(labels).sum()),
                count + labels.numel(),
            ]
    elapsed = time.perf_counter() - start
    fixed_rows = [
        {"route": str(list(route)), "accuracy": correct / count,
         "correct": correct, "total": count}
        for route, (correct, count) in fixed_stats.items()
    ]
    return {
        "full_accuracy": full_correct / total,
        "router_accuracy": router_correct / total,
        "full_correct": full_correct,
        "router_correct": router_correct,
        "total": total,
        "router_eval_time_s": elapsed,
        "routes": route_counter,
        "route_accuracy": {
            route: route_correct[route] / route_total[route]
            for route in route_counter
        },
        "fixed_rows": fixed_rows,
    }


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def resolve_model_id(model_id):
    if model_id != "distilbert-base-uncased-finetuned-sst-2-english":
        return model_id
    roots = [
        project_hf_home / "hub" / "models--distilbert-base-uncased-finetuned-sst-2-english" / "snapshots",
        user_hf_home / "hub" / "models--distilbert-base-uncased-finetuned-sst-2-english" / "snapshots",
    ]
    for root in roots:
        candidates = sorted(path for path in root.glob("*") if path.is_dir())
        if candidates:
            return str(candidates[-1])
    raise FileNotFoundError("No local DistilBERT snapshot found: " + ", ".join(map(str, roots)))


def main(args):
    set_seed(args.seed)
    config.MODEL_ID = resolve_model_id(args.model_id)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    datasets = bert_impl.load_sst2(tokenizer)
    train_ds = datasets["train"]
    eval_ds = datasets["validation"]
    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    train_loader = make_loader(train_ds, args.batch_size, True, args.seed)
    eval_loader = make_loader(eval_ds, args.eval_batch_size, False, args.seed)

    num_layers = len(model.distilbert.transformer.layer)
    if args.target_layers < 2 or args.target_layers > num_layers:
        raise ValueError(f"target_layers must be in [2, {num_layers}]")
    legal_routes = complete_routes(num_layers, args.target_layers)
    candidate_layers = list(range(1, num_layers - 1))
    route_positions = {layer: index for index, layer in enumerate(candidate_layers)}
    _, pair_mask = pair_masks(
        candidate_layers, legal_routes, config.DEVICE,
        next(model.parameters()).dtype,
    )
    print(f"Target {args.target_layers}/{num_layers}; legal routes={len(legal_routes)}", flush=True)

    dtype = next(model.parameters()).dtype
    adapter = DynamicAdapter(
        model.config.hidden_size, args.adapter_bottleneck_dim, num_layers, dtype,
    ).to(config.DEVICE)
    router = InteractionRouteValueRouter(
        model.config.hidden_size, len(candidate_layers), args.router_hidden_dim, dtype,
    ).to(config.DEVICE)

    coverage_optimizer = AdamW(adapter.parameters(), lr=args.adapter_learning_rate)
    coverage_history = []
    start = time.perf_counter()
    for epoch in range(0, args.coverage_adapter_epochs + 1):
        if epoch == 0:
            train_loss = 0.0
        else:
            train_loss = coverage_train_epoch(
                model, adapter, train_loader, legal_routes, args, coverage_optimizer, epoch,
            )
        fixed_rows = evaluate_fixed_routes(model, adapter, eval_loader, legal_routes)
        best_row = max(fixed_rows, key=lambda row: row["accuracy"])
        average_accuracy = sum(row["accuracy"] for row in fixed_rows) / len(fixed_rows)
        coverage_history.append((epoch, train_loss, average_accuracy, best_row["accuracy"], best_row["route"]))
        torch.save(
            {"epoch": epoch, "adapter_state_dict": adapter.state_dict(), "args": vars(args)},
            checkpoint_dir / f"coverage_adapter_epoch_{epoch:03d}.pt",
        )
        write_rows(output_dir / f"fixed_routes_coverage_epoch_{epoch:03d}.csv", fixed_rows)
        print(
            f"[coverage] epoch={epoch} loss={train_loss:.6f} "
            f"average_fixed={average_accuracy:.4f} best_fixed={best_row['accuracy']:.4f} "
            f"route={best_row['route']}",
            flush=True,
        )
    coverage_seconds = time.perf_counter() - start

    router_optimizer = AdamW(router.parameters(), lr=args.router_learning_rate)
    router_history = []
    start = time.perf_counter()
    for epoch in range(1, args.router_epochs + 1):
        route_loss = route_value_train_epoch(
            model, adapter, router, train_loader, legal_routes, route_positions,
            pair_mask, args, router_optimizer, epoch,
        )
        result = evaluate_router(
            model, adapter, router, eval_loader, legal_routes, route_positions, pair_mask,
        )
        best_fixed = max(row["accuracy"] for row in result["fixed_rows"])
        router_history.append((epoch, route_loss, result["router_accuracy"], best_fixed))
        print(
            f"[router] epoch={epoch} loss={route_loss:.6f} "
            f"router_accuracy={result['router_accuracy']:.4f} best_fixed={best_fixed:.4f}",
            flush=True,
        )
    router_seconds = time.perf_counter() - start

    final_checkpoint = checkpoint_dir / "interaction_route_value_router.pt"
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "router_state_dict": router.state_dict(),
            "target_layers": args.target_layers,
            "candidate_layers": candidate_layers,
            "args": vars(args),
        },
        final_checkpoint,
    )
    with (output_dir / "coverage_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "average_fixed_accuracy", "best_fixed_accuracy", "best_route"])
        writer.writerows(coverage_history)
    with (output_dir / "router_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "route_value_loss", "router_accuracy", "best_fixed_accuracy"])
        writer.writerows(router_history)

    report = output_dir / args.output_name
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# DistilBERT Interaction-Aware Route-Value Router\n\n")
        handle.write("The router predicts unary and pairwise route values from the layer-0 hidden state. "
                     "The adapter is first trained using randomly sampled complete routes.\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- Train/eval samples: `{len(train_ds)}` / `{len(eval_ds)}`\n")
        handle.write(f"- Target executed layers: `{args.target_layers}` / `{num_layers}`\n")
        handle.write(f"- Legal complete routes: `{len(legal_routes)}`\n")
        handle.write(f"- Random-route adapter epochs: `{args.coverage_adapter_epochs}`\n")
        handle.write(f"- Router epochs: `{args.router_epochs}`\n")
        handle.write(f"- Routes sampled per router batch: `{args.routes_per_batch}`\n")
        handle.write(f"- Coverage training wall time: `{coverage_seconds:.2f}` seconds\n")
        handle.write(f"- Router training/eval wall time: `{router_seconds:.2f}` seconds\n\n")
        handle.write("## Random-route adapter coverage\n\n")
        handle.write("| Epoch | Train loss | Average fixed-route accuracy | Best fixed-route accuracy | Best route |\n")
        handle.write("|---:|---:|---:|---:|---|\n")
        for row in coverage_history:
            handle.write(f"| {row[0]} | {row[1]:.6f} | {row[2]:.4f} | {row[3]:.4f} | `{row[4]}` |\n")
        handle.write("\n## Interaction router\n\n")
        handle.write("| Epoch | Route-value loss | Router accuracy | Best fixed-route accuracy |\n")
        handle.write("|---:|---:|---:|---:|\n")
        for row in router_history:
            handle.write(f"| {row[0]} | {row[1]:.6f} | {row[2]:.4f} | {row[3]:.4f} |\n")
        if router_history:
            final_result = evaluate_router(
                model, adapter, router, eval_loader, legal_routes, route_positions, pair_mask,
            )
            handle.write("\n## Final route distribution\n\n")
            handle.write(f"- Full DistilBERT accuracy: `{final_result['full_accuracy']:.4f}`\n")
            handle.write(f"- Interaction router accuracy: `{final_result['router_accuracy']:.4f}`\n")
            for route, count in final_result["routes"].most_common():
                handle.write(f"- `{list(route)}`: {count}, accuracy=`{final_result['route_accuracy'][route]:.4f}`\n")
        handle.write(f"\n- Final checkpoint: `{final_checkpoint}`\n")
    print(f"Saved checkpoint: {final_checkpoint}\nSaved report: {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--target_layers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--coverage_adapter_epochs", type=int, default=3)
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
    parser.add_argument("--checkpoint_dir", default="./checkpoints_distilbert_sst2/interaction_route_value")
    parser.add_argument("--output_dir", default="./results_interaction_route_value")
    parser.add_argument("--output_name", default="interaction_route_value_bert.md")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
