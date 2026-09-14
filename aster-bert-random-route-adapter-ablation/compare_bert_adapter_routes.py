"""DistilBERT adapter ablation: predictor routes vs random complete routes."""

import argparse
import csv
import itertools
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

ROOT = Path(__file__).resolve().parents[1]
BERT_DIR = ROOT / "aster-bert"
sys.path.insert(0, str(BERT_DIR))
# Prefer the project cache, but fall back to the user's existing offline cache.
# This keeps local runs independent of the shell's Python/HF configuration.
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


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def complete_routes(num_layers, target_layers):
    middle = target_layers - 2
    candidates = list(range(1, num_layers - 1))
    return [[0] + list(combo) + [num_layers - 1]
            for combo in itertools.combinations(candidates, middle)]


def run_route(model, adapter, input_ids, attention_mask, route):
    state = model.distilbert.embeddings(input_ids=input_ids)
    states = []
    previous = -1
    for layer_index in route:
        if layer_index > previous + 1 and previous >= 0:
            state = adapter(state, previous, layer_index)
        state = layer_hidden_state(model.distilbert.transformer.layer[layer_index](
            state, attn_mask=attention_mask))
        states.append((layer_index, state))
        previous = layer_index
    return states


def loss_for_states(model, states, teacher_logits, teacher_hidden, labels, args):
    hidden_kd = []
    for layer_index, student_state in states:
        hidden_kd.append(F.mse_loss(
            student_state[:, 0, :], teacher_hidden[layer_index + 1][:, 0, :]))
    kd = torch.stack(hidden_kd).mean() if hidden_kd else torch.zeros((), device=labels.device)
    logits = logits_from_hidden(model, states[-1][1])
    ce = F.cross_entropy(logits, labels)
    return args.ce_weight * ce + args.kd_weight * kd


def teacher_batch(model, input_ids, attention_mask):
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    return output.logits.detach(), tuple(h.detach() for h in output.hidden_states)


def select_predictor_routes(model, predictor, input_ids, attention_mask, target_layers):
    with torch.no_grad():
        layer0 = layer_hidden_state(model.distilbert.transformer.layer[0](
            model.distilbert.embeddings(input_ids=input_ids), attn_mask=attention_mask))
        scores = predictor(layer0[:, 0, :])
    return bert_impl.select_middle_topk_routes(
        scores, target_layers, len(model.distilbert.transformer.layer),
        list(range(1, len(model.distilbert.transformer.layer) - 1)))


def train_epoch(model, predictor, adapter, loader, legal_routes, mode, args, optimizer, epoch):
    model.eval(); predictor.eval(); adapter.train(); running = 0.0
    progress = tqdm(loader, desc=f"{mode} adapter epoch {epoch}")
    for step, batch in enumerate(progress, 1):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        teacher_logits, teacher_hidden = teacher_batch(model, input_ids, attention_mask)
        if mode == "predictor_route":
            routes = select_predictor_routes(model, predictor, input_ids, attention_mask, args.target_layers)
            # Routes are predicted per sample. Group identical routes so each
            # loss is evaluated only on the samples that actually use it.
            groups = defaultdict(list)
            for index, route in enumerate(routes):
                groups[tuple(route)].append(index)
            losses = []
            for route, indices in groups.items():
                index_tensor = torch.tensor(indices, device=config.DEVICE)
                sub_ids = input_ids.index_select(0, index_tensor)
                sub_mask = attention_mask.index_select(0, index_tensor)
                sub_labels = labels.index_select(0, index_tensor)
                sub_logits = teacher_logits.index_select(0, index_tensor)
                sub_hidden = tuple(h.index_select(0, index_tensor) for h in teacher_hidden)
                states = run_route(model, adapter, sub_ids, sub_mask, list(route))
                losses.append(loss_for_states(model, states, sub_logits, sub_hidden, sub_labels, args))
            loss = torch.stack(losses).mean()
        else:
            route = random.choice(legal_routes)
            states = run_route(model, adapter, input_ids, attention_mask, route)
            loss = loss_for_states(model, states, teacher_logits, teacher_hidden, labels, args)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad_norm)
        optimizer.step(); running += float(loss.detach())
        if step % args.log_interval == 0:
            progress.set_postfix(loss=f"{float(loss):.4f}")
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
    return running / max(1, min(len(loader), args.max_train_batches) if args.max_train_batches > 0 else len(loader))


@torch.no_grad()
def evaluate(model, predictor, adapter, loader, legal_routes, args):
    model.eval(); predictor.eval(); adapter.eval()
    predictor_correct = predictor_total = 0
    fixed = {tuple(route): {"correct": 0, "total": 0} for route in legal_routes}
    for batch in tqdm(loader, desc="Evaluating adapter routes"):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        with torch.no_grad():
            full = model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
            teacher_hidden = tuple(h.detach() for h in full.hidden_states)
            layer0 = layer_hidden_state(model.distilbert.transformer.layer[0](
                model.distilbert.embeddings(input_ids=input_ids), attn_mask=attention_mask))
            scores = predictor(layer0[:, 0, :])
            predicted_routes = bert_impl.select_middle_topk_routes(
                scores, args.target_layers, len(model.distilbert.transformer.layer),
                list(range(1, len(model.distilbert.transformer.layer) - 1)))
        for index, route in enumerate(predicted_routes):
            states = run_route(model, adapter, input_ids[index:index + 1],
                               attention_mask[index:index + 1], route)
            pred = logits_from_hidden(model, states[-1][1]).argmax(-1)
            predictor_correct += int(pred.eq(labels[index:index + 1]).sum())
            predictor_total += 1
        for route in legal_routes:
            states = run_route(model, adapter, input_ids, attention_mask, route)
            pred = logits_from_hidden(model, states[-1][1]).argmax(-1)
            s = fixed[tuple(route)]
            s["correct"] += int(pred.eq(labels).sum())
            s["total"] += labels.numel()
    rows = [{"route": str(list(route)), "accuracy": s["correct"] / s["total"],
             "correct": s["correct"], "total": s["total"]}
            for route, s in fixed.items()]
    return predictor_correct / predictor_total, rows


def main(args):
    model_id = args.model_id
    if model_id == "distilbert-base-uncased-finetuned-sst-2-english":
        snapshot_roots = [
            project_hf_home / "hub" / "models--distilbert-base-uncased-finetuned-sst-2-english" / "snapshots",
            user_hf_home / "hub" / "models--distilbert-base-uncased-finetuned-sst-2-english" / "snapshots",
        ]
        for snapshots in snapshot_roots:
            candidates = sorted(p for p in snapshots.glob("*") if p.is_dir())
            if candidates:
                model_id = str(candidates[-1])
                break
        else:
            raise FileNotFoundError(
                "No local DistilBERT snapshot found in: "
                + ", ".join(str(p) for p in snapshot_roots)
            )
    config.MODEL_ID = model_id
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
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
    legal_routes = complete_routes(num_layers, args.target_layers)
    candidates = list(range(1, num_layers - 1))
    dtype = next(model.parameters()).dtype
    predictor = bert_impl.EarlyClsMiddleContributionPredictor(
        model.config.hidden_size, args.predictor_hidden_dim, candidates, dtype).to(config.DEVICE)
    predictor.load_state_dict(torch.load(args.predictor_checkpoint, map_location=config.DEVICE)["predictor_state_dict"])
    for p in predictor.parameters(): p.requires_grad_(False)
    initial_adapter = DynamicAdapter(model.config.hidden_size, args.bottleneck_dim, num_layers, dtype).to(config.DEVICE)
    initial_state = {k: v.detach().clone() for k, v in initial_adapter.state_dict().items()}
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for mode in ("predictor_route", "random_route"):
        set_seed(args.seed)
        adapter = DynamicAdapter(model.config.hidden_size, args.bottleneck_dim, num_layers, dtype).to(config.DEVICE)
        adapter.load_state_dict(initial_state)
        optimizer = AdamW(adapter.parameters(), lr=args.learning_rate)
        branch = output / mode; branch.mkdir(exist_ok=True)
        history = []
        for epoch in range(0, args.epochs + 1):
            loss = 0.0 if epoch == 0 else train_epoch(
                model, predictor, adapter, train_loader, legal_routes, mode, args, optimizer, epoch)
            predictor_acc, rows = evaluate(model, predictor, adapter, eval_loader, legal_routes, args)
            best_fixed = max(row["accuracy"] for row in rows)
            history.append((epoch, loss, predictor_acc, best_fixed))
            torch.save({"epoch": epoch, "adapter_state_dict": adapter.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "args": vars(args)},
                       branch / f"adapter_epoch_{epoch:03d}.pt")
            with (branch / f"routes_epoch_{epoch:03d}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            print(f"[{mode}] epoch={epoch} predictor_accuracy={predictor_acc:.4f} best_fixed={best_fixed:.4f}", flush=True)
        with (branch / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(["epoch", "train_loss", "predictor_route_accuracy", "best_fixed_route_accuracy"]); writer.writerows(history)
        with (branch / "report.md").open("w", encoding="utf-8") as handle:
            handle.write(f"# DistilBERT adapter route ablation: `{mode}`\n\n")
            handle.write(f"- Train/eval samples: `{len(train_ds)}` / `{len(eval_ds)}`\n- Target: `{args.target_layers}/6` layers\n- Epochs: `{args.epochs}`\n")
            handle.write("- Backbone and predictor frozen; both branches start from the same adapter initialization.\n\n")
            handle.write("| Epoch | Train loss | Predictor-route accuracy | Best fixed-route accuracy |\n|---:|---:|---:|---:|\n")
            for row in history: handle.write(f"| {row[0]} | {row[1]:.6f} | {row[2]:.4f} | {row[3]:.4f} |\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--predictor_checkpoint", required=True)
    parser.add_argument("--target_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--predictor_hidden_dim", type=int, default=256)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--ce_weight", type=float, default=10.0)
    parser.add_argument("--kd_weight", type=float, default=1.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--output_dir", default="./results_bert_adapter_route_ablation")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
