import argparse
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_bert as config
from components import DynamicAdapter
from model_utils_bert import load_model_and_tokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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
    cls_state = hidden_state[:, 0]
    if hasattr(model, "pre_classifier"):
        cls_state = model.pre_classifier(cls_state)
    return model.classifier(cls_state)


def load_sst2(tokenizer):
    raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)

    def preprocess_function(examples):
        return tokenizer(
            examples[config.DATASET_TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )

    tokenized = raw_datasets.map(preprocess_function, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in [config.DATASET_TEXT_COLUMN, "idx"] if c in tokenized["train"].column_names]
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


def cls_cosine_delta(h_in, h_out):
    return 1.0 - F.cosine_similarity(h_in[:, 0, :], h_out[:, 0, :], dim=-1, eps=1e-8)


def pairwise_ranking_loss(preds, labels, margin):
    label_diff = labels.unsqueeze(2) - labels.unsqueeze(1)
    pred_diff = preds.unsqueeze(2) - preds.unsqueeze(1)
    valid = label_diff.abs() > 1e-6
    if not valid.any():
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    target_sign = label_diff.sign()
    losses = F.relu(margin - target_sign * pred_diff)
    return losses[valid].mean()


class EarlyClsMiddleContributionPredictor(nn.Module):
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
        return F.softplus(self.forward_logits(early_cls))


def topk_multilabel_targets(labels, k):
    targets = torch.zeros_like(labels)
    if k <= 0:
        return targets
    top_pos = torch.topk(labels, k=k, dim=1).indices
    targets.scatter_(1, top_pos, 1.0)
    return targets


def select_middle_topk_routes(scores, target_executed_layers, num_layers, candidate_layers):
    middle_budget = target_executed_layers - 2
    if target_executed_layers < 2:
        raise ValueError("target_executed_layers must include forced first and final layers.")
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


@torch.no_grad()
def compute_early_cls_and_labels(model, input_ids, attention_mask, candidate_layers):
    embeddings = model.distilbert.embeddings(input_ids=input_ids)
    layer0_out = layer_hidden_state(
        model.distilbert.transformer.layer[0](embeddings, attn_mask=attention_mask)
    )
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    labels = torch.stack(
        [cls_cosine_delta(hidden_states[layer_idx], hidden_states[layer_idx + 1]) for layer_idx in candidate_layers],
        dim=1,
    )
    return layer0_out[:, 0, :].detach(), labels.detach()


def train_predictor(model, predictor, dataloader, candidate_layers, args):
    optimizer = AdamW(predictor.parameters(), lr=args.predictor_learning_rate)
    model.eval()
    predictor.train()
    for p in model.parameters():
        p.requires_grad_(False)

    sync_if_cuda()
    start = time.perf_counter()
    for epoch in range(args.predictor_epochs):
        pbar = tqdm(dataloader, desc=f"Early-CLS predictor epoch {epoch + 1}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            if args.max_predictor_batches is not None and step >= args.max_predictor_batches:
                break
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            early_cls, labels = compute_early_cls_and_labels(model, input_ids, attention_mask, candidate_layers)
            middle_budget = args.target_executed_layers - 2
            if args.predictor_loss == "bce":
                logits = predictor.forward_logits(early_cls)
                targets = topk_multilabel_targets(labels, middle_budget)
                loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
                loss = loss_bce
                loss_mse = torch.tensor(0.0, device=config.DEVICE)
                loss_rank = torch.tensor(0.0, device=config.DEVICE)
            else:
                preds = predictor(early_cls)
                loss_mse = F.mse_loss(preds, labels)
                loss_rank = pairwise_ranking_loss(preds, labels, args.ranking_margin)
                loss = loss_mse + args.ranking_loss_weight * loss_rank
                loss_bce = torch.tensor(0.0, device=config.DEVICE)
            (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), config.CLIP_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % config.LOG_INTERVAL == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "bce": f"{loss_bce.item():.6f}",
                    "mse": f"{loss_mse.item():.6f}",
                    "rank": f"{loss_rank.item():.6f}",
                })
    sync_if_cuda()
    return time.perf_counter() - start, optimizer


def run_early_cls_router(model, predictor, adapter, input_ids, attention_mask, candidate_layers, args, collect_routes=False):
    batch_size = input_ids.shape[0]
    num_layers = len(model.distilbert.transformer.layer)
    hidden_state = model.distilbert.embeddings(input_ids=input_ids)
    hidden_state = layer_hidden_state(model.distilbert.transformer.layer[0](hidden_state, attn_mask=attention_mask))

    with torch.no_grad():
        scores = predictor(hidden_state[:, 0, :])
        routes = select_middle_topk_routes(scores, args.target_executed_layers, num_layers, candidate_layers)

    final_states = hidden_state.clone()
    executed_counts = torch.ones(batch_size, dtype=torch.long, device=config.DEVICE)
    skipped_counts = torch.zeros(batch_size, dtype=torch.long, device=config.DEVICE)
    kd_pairs = [(1, torch.arange(batch_size, device=config.DEVICE), hidden_state)]

    for i, route in enumerate(routes):
        sample_state = hidden_state[i:i + 1]
        sample_mask = attention_mask[i:i + 1]
        prev_layer = 0
        for layer_idx in route[1:]:
            if layer_idx > prev_layer + 1:
                sample_state = adapter(sample_state, prev_layer, layer_idx)
                skipped_counts[i] += layer_idx - prev_layer - 1
            sample_state = layer_hidden_state(
                model.distilbert.transformer.layer[layer_idx](sample_state, attn_mask=sample_mask)
            )
            executed_counts[i] += 1
            kd_pairs.append((layer_idx + 1, torch.tensor([i], device=config.DEVICE), sample_state))
            prev_layer = layer_idx

        final_states[i:i + 1] = sample_state

    return final_states, kd_pairs, executed_counts, skipped_counts, routes if collect_routes else None


def train_adapter(model, predictor, adapter, train_dataloader, candidate_layers, args):
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
        pbar = tqdm(train_dataloader, desc=f"Early-CLS adapter epoch {epoch + 1}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            if args.max_adapter_batches is not None and step >= args.max_adapter_batches:
                break
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            with torch.no_grad():
                teacher_outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                teacher_hidden_states = teacher_outputs.hidden_states

            final_hidden, kd_pairs, _, _, _ = run_early_cls_router(
                model, predictor, adapter, input_ids, attention_mask, candidate_layers, args
            )
            logits = logits_from_hidden(model, final_hidden)
            loss_ce = F.cross_entropy(logits, labels)
            kd_losses = []
            for teacher_idx, group_indices, student_state in kd_pairs:
                teacher_state = teacher_hidden_states[teacher_idx].detach().index_select(0, group_indices)
                kd_losses.append(F.mse_loss(student_state[:, 0, :], teacher_state[:, 0, :]))
            loss_kd = torch.stack(kd_losses).mean() if kd_losses else torch.tensor(0.0, device=config.DEVICE)
            loss = args.ce_loss_weight * loss_ce + args.kd_loss_weight * loss_kd
            if not loss.requires_grad:
                continue
            (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.CLIP_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % config.LOG_INTERVAL == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "ce": f"{loss_ce.item():.4f}", "kd": f"{loss_kd.item():.4f}"})

    sync_if_cuda()
    return time.perf_counter() - start, optimizer


@torch.inference_mode()
def evaluate_full(model, dataloader):
    correct = 0
    samples = 0
    sync_if_cuda()
    start = time.perf_counter()
    for batch in tqdm(dataloader, desc="Full baseline eval"):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = torch.argmax(logits, dim=-1)
        correct += (pred == labels).sum().item()
        samples += labels.size(0)
    sync_if_cuda()
    return correct, samples, time.perf_counter() - start


@torch.inference_mode()
def evaluate_router(model, predictor, adapter, dataloader, candidate_layers, args):
    correct = 0
    samples = 0
    total_executed = 0
    total_skipped = 0
    route_counter = Counter()
    route_correct = defaultdict(int)
    route_total = defaultdict(int)

    sync_if_cuda()
    start = time.perf_counter()
    for batch in tqdm(dataloader, desc="Early-CLS router eval"):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        final_hidden, _, executed_counts, skipped_counts, routes = run_early_cls_router(
            model, predictor, adapter, input_ids, attention_mask, candidate_layers, args, collect_routes=True
        )
        logits = logits_from_hidden(model, final_hidden)
        pred = torch.argmax(logits, dim=-1)
        is_correct = pred == labels
        correct += is_correct.sum().item()
        samples += labels.size(0)
        total_executed += executed_counts.sum().item()
        total_skipped += skipped_counts.sum().item()
        for i, route in enumerate(routes):
            route_key = tuple(route)
            route_counter[route_key] += 1
            route_total[route_key] += 1
            route_correct[route_key] += int(is_correct[i].item())

    sync_if_cuda()
    route_accuracy = {
        route: route_correct[route] / route_total[route]
        for route in route_total
    }
    return (
        correct,
        samples,
        time.perf_counter() - start,
        total_executed / samples,
        total_skipped / samples,
        route_counter,
        route_accuracy,
    )


def write_report(path, args, candidate_layers, full_stats, router_stats, predictor_seconds, adapter_seconds, checkpoint_path):
    full_correct, full_samples, full_seconds = full_stats
    (
        route_correct,
        route_samples,
        route_seconds,
        avg_executed,
        avg_skipped,
        route_counter,
        route_accuracy,
    ) = router_stats
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Early-CLS Top-K Contribution Router BERT Experiment\n\n")
        f.write(
            "This experiment first executes DistilBERT layer 0, uses the layer-0 CLS representation "
            "to predict contribution scores for middle layers, selects top-k middle layers, and always executes the final layer.\n\n"
        )
        f.write(
            "This version can train the contribution predictor either by regression/ranking or by multi-label BCE. "
            "The multi-label BCE variant converts each sample's true top-k middle layers into selected/not-selected labels, "
            "so route mistakes such as predicting `[2,4]` when the label is `[1,4]` are directly penalized.\n\n"
        )
        f.write("## Setup\n\n")
        f.write("- Label metric: per-middle-layer `cls_cosine_delta = 1 - cosine(input_CLS, output_CLS)`\n")
        if args.predictor_loss == "bce":
            f.write("- Predictor loss: multi-label `BCEWithLogitsLoss` on true top-k middle-layer selections\n")
        else:
            f.write(f"- Predictor loss: `MSE + {args.ranking_loss_weight} * pairwise_ranking_loss`, margin `{args.ranking_margin}`\n")
        f.write("- Forced layers: `[0, 5]`\n")
        f.write(f"- Candidate middle layers: `{candidate_layers}`\n")
        f.write(f"- Target executed layers: `{args.target_executed_layers}` / `6`\n")
        f.write(f"- Middle layers selected per sample: `{args.target_executed_layers - 2}`\n")
        f.write(f"- Checkpoint: `{checkpoint_path}`\n\n")

        f.write("## Results\n\n")
        f.write("| Method | Accuracy | Correct / Total | Time (s) | Samples/s | ms/sample |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| Full baseline bs=1 | {full_correct / full_samples:.4f} | {full_correct} / {full_samples} | "
            f"{full_seconds:.4f} | {full_samples / full_seconds:.2f} | {full_seconds * 1000 / full_samples:.3f} |\n"
        )
        f.write(
            f"| Early-CLS top-k router bs=1 | {route_correct / route_samples:.4f} | {route_correct} / {route_samples} | "
            f"{route_seconds:.4f} | {route_samples / route_seconds:.2f} | {route_seconds * 1000 / route_samples:.3f} |\n"
        )

        f.write("\n## Runtime And Routing\n\n")
        f.write(f"- Predictor training wall time: `{predictor_seconds:.2f}` seconds\n")
        f.write(f"- Adapter training wall time: `{adapter_seconds:.2f}` seconds\n")
        f.write(f"- Average executed layers: `{avg_executed:.4f}` / `6`\n")
        f.write(f"- Average skipped layers: `{avg_skipped:.4f}` / `6`\n")
        f.write(f"- Theoretical speedup from avg executed layers: `{6 / avg_executed:.4f}x`\n")
        f.write(f"- Actual speedup vs full bs=1: `{full_seconds / route_seconds:.4f}x`\n\n")

        f.write("## Executed Route Distribution\n\n")
        f.write("| Route | Samples | Accuracy |\n")
        f.write("|---|---:|---:|\n")
        for route, count in route_counter.most_common():
            f.write(f"| `{list(route)}` | {count} | {route_accuracy[route]:.4f} |\n")


def main(args):
    set_seed(config.SEED)
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.checkpoint_suffix:
        config.CHECKPOINT_DIR = os.path.join(config.CHECKPOINT_DIR, args.checkpoint_suffix)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    tokenized = load_sst2(tokenizer)
    train_loader = DataLoader(tokenized["train"], batch_size=config.BATCH_SIZE, shuffle=True)
    eval_dataset = tokenized["validation"]
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    hidden_dim = model.config.hidden_size
    num_layers = len(model.distilbert.transformer.layer)
    if num_layers != 6:
        raise ValueError(f"This experiment expects DistilBERT with 6 layers, got {num_layers}.")
    candidate_layers = list(range(1, num_layers - 1))
    dtype = next(model.parameters()).dtype
    predictor = EarlyClsMiddleContributionPredictor(
        hidden_dim,
        args.predictor_hidden_dim,
        candidate_layers,
        dtype,
    ).to(config.DEVICE)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, dtype).to(config.DEVICE)

    predictor_seconds, predictor_optimizer = train_predictor(model, predictor, train_loader, candidate_layers, args)
    adapter_seconds, adapter_optimizer = train_adapter(model, predictor, adapter, train_loader, candidate_layers, args)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "early_cls_topk_contribution_router_bert_checkpoint.pt")
    torch.save(
        {
            "predictor_state_dict": predictor.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
            "predictor_optimizer_state_dict": predictor_optimizer.state_dict(),
            "adapter_optimizer_state_dict": adapter_optimizer.state_dict(),
            "candidate_layers": candidate_layers,
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")

    full_stats = evaluate_full(model, eval_loader)
    router_stats = evaluate_router(model, predictor, adapter, eval_loader, candidate_layers, args)
    report_path = os.path.join(args.output_dir, args.output_name)
    write_report(report_path, args, candidate_layers, full_stats, router_stats, predictor_seconds, adapter_seconds, checkpoint_path)
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early-CLS top-k middle-layer contribution router for ASTER-BERT.")
    parser.add_argument("--target_executed_layers", type=int, default=4)
    parser.add_argument("--predictor_hidden_dim", type=int, default=256)
    parser.add_argument("--predictor_epochs", type=int, default=1)
    parser.add_argument("--adapter_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--predictor_learning_rate", type=float, default=1e-4)
    parser.add_argument("--adapter_learning_rate", type=float, default=5e-5)
    parser.add_argument("--predictor_loss", choices=["mse_ranking", "bce"], default="mse_ranking")
    parser.add_argument("--ranking_loss_weight", type=float, default=0.1)
    parser.add_argument("--ranking_margin", type=float, default=0.02)
    parser.add_argument("--ce_loss_weight", type=float, default=10.0)
    parser.add_argument("--kd_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_predictor_batches", type=int, default=None)
    parser.add_argument("--max_adapter_batches", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--checkpoint_suffix", type=str, default="early_cls_topk_contribution_router_cls4")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    parser.add_argument("--output_name", type=str, default="early_cls_topk_contribution_router_cls4.md")
    main(parser.parse_args())
