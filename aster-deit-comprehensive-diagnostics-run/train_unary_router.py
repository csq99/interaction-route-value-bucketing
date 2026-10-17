"""Train a unary-only route-value router while keeping the Adapter frozen."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aster-deit-interaction-multibudget-run"))
import train_interaction_multibudget_deit as shared

impl = shared.impl


class UnaryRouteValueRouter(nn.Module):
    def __init__(self, hidden_dim, candidate_count, hidden_size, dtype):
        super().__init__()
        self.candidate_count = candidate_count
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_size, device=impl.config.DEVICE, dtype=dtype), nn.GELU(),
            nn.Linear(hidden_size, hidden_size, device=impl.config.DEVICE, dtype=dtype), nn.GELU(),
            nn.Linear(hidden_size, candidate_count, device=impl.config.DEVICE, dtype=dtype),
        )

    def forward(self, hidden):
        patches = hidden[:, 1:, :]
        summary = torch.cat((hidden[:, 0, :], patches.mean(1), patches.std(1, unbiased=False)), dim=-1)
        return self.network(summary)


@torch.inference_mode()
def evaluate(model, layers, adapter, router, loader, routes, candidates):
    model.eval(); adapter.eval(); router.eval()
    total = full_correct = dynamic_correct = 0
    usage, route_correct = Counter(), Counter()
    for images, labels in tqdm(loader, desc=f"Evaluate unary router {routes.size(1) + 2}/12"):
        images, labels = images.to(impl.config.DEVICE), labels.to(impl.config.DEVICE)
        teacher_logits, teacher_hidden = shared.teacher_batch(model, images)
        hidden0 = teacher_hidden[1]
        route_indices = router(hidden0)[:, routes].sum(-1).argmax(1)
        selected = routes.index_select(0, route_indices)
        hidden = shared.execute_grouped_routes(layers, adapter, hidden0, selected, candidates)
        predicted = impl.logits_from_hidden(model, hidden).argmax(-1)
        full_correct += int(teacher_logits.argmax(-1).eq(labels).sum())
        dynamic_correct += int(predicted.eq(labels).sum())
        total += labels.numel()
        for index, route_index in enumerate(route_indices.cpu().tolist()):
            usage[route_index] += 1
            route_correct[route_index] += int(predicted[index].eq(labels[index]))
    return {
        "samples": total, "full_accuracy": full_correct / total,
        "dynamic_accuracy": dynamic_correct / total, "selected_route_count": len(usage),
        "max_route_share": max(usage.values()) / total,
    }, usage, route_correct


def main(args):
    impl.set_seed(args.seed)
    impl.config.MODEL_ID = args.model_id
    model, transform = impl.load_model_and_processor(); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    layers = impl.get_encoder_layers(model); num_layers = len(layers)
    dtype = next(model.parameters()).dtype
    adapter_state = torch.load(args.adapter_checkpoint, map_location=impl.config.DEVICE, weights_only=False)
    adapter_args = adapter_state["args"]
    adapter = impl.DynamicAdapter(model.config.hidden_size, adapter_args["adapter_bottleneck_dim"], num_layers, dtype).to(impl.config.DEVICE)
    adapter.load_state_dict(adapter_state["adapter_state_dict"]); adapter.eval()
    candidates = list(range(1, num_layers - 1))
    routes_by_target = {t: impl.all_middle_routes(len(candidates), t - 2, impl.config.DEVICE) for t in args.target_layers}
    masks_by_target = {t: impl.pair_masks(len(candidates), r, dtype)[1] for t, r in routes_by_target.items()}
    loader, count = shared.make_loader(args.train_csv, transform, args.batch_size, args.max_train_samples, args.num_workers, True)
    router = UnaryRouteValueRouter(model.config.hidden_size, len(candidates), args.router_hidden_dim, dtype).to(impl.config.DEVICE)
    optimizer = AdamW(router.parameters(), lr=args.router_learning_rate)
    history = []
    for epoch in range(1, args.router_epochs + 1):
        router.train(); total_loss = 0.0
        progress = tqdm(loader, desc=f"Unary router epoch {epoch}")
        for step, (images, labels) in enumerate(progress, 1):
            images, labels = images.to(impl.config.DEVICE), labels.to(impl.config.DEVICE)
            teacher_logits, teacher_hidden = shared.teacher_batch(model, images)
            hidden0 = teacher_hidden[1]
            target = args.target_layers[torch.randint(len(args.target_layers), (1,)).item()]
            routes = routes_by_target[target]
            sample_count = min(args.routes_per_batch, len(routes))
            sampled_indices = torch.randperm(len(routes), device=impl.config.DEVICE)[:sample_count]
            sampled_routes = routes.index_select(0, sampled_indices)
            losses = shared.route_losses(model, layers, adapter, hidden0, teacher_logits, labels,
                                         sampled_routes, candidates, args.route_kd_weight)
            scores = router(hidden0)[:, sampled_routes].sum(-1)
            target_distribution = F.softmax(-losses / args.route_label_temperature, dim=-1)
            loss = -(target_distribution * F.log_softmax(scores, dim=-1)).sum(-1).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), args.clip_grad_norm); optimizer.step()
            total_loss += float(loss.detach())
            if step % args.log_interval == 0: progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        mean_loss = total_loss / max(1, len(loader)); history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"[unary_router] epoch={epoch} loss={mean_loss:.6f}", flush=True)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "unary_router.pt"
    torch.save({"router_type": "unary", "adapter_state_dict": adapter.state_dict(),
                "router_state_dict": router.state_dict(), "target_layers": args.target_layers,
                "candidate_layers": candidates, "args": vars(args), "adapter_args": adapter_args}, checkpoint)
    with (output / "router_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    eval_loader, eval_count = shared.make_loader(
        args.eval_csv, transform, args.eval_batch_size, args.max_eval_samples, args.num_workers, False
    )
    summaries, distributions = [], []
    for target in args.target_layers:
        routes = routes_by_target[target]
        summary, usage, route_correct = evaluate(model, layers, adapter, router, eval_loader, routes, candidates)
        summary.update({"target_layers": target, "legal_routes": len(routes)})
        summaries.append(summary)
        for index, count_used in usage.most_common():
            route = [0] + [candidates[item] for item in routes[index].cpu().tolist()] + [num_layers - 1]
            distributions.append({
                "target_layers": target, "route": str(route), "count": count_used,
                "share": count_used / eval_count,
                "conditional_accuracy": route_correct[index] / count_used,
            })
    with (output / "accuracy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys()); writer.writeheader(); writer.writerows(summaries)
    with (output / "route_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=distributions[0].keys()); writer.writeheader(); writer.writerows(distributions)
    print(f"Saved unary router checkpoint: {checkpoint}; train samples: {count}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True); parser.add_argument("--train_csv", required=True); parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--adapter_checkpoint", required=True); parser.add_argument("--target_layers", type=int, nargs="+", default=[4, 6, 7])
    parser.add_argument("--max_train_samples", type=int, default=40000); parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_eval_samples", type=int, default=5000); parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1); parser.add_argument("--router_epochs", type=int, default=1)
    parser.add_argument("--routes_per_batch", type=int, default=8); parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--router_learning_rate", type=float, default=1e-4); parser.add_argument("--route_kd_weight", type=float, default=1.0)
    parser.add_argument("--route_label_temperature", type=float, default=1.0); parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10); parser.add_argument("--output_dir", required=True); parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
