import argparse
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_bert as config
from components import DynamicAdapter, ScoringModel
from model_utils_bert import load_model_and_tokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layer_hidden_state(layer_output):
    return layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_inference_seconds(fn):
    sync_if_cuda()
    start = time.perf_counter()
    result = fn()
    sync_if_cuda()
    return time.perf_counter() - start, result


def build_eval_dataset(tokenizer):
    raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)

    def preprocess_function(examples):
        return tokenizer(
            examples[config.DATASET_TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )

    eval_dataset = raw_datasets["validation"].map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.remove_columns(
        [c for c in [config.DATASET_TEXT_COLUMN, "idx"] if c in eval_dataset.column_names]
    )
    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format("torch")
    return eval_dataset


@torch.inference_mode()
def run_baseline(model, dataloader):
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc=f"Baseline bs={dataloader.batch_size}"):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    return total_correct, total_samples


@torch.inference_mode()
def predict_aster(model, scorer, adapter, input_ids, attention_mask):
    num_layers = len(model.distilbert.transformer.layer)
    student_hidden_state = model.distilbert.embeddings(input_ids=input_ids)
    l_curr = 0
    executed_layers_count = 0
    transitions = []

    while l_curr < num_layers - 1:
        executed_layers_count += 1
        student_hidden_state = layer_hidden_state(
            model.distilbert.transformer.layer[l_curr](
                student_hidden_state,
                attn_mask=attention_mask,
            )
        )
        cls_state = student_hidden_state[:, 0, :]

        required_future_steps = config.MIN_TOTAL_EXECUTED_LAYERS - executed_layers_count
        max_allowed_l_next = num_layers - required_future_steps if required_future_steps > 0 else num_layers
        start_candidate = l_curr + 1
        end_candidate = min(num_layers, max_allowed_l_next + 1)
        candidate_layers = list(range(start_candidate, end_candidate))

        if not candidate_layers and start_candidate < num_layers:
            candidate_layers = [start_candidate]
        if not candidate_layers:
            break

        scores = scorer(h_cls=cls_state, l_curr=l_curr, candidate_layers=candidate_layers)
        action_index = torch.argmax(scores, dim=-1).item()
        l_next = candidate_layers[action_index]
        transitions.append((l_curr, l_next))

        if l_next > l_curr + 1:
            student_hidden_state = adapter(student_hidden_state, l_curr, l_next)

        l_curr = l_next

    for final_l in range(l_curr, num_layers):
        student_hidden_state = layer_hidden_state(
            model.distilbert.transformer.layer[final_l](
                student_hidden_state,
                attn_mask=attention_mask,
            )
        )
        executed_layers_count += 1

    cls_token_final = student_hidden_state[:, 0]
    logits = (
        model.classifier(model.pre_classifier(cls_token_final))
        if hasattr(model, "pre_classifier")
        else model.classifier(cls_token_final)
    )
    prediction = torch.argmax(logits, dim=-1).item()
    return prediction, executed_layers_count, transitions


@torch.inference_mode()
def run_aster(model, scorer, adapter, dataloader):
    total_correct = 0
    total_samples = 0
    total_executed_layers = 0
    executed_counter = Counter()
    skipped_counter = Counter()
    transition_counter = Counter()
    path_counter = Counter()

    for batch in tqdm(dataloader, desc="ASTER bs=1"):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        label = batch["labels"].item()

        prediction, executed_layers, transitions = predict_aster(
            model, scorer, adapter, input_ids, attention_mask
        )
        total_correct += int(prediction == label)
        total_samples += 1
        total_executed_layers += executed_layers
        executed_counter[executed_layers] += 1
        skipped_counter[len(model.distilbert.transformer.layer) - executed_layers] += 1
        transition_counter.update(transitions)
        path_counter[tuple(transitions)] += 1

    return {
        "correct": total_correct,
        "samples": total_samples,
        "total_executed_layers": total_executed_layers,
        "executed_counter": executed_counter,
        "skipped_counter": skipped_counter,
        "transition_counter": transition_counter,
        "path_counter": path_counter,
    }


def print_timing(name, seconds, samples, correct):
    print(f"{name}:")
    print(f"  accuracy: {correct / samples:.4f} ({correct}/{samples})")
    print(f"  inference_time_sec: {seconds:.4f}")
    print(f"  samples_per_sec: {samples / seconds:.2f}")
    print(f"  ms_per_sample: {seconds * 1000 / samples:.3f}")


def main(args):
    set_seed(config.SEED)
    if args.target_executed_layers is not None:
        config.TARGET_EXECUTED_LAYERS = args.target_executed_layers
        config.MIN_TOTAL_EXECUTED_LAYERS = args.target_executed_layers
        print(f"target_executed_layers: {config.TARGET_EXECUTED_LAYERS}")
    checkpoint_path = args.checkpoint_path or os.path.join(
        config.CHECKPOINT_DIR, "aster_bert_checkpoint.pt"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"device: {config.DEVICE}")
    print(f"checkpoint: {checkpoint_path}")

    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    hidden_dim = model.config.hidden_size
    num_layers = len(model.distilbert.transformer.layer)
    model_dtype = next(model.parameters()).dtype

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, model_dtype).to(config.DEVICE)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, model_dtype).to(config.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    scorer.load_state_dict(checkpoint["scorer_state_dict"])
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    scorer.eval()
    adapter.eval()

    eval_dataset = build_eval_dataset(tokenizer)
    baseline_loader_bs1 = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    baseline_loader_bs16 = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    aster_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Warm up kernels and caches. These warmups are intentionally excluded from timing.
    warmup_loader = DataLoader(eval_dataset.select(range(min(args.warmup_samples, len(eval_dataset)))), batch_size=1)
    run_baseline(model, warmup_loader)
    run_aster(model, scorer, adapter, warmup_loader)

    baseline_bs1_time, baseline_bs1 = elapsed_inference_seconds(
        lambda: run_baseline(model, baseline_loader_bs1)
    )
    baseline_bs16_time, baseline_bs16 = elapsed_inference_seconds(
        lambda: run_baseline(model, baseline_loader_bs16)
    )
    aster_time, aster_stats = elapsed_inference_seconds(
        lambda: run_aster(model, scorer, adapter, aster_loader)
    )

    print("\n" + "=" * 72)
    print("Inference Timing Summary")
    print("-" * 72)
    print_timing("Baseline full model, batch_size=1", baseline_bs1_time, baseline_bs1[1], baseline_bs1[0])
    print_timing("Baseline full model, batch_size=16", baseline_bs16_time, baseline_bs16[1], baseline_bs16[0])
    print_timing("ASTER greedy, batch_size=1", aster_time, aster_stats["samples"], aster_stats["correct"])

    avg_layers = aster_stats["total_executed_layers"] / aster_stats["samples"]
    theoretical_speedup = num_layers / avg_layers
    actual_speedup_vs_bs1 = baseline_bs1_time / aster_time
    actual_speedup_vs_bs16 = baseline_bs16_time / aster_time

    print("\nLayer Skipping Summary")
    print("-" * 72)
    print(f"  avg_executed_layers: {avg_layers:.4f} / {num_layers}")
    print(f"  avg_skipped_layers: {num_layers - avg_layers:.4f} / {num_layers}")
    print(f"  theoretical_layer_speedup: {theoretical_speedup:.4f}x")
    print(f"  actual_speedup_vs_baseline_bs1: {actual_speedup_vs_bs1:.4f}x")
    print(f"  actual_speedup_vs_baseline_bs16: {actual_speedup_vs_bs16:.4f}x")

    print("\nExecuted Layer Count Distribution")
    print("-" * 72)
    for executed, count in sorted(aster_stats["executed_counter"].items()):
        print(f"  executed={executed}: {count} ({100 * count / aster_stats['samples']:.2f}%)")

    print("\nSkipped Layer Count Distribution")
    print("-" * 72)
    for skipped, count in sorted(aster_stats["skipped_counter"].items()):
        print(f"  skipped={skipped}: {count} ({100 * count / aster_stats['samples']:.2f}%)")

    print("\nTop Layer Transitions")
    print("-" * 72)
    for (src, dst), count in aster_stats["transition_counter"].most_common(args.top_k):
        skipped_between = max(0, dst - src - 1)
        print(f"  {src}->{dst} (skip_between={skipped_between}): {count}")

    print("\nTop Decision Paths")
    print("-" * 72)
    for path, count in aster_stats["path_counter"].most_common(args.top_k):
        readable = " -> ".join([f"{src}:{dst}" for src, dst in path])
        print(f"  {readable}: {count} ({100 * count / aster_stats['samples']:.2f}%)")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DistilBERT baseline vs ASTER layer skipping.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--target_executed_layers", type=int, default=None)
    parser.add_argument("--warmup_samples", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=10)
    main(parser.parse_args())
