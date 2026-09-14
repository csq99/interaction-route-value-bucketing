"""Run paper-parameter SST-2 comparisons for original ASTER and Early-CLS+BCE routing.

The new router remains sample-dynamic: it predicts the middle-layer top-k after
layer 0.  Route bucketing is only an inference scheduler and never changes a
sample's selected route.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT_DIR = RUN_DIR.parent
SOURCE_DIR = ROOT_DIR / "aster-bert"
RESULT_DIR = RUN_DIR / "results"
LOG_DIR = RUN_DIR / "logs"

TARGETS = {
    "1.2x": 5,
    "1.5x": 4,
    "3.0x": 2,
}

ORIGINAL_CHECKPOINTS = {
    "1.2x": SOURCE_DIR / "checkpoints_distilbert_sst2" / "target_5_layers" / "aster_bert_checkpoint.pt",
    "1.5x": SOURCE_DIR / "checkpoints_distilbert_sst2" / "target_4_layers_rerun_original" / "aster_bert_checkpoint.pt",
    "3.0x": SOURCE_DIR / "checkpoints_distilbert_sst2" / "target_2_layers" / "aster_bert_checkpoint.pt",
}


def run_command(command, cwd, log_path):
    print("Running:", " ".join(str(item) for item in command))
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def extract_router_report(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 5 or fields[0] not in {
            "Full baseline",
            "Dynamic router, per-sample dispatch",
            "Dynamic router, route-bucketed dispatch",
        }:
            continue
        rows[fields[0]] = {
            "accuracy": float(fields[1]),
            "seconds": float(fields[2]),
            "samples_per_sec": float(fields[3]),
            "speedup": float(fields[4].removesuffix("x")),
        }
    if len(rows) != 3:
        raise RuntimeError(f"Could not parse router results from {path}")
    return rows


def extract_original_log(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "ASTER greedy, batch_size=1:":
            continue
        accuracy_match = re.search(r"accuracy:\s+([0-9.]+)", lines[index + 1])
        time_match = re.search(r"inference_time_sec:\s+([0-9.]+)", lines[index + 2])
        if accuracy_match and time_match:
            return {"accuracy": float(accuracy_match.group(1)), "seconds": float(time_match.group(1))}
    return None


def write_summary(target_rows):
    path = RESULT_DIR / "paper_params_early_cls_bce_vs_original_aster.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DistilBERT/SST-2 Paper-Parameter Comparison\n\n")
        handle.write("## Controlled Configuration\n\n")
        handle.write("- Base model: `distilbert-base-uncased-finetuned-sst-2-english`\n")
        handle.write("- Dataset: GLUE/SST-2 validation (872 samples)\n")
        handle.write("- Train batch size: `16`\n")
        handle.write("- Gradient accumulation steps: `4`\n")
        handle.write("- Learning rate: `5e-5` for the new predictor and adapter\n")
        handle.write("- Training epochs: `1` offline predictor-pretraining epoch plus `3` adapter-task-training epochs\n")
        handle.write("- Maximum sequence length: `128`\n")
        handle.write("- New method: execute layer 0 and layer 5, predict top-k middle layers once from layer-0 CLS using multi-label BCE, then run the selected route with DynamicAdapter.\n")
        handle.write("- Original baseline: existing target-controlled ASTER MDP checkpoints trained with the same base training hyperparameters.\n\n")
        handle.write("## Batch-1 Latency Comparison\n\n")
        handle.write("| Target | Executed Layers | Full Acc | Original ASTER Acc | Original ASTER Time (s) | New Router Acc | New Router Per-sample Time (s) | New Router Speedup vs Full |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for target, row in target_rows.items():
            original = row.get("original") or {}
            router = row.get("router_bs1") or {}
            full = router.get("Full baseline", {})
            dynamic = router.get("Dynamic router, per-sample dispatch", {})
            handle.write(
                f"| {target} | {row['layers']} / 6 | {full.get('accuracy', float('nan')):.4f} | "
                f"{original.get('accuracy', float('nan')):.4f} | {original.get('seconds', float('nan')):.4f} | "
                f"{dynamic.get('accuracy', float('nan')):.4f} | {dynamic.get('seconds', float('nan')):.4f} | "
                f"{dynamic.get('speedup', float('nan')):.4f}x |\n"
            )

        handle.write("\n## Batch-32 Throughput Comparison For New Router\n\n")
        handle.write("Original ASTER is intentionally omitted here because its public implementation dispatches its greedy MDP sample by sample. The new method is shown both before and after route bucketing.\n\n")
        handle.write("| Target | Full Time (s) | Per-sample Dynamic Time (s) | Bucketed Time (s) | Bucketed Acc | Bucketed Speedup vs Full |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for target, row in target_rows.items():
            report = row.get("router_bs32") or {}
            full = report.get("Full baseline", {})
            per_sample = report.get("Dynamic router, per-sample dispatch", {})
            bucketed = report.get("Dynamic router, route-bucketed dispatch", {})
            handle.write(
                f"| {target} | {full.get('seconds', float('nan')):.4f} | {per_sample.get('seconds', float('nan')):.4f} | "
                f"{bucketed.get('seconds', float('nan')):.4f} | {bucketed.get('accuracy', float('nan')):.4f} | "
                f"{bucketed.get('speedup', float('nan')):.4f}x |\n"
            )

        handle.write("\n## Interpretation Rules\n\n")
        handle.write("- Theoretical speedup is fixed by the layer budget: `6 / executed_layers`.\n")
        handle.write("- Batch-1 measures latency. Batch-32 measures throughput. They must not be compared as the same metric.\n")
        handle.write("- Route bucketing does not turn routing into a static route: the predictor first selects each sample's route; only samples with identical selected routes are run together.\n")
    print(f"Saved combined report: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="*", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_original_benchmark", action="store_true")
    return parser.parse_args()


def main(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = {}

    for target in args.targets:
        layers = TARGETS[target]
        slug = target.replace(".", "_").replace("x", "x")
        checkpoint_dir = RUN_DIR / "checkpoints_distilbert_sst2" / f"early_cls_bce_{slug}"
        checkpoint = checkpoint_dir / "early_cls_topk_contribution_router_bert_checkpoint.pt"

        if not args.skip_train:
            train_command = [
                args.python, str(SOURCE_DIR / "train_early_cls_topk_contribution_router_bert.py"),
                "--target_executed_layers", str(layers),
                "--batch_size", "16",
                "--predictor_loss", "bce",
                "--predictor_epochs", "1",
                "--adapter_epochs", "3",
                "--predictor_learning_rate", "5e-5",
                "--adapter_learning_rate", "5e-5",
                "--checkpoint_suffix", f"early_cls_bce_{slug}",
                "--output_dir", str(RESULT_DIR),
                "--output_name", f"early_cls_bce_{slug}_training_eval.md",
            ]
            run_command(train_command, RUN_DIR, LOG_DIR / f"early_cls_bce_{slug}_train.log")

        if not checkpoint.exists():
            raise FileNotFoundError(f"New-router checkpoint missing: {checkpoint}")

        benchmark_outputs = {}
        for batch_size in (1, 32):
            output = RESULT_DIR / f"early_cls_bce_{slug}_route_bucket_bs{batch_size}.md"
            command = [
                args.python, str(ROOT_DIR / "aster-bert-route-bucket-run" / "benchmark_early_cls_route_bucketing.py"),
                "--checkpoint", str(checkpoint),
                "--target_executed_layers", str(layers),
                "--batch_size", str(batch_size),
                "--repeats", str(args.repeats),
                "--warmup", str(args.warmup),
                "--output", str(output),
            ]
            run_command(command, RUN_DIR, LOG_DIR / f"early_cls_bce_{slug}_bucket_bs{batch_size}.log")
            benchmark_outputs[batch_size] = extract_router_report(output)

        original = None
        if not args.skip_original_benchmark:
            original_checkpoint = ORIGINAL_CHECKPOINTS[target]
            if not original_checkpoint.exists():
                raise FileNotFoundError(f"Original ASTER checkpoint missing: {original_checkpoint}")
            original_log = LOG_DIR / f"original_aster_{slug}_bs1.log"
            command = [
                args.python, "benchmark_bert_aster.py",
                "--checkpoint_path", str(original_checkpoint),
                "--target_executed_layers", str(layers),
            ]
            run_command(command, SOURCE_DIR, original_log)
            original = extract_original_log(original_log)
            if original is None:
                raise RuntimeError(f"Could not parse original ASTER benchmark: {original_log}")

        rows[target] = {
            "layers": layers,
            "router_bs1": benchmark_outputs[1],
            "router_bs32": benchmark_outputs[32],
            "original": original,
        }

    write_summary(rows)


if __name__ == "__main__":
    main(parse_args())
