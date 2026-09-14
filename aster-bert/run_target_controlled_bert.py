import argparse
import os
import subprocess
import sys
from pathlib import Path


TARGETS = {
    "1.2x": 5,
    "1.5x": 4,
    "3.0x": 2,
}


def run_command(command, cwd, log_path):
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}: {' '.join(command)}")


def main():
    parser = argparse.ArgumentParser(description="Run target-controlled ASTER-BERT reproduction experiments.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per target.")
    parser.add_argument("--skip_train", action="store_true", help="Only run evaluation for existing checkpoints.")
    parser.add_argument("--targets", nargs="*", choices=TARGETS.keys(), default=list(TARGETS.keys()))
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    result_dir = workdir / "target_controlled_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for speedup_name in args.targets:
        target_layers = TARGETS[speedup_name]
        target_name = f"target_{target_layers}_layers"
        train_log = result_dir / f"{target_name}_train.log"
        eval_log = result_dir / f"{target_name}_eval.log"
        eval_md = result_dir / f"{target_name}_eval.md"

        if not args.skip_train:
            train_cmd = [
                args.python,
                "train_bert.py",
                "--target_executed_layers",
                str(target_layers),
                "--num_epochs",
                str(args.epochs),
            ]
            print(f"Training {speedup_name} ({target_layers} layers). Log: {train_log}")
            run_command(train_cmd, workdir, train_log)

        eval_cmd = [
            args.python,
            "evaluate_bert_aster.py",
            "--target_executed_layers",
            str(target_layers),
            "--output_file",
            str(eval_md),
        ]
        print(f"Evaluating {speedup_name} ({target_layers} layers). Log: {eval_log}")
        run_command(eval_cmd, workdir, eval_log)
        summary_rows.append((speedup_name, target_layers, eval_md.name, train_log.name, eval_log.name))

    summary_path = result_dir / "TARGET_CONTROLLED_SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Target-Controlled ASTER-BERT Reproduction\n\n")
        f.write("| Paper Target | Target Executed Layers | Eval Result | Train Log | Eval Log |\n")
        f.write("|---|---:|---|---|---|\n")
        for speedup_name, target_layers, eval_md, train_log, eval_log in summary_rows:
            f.write(
                f"| `{speedup_name}` | `{target_layers}` | "
                f"[{eval_md}]({eval_md}) | [{train_log}]({train_log}) | [{eval_log}]({eval_log}) |\n"
            )
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
