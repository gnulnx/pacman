"""
dojo_eval.py — Parallel evaluation for Dojo RL agents.

Adds multiprocessing support to evaluate multiple models concurrently.

Each model file is evaluated in its own process to avoid GIL contention
and interference between torch/pygame instances. Results are collected,
grouped by maze size, sorted, and written to evaluation_summary.txt.

───────────────────────────────
Usage Examples
───────────────────────────────
# Evaluate all models under runs/ using 4 worker processes
dojo eval runs/ --models best,final --procs 4
"""

import os
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import click

from eval_agent import (
    evaluate_cross_size,
    evaluate_full_model_random_pacman_start_same_size_map,
    evaluate_random_start_same_size_map,
)


# ---------------------------------------------------------------------
# Worker function: executed in separate process
# ---------------------------------------------------------------------
def _evaluate_single_model(args, device="cpu"):
    model_path, episodes, delay, fps = args
    try:
        stage_dir = os.path.dirname(model_path)
        config_path = os.path.join(stage_dir, "config.pkl")
        if not os.path.exists(config_path):
            return (model_path, None, "missing config.pkl", None)

        with open(config_path, "rb") as f_cfg:
            cfg = pickle.load(f_cfg)
        spec = cfg["maze_spec"]
        size_label = f"{spec.width}x{spec.height}"

        r1 = evaluate_full_model_random_pacman_start_same_size_map(
            model_path, spec, episodes=episodes, delay=delay, fps=fps, device=device
        )
        r2 = evaluate_random_start_same_size_map(
            model_path, spec, episodes=episodes, delay=delay, fps=fps, device=device
        )
        r3 = evaluate_cross_size(model_path, spec, episodes_per_size=episodes, delay=delay, fps=fps, device=device)

        final_score = (r1 + r2 + sum(r3.values())) / (2 + len(r3))
        return (model_path, final_score, None, size_label)

    except Exception as e:
        return (model_path, None, str(e), None)


# ---------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------
@click.command("eval")
@click.argument("path", type=click.Path(exists=True))
@click.option("--episodes", default=10, help="Number of episodes per test.")
@click.option("--delay", default=0.0, help="Frame delay (0 for max speed).")
@click.option("--fps", default=1000, help="Simulation FPS.")
@click.option("--device", default="cpu", help="Device to run evaluations on (cpu, mps, or cuda).")
@click.option(
    "--models",
    default="best,final",
    help="Comma-separated checkpoint types to evaluate (options: best,final,model).",
)
@click.option("--procs", default=min(4, cpu_count()), help="Number of parallel processes.")
@click.option("--output", default="evaluation_summary.txt", help="File to save results.")
def eval_cmd(path, episodes, delay, fps, device, models, procs, output):
    """Evaluate one model file or all models in a directory (in parallel)."""
    start_time = time.time()
    selected_types = [m.strip().lower() for m in models.split(",")]
    print(f"🔎 Evaluating {path} (checkpoints: {', '.join(selected_types)}, procs={procs})")

    # Collect model paths
    model_paths = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith(".pt"):
                    continue
                if any(f"{t}_model" in file.lower() for t in selected_types):
                    model_paths.append(os.path.join(root, file))
        model_paths.sort()
        print(f"🧩 Found {len(model_paths)} model(s) to evaluate under {path}")
    else:
        model_paths = [path]

    if not model_paths:
        print("❌ No model files found matching your filters.")
        return

    results_by_size = defaultdict(list)
    with open(output, "w") as f:
        f.write(f"📊 Dojo Evaluation Summary ({time.ctime()})\n")
        f.write(f"Evaluating {len(model_paths)} model(s) from {path}\n")
        f.write(f"Checkpoint types: {', '.join(selected_types)}\n\n")
        f.flush()

        # Run in parallel
        tasks = [(m, episodes, delay, fps) for m in model_paths]
        with Pool(processes=procs) as pool:
            for idx, (model_path, score, err, size_label) in enumerate(
                pool.imap_unordered(_evaluate_single_model, [(task + (device,)) for task in tasks]), 1
            ):
                if err:
                    print(f"❌ [{idx}/{len(model_paths)}] {os.path.basename(model_path)} failed: {err}")
                    f.write(f"ERROR {model_path}: {err}\n")
                else:
                    results_by_size[size_label].append((score, model_path))
                    print(f"✅ [{idx}/{len(model_paths)}] {os.path.basename(model_path)} → {score:.4f}")
                    f.write(f"{size_label:<6} {score:>8.4f}  {model_path}\n")
                f.flush()

        # Sort results at the end
        f.write("\n📈 Sorted Results by Size:\n\n")
        for size in sorted(results_by_size.keys()):
            f.write(f"{size}\n")
            results_by_size[size].sort(key=lambda x: x[0], reverse=True)
            for score, model in results_by_size[size]:
                f.write(f"  {score:>8.4f}  {os.path.basename(os.path.dirname(model))}/{os.path.basename(model)}\n")
            f.write("\n")

    elapsed = time.time() - start_time
    print(f"\n✅ Evaluation complete in {elapsed:.1f}s")
    print(f"📝 Results saved to {output}")

    # Print quick summary
    for size, entries in sorted(results_by_size.items()):
        print(f"\n📦 {size}")
        for score, model in sorted(entries, key=lambda x: x[0], reverse=True):
            print(f"  {score:>8.3f}  {os.path.basename(os.path.dirname(model))}/{os.path.basename(model)}")
