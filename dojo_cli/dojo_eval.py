"""
dojo_eval.py — Unified evaluation command for Dojo RL agents.

This module defines the `dojo eval` CLI command, which can evaluate:
  • A single trained model file (.pt)
  • An entire directory of stage outputs (e.g., runs/)

It supports automatic grouping by maze size, optional filtering by
checkpoint type (best/final), and saves results incrementally to disk.

─────────────────────────────────────────────────────────────
Usage Examples
─────────────────────────────────────────────────────────────

1️⃣ Evaluate a single model:
    dojo eval runs/stage56/final_model.pt --delay 0 --fps 1000

2️⃣ Evaluate all models under a directory:
    dojo eval runs/ --models best,final --delay 0 --fps 1000

3️⃣ Evaluate only “best” models:
    dojo eval runs/ --models best

─────────────────────────────────────────────────────────────
Output Example
─────────────────────────────────────────────────────────────

📊 Dojo Evaluation Summary (Tue Oct 28 09:26:56 2025)
Evaluating 168 model(s) from runs/

2x2
  0.1773  stage16/final_model.pt
  0.1512  stage15/best_model.pt

4x4
  0.5932  stage33/final_model.pt
  0.5538  stage17/best_model.pt

8x8
  0.6771  stage56/final_model.pt

─────────────────────────────────────────────────────────────
Notes
─────────────────────────────────────────────────────────────
• By default, all `.pt` files containing “best” or “final” are evaluated.
• Use `--models` to limit to one or both (e.g., “best,final”).
• Results are grouped by maze size and sorted descending (best → worst).
• A running log is saved to evaluation_summary.txt during execution.
"""

import os
import pickle
import time
from collections import defaultdict

import click

from eval_agent import (
    evaluate_cross_size,
    evaluate_full_model_random_pacman_start_same_size_map,
    evaluate_random_start_same_size_map,
)


@click.command("eval")
@click.argument("path", type=click.Path(exists=True))
@click.option("--episodes", default=10, help="Number of episodes per test.")
@click.option("--delay", default=0.0, help="Frame delay (0 for max speed).")
@click.option("--fps", default=1000, help="Simulation FPS.")
@click.option(
    "--models",
    default="best,final",
    help="Comma-separated list of checkpoint types to evaluate (options: best,final,model).",
)
@click.option("--output", default="evaluation_summary.txt", help="File to save running results.")
def eval_cmd(path, episodes, delay, fps, models, output):
    """
    Evaluate one model file or an entire directory of trained Dojo models.
    Automatically groups results by maze size and ranks by performance.
    """
    start_time = time.time()
    selected_types = [m.strip().lower() for m in models.split(",")]
    print(f"🔎 Evaluating: {path} (checkpoints: {', '.join(selected_types)})")

    # Determine if path is single model or directory
    model_paths = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith(".pt"):
                    continue
                # Filter by model type keywords
                if any(f"{t}_model" in file.lower() for t in selected_types):
                    model_paths.append(os.path.join(root, file))
        model_paths.sort()
        print(f"🧩 Found {len(model_paths)} matching models under {path}")
    else:
        model_paths = [path]

    if not model_paths:
        print("❌ No model files found matching your filter.")
        return

    results_by_size = defaultdict(list)
    with open(output, "w") as f:
        f.write(f"📊 Dojo Evaluation Summary ({time.ctime()})\n")
        f.write(f"Evaluating {len(model_paths)} model(s) from {path}\n")
        f.write(f"Checkpoint types: {', '.join(selected_types)}\n\n")
        f.flush()

        for model_path in model_paths:
            stage_dir = os.path.dirname(model_path)
            config_path = os.path.join(stage_dir, "config.pkl")
            if not os.path.exists(config_path):
                print(f"⚠️ Skipping {model_path}: missing config.pkl")
                continue

            with open(config_path, "rb") as f_cfg:
                cfg = pickle.load(f_cfg)
            spec = cfg["maze_spec"]
            size_label = f"{spec.width}x{spec.height}"

            print(f"\n🎯 Evaluating {model_path} ({size_label})")
            try:
                r1 = evaluate_full_model_random_pacman_start_same_size_map(
                    model_path, spec, episodes=episodes, delay=delay, fps=fps
                )
                r2 = evaluate_random_start_same_size_map(model_path, spec, episodes=episodes, delay=delay, fps=fps)
                r3 = evaluate_cross_size(model_path, spec, episodes_per_size=episodes, delay=delay, fps=fps)
                final_score = (r1 + r2 + sum(r3.values())) / (2 + len(r3))
            except Exception as e:
                print(f"❌ Error evaluating {model_path}: {e}")
                final_score = float("nan")

            results_by_size[size_label].append((final_score, model_path))
            f.write(f"{size_label:<6} {final_score:>8.4f}  {model_path}\n")
            f.flush()

        # Sort all results and write final section
        f.write("\n📈 Sorted Results by Size:\n\n")
        for size in sorted(results_by_size.keys()):
            f.write(f"{size}\n")
            results_by_size[size].sort(key=lambda x: x[0], reverse=True)
            for score, model in results_by_size[size]:
                f.write(f"  {score:>8.4f}  {os.path.basename(os.path.dirname(model))}/{os.path.basename(model)}\n")
            f.write("\n")

    elapsed = time.time() - start_time
    print(f"\n✅ Evaluation complete in {elapsed:.1f}s")
    print(f"📝 Results saved to {output}\n")

    # Pretty print summary table
    for size, entries in sorted(results_by_size.items()):
        print(f"\n📦 {size}")
        for score, model in sorted(entries, key=lambda x: x[0], reverse=True):
            print(f"  {score:>8.3f}  {os.path.basename(os.path.dirname(model))}/{os.path.basename(model)}")
