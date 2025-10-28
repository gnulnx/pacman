#!/usr/bin/env python
"""
dojo_eval.py
------------
Extended evaluation CLI for Dojo RL models.

Features
========
• Evaluate a single model or an entire directory (e.g. runs/)
• Limit checkpoint types via --models (best, final)
• Force CPU evaluation with torch thread control (default)
• Produce a live-updating summary file sorted by map size and score
• Show Top-N results per size category and Top-5 overall winners

Example usage
=============
# Evaluate one model
dojo eval runs/stage56/final_model.pt --episodes 10 --device cpu

# Evaluate all stages in runs/ using both best and final models
dojo eval runs/ --models best,final --top-n 10

NOTE Performance testing shows it's faster to on cpu than mps for eval:
----------------------------------------
Device	User CPU Time	System Time	CPU Utilization	Wall Time (Real)
CPU	1345 s	2253 s	856 %	7 min 0 sec
MPS	509 s	129 s	103 %	10 min 17 sec
"""

import os
import pickle
import sys
import time
from multiprocessing import Pool

import click

from eval_agent import (
    evaluate_cross_size,
    evaluate_full_model_random_pacman_start_same_size_map,
    evaluate_random_start_same_size_map,
)
from pacman_env import MazeSpec


# ---------- helper for per-model evaluation ----------
def _evaluate_single_model(args):
    model_path, episodes, delay, fps, device = args
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    # torch.set_default_device("cpu")  # enforce CPU; use device flag later if expanded

    stage_dir = os.path.dirname(model_path)
    cfg_path = os.path.join(stage_dir, "config.pkl")
    if not os.path.exists(cfg_path):
        return None

    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
    spec: MazeSpec = cfg["maze_spec"]

    try:
        r1 = evaluate_full_model_random_pacman_start_same_size_map(
            model_path, spec, episodes=episodes, delay=delay, fps=fps, device=device
        )
        r2 = evaluate_random_start_same_size_map(
            model_path, spec, episodes=episodes, delay=delay, fps=fps, device=device
        )
        r3 = evaluate_cross_size(model_path, spec, episodes_per_size=episodes, delay=delay, fps=fps, device=device)
        final_score = (r1 + r2 + sum(r3.values())) / (2 + len(r3))
        return ((spec.width, spec.height), final_score, model_path)
    except Exception as e:
        print(f"❌ Error evaluating {model_path}: {e}")
        return None


# ---------- main CLI command ----------
@click.command("eval")
@click.argument("path", type=click.Path(exists=True))
@click.option("--episodes", default=10, help="Number of episodes per test.")
@click.option("--delay", default=0.0, help="Frame delay (0 for max speed).")
@click.option("--fps", default=1000, help="Frames per second.")
@click.option("--models", default="best,final", help="Comma-separated model types to evaluate.")
@click.option("--device", default="cpu", help="Device to use (cpu, mps, cuda).")
@click.option("--procs", default=4, help="Number of parallel processes.")
@click.option("--top-n", default=10, help="Show only top N per maze size (default 10).")
def eval_cmd(path, episodes, delay, fps, models, device, procs, top_n):
    """Evaluate one model or all models in a directory and summarize results."""
    model_types = [m.strip() for m in models.split(",") if m.strip()]
    summary_path = "evaluation_summary.txt"

    with open(summary_path, "w") as f:
        f.write(f"📊 Dojo Evaluation Summary ({time.ctime()})\n")
        f.write(f"Evaluating models from {path}\n")
        f.write(f"Checkpoint types: {', '.join(model_types)}\n\n")

    # --- gather model paths ---
    tasks = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for mtype in model_types:
                fname = f"{mtype}_model.pt"
                if fname in files:
                    tasks.append((os.path.join(root, fname), episodes, delay, fps, device))
    else:
        tasks.append((path, episodes, delay, fps, device))

    if not tasks:
        print("⚠️  No matching model files found.")
        sys.exit(1)

    print(f"🧠 Evaluating {len(tasks)} model(s) using {procs} process(es)...")

    # --- parallel evaluation ---
    results = []
    with Pool(processes=procs) as pool:
        for res in pool.imap_unordered(_evaluate_single_model, tasks):
            if res is None:
                continue
            results.append(res)
            (w, h), score, path_ = res
            # live logging
            with open(summary_path, "a") as f:
                f.write(f"{w}x{h:<5} {score:>9.4f}  {path_}\n")
            print(f"✅ {w}x{h}  {score:.4f}  {os.path.basename(path_)}")
    if not results:
        print("❌ No successful evaluations.")
        sys.exit(1)

    # --- sort and group results ---
    by_size = {}
    for (w, h), score, path_ in results:
        key = f"{w}x{h}"
        by_size.setdefault(key, []).append((score, path_))

    with open(summary_path, "a") as f:
        f.write("\n📈 Sorted Results by Size:\n\n")
        for size in sorted(by_size):
            f.write(f"{size}\n")
            sorted_results = sorted(by_size[size], key=lambda x: x[0], reverse=True)
            top_entries = sorted_results[:top_n]
            for sc, pth in top_entries:
                f.write(f"    {sc:8.4f}  {os.path.basename(os.path.dirname(pth))}/{os.path.basename(pth)}\n")
            f.write("\n")

        # overall top 5
        # Top 5 overall models across all map sizes
        all_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:5]
        f.write("🏆 Top 5 Overall Models\n")
        for (w, h), sc, pth in all_sorted:
            label = f"{w}x{h}"
            f.write(f"    {label:<6} {sc:8.4f}  {os.path.basename(os.path.dirname(pth))}/{os.path.basename(pth)}\n")

    print(f"\n✅ Evaluation complete. Results saved to {summary_path}")


if __name__ == "__main__":
    eval_cmd()
