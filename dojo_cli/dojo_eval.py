#!/usr/bin/env python
"""
dojo_eval.py
------------
Extended evaluation CLI for Dojo RL models.

Features
========
• Evaluate one or many models (paths or directories)
• Limit checkpoint types via --models (best, final, model)
• Parallel evaluation with configurable --procs
• Force CPU/MPS/CUDA via --device
• Normalized efficiency scoring (fair across maze sizes)
• Live-updating results + sorted summary + Top-N per size
• Auto-suggests a rerun command for the top 10

Example usage
=============
# Evaluate one model
dojo eval runs/stage56/final_model.pt --episodes 10 --device cpu

# Evaluate all runs
dojo eval runs/ --models best,final --top-n 10 --procs 8
"""

import concurrent.futures
import datetime
import os
import pickle

import click

from eval_agent import (  # noqa
    evaluate_cross_size,
    evaluate_full_model_random_pacman_start_same_size_map,
    evaluate_random_start_same_size_map,
)
from pacman_env import MazeSpec


# ---------------------------------------------------------------------------
# helper: single-model evaluation (with normalized scoring)
# ---------------------------------------------------------------------------
def _evaluate_single_model(model_path, episodes, delay, fps, device, num_pellets=None):
    """Worker process function for evaluating a single model file."""
    stage_dir = os.path.dirname(model_path)
    config_path = os.path.join(stage_dir, "config.pkl")
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    spec: MazeSpec = cfg["maze_spec"]

    r1 = r2 = 1.0
    output = False
    show_pacman = False
    # num_pellets = 2

    # --- run standard suite ---
    # r1 = evaluate_full_model_random_pacman_start_same_size_map(
    #     model_path, spec, episodes=episodes, delay=delay, fps=fps, device=device, output=output, show_pacman=show_pacman
    # )
    # r2 = evaluate_random_start_same_size_map(
    #     model_path, spec, episodes=episodes, delay=delay, fps=fps, device=device, output=output, show_pacman=show_pacman
    # )

    r3 = evaluate_cross_size(
        model_path,
        spec,
        episodes_per_size=episodes,
        target_sizes=[8],
        delay=delay,
        fps=fps,
        device=device,
        output=output,
        show_pacman=show_pacman,
        num_pellets=num_pellets,
    )
    print("Results - full:", r1, "rand:", r2, "cross-size:", r3)
    base_score = (r1 + r2 + sum(r3.values())) / (2 + len(r3))
    return (spec.width, spec.height), base_score, model_path


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------
@click.command("eval")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--episodes", default=10, help="Number of episodes per test.")
@click.option("--delay", default=0.0, help="Frame delay (0 for max speed).")
@click.option("--fps", default=1000, help="Frames per second during evaluation.")
@click.option(
    "--models",
    default="best,final",
    help="Comma-separated checkpoint types to evaluate (best,final,model).",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "mps", "cuda"], case_sensitive=False),
    help="Device to run evaluations on.",
)
@click.option("--procs", default=4, help="Number of parallel processes.")
@click.option("--top-n", default=10, help="Show only top N per maze size (default 10).")
@click.option("--num-pellets", default=None, type=int, help="Number of pellets to use in evaluation.")
@click.option("--ignore-files-like", default="", help="Ignore model files containing this string.")
def eval_cmd(paths, episodes, delay, fps, models, device, procs, top_n, num_pellets, ignore_files_like):
    """
    Evaluate one or more trained Dojo models.

    Examples:
        dojo eval runs/ --models best,final
        dojo eval runs/stage56/final_model.pt
        dojo eval runs/stage35/final_model.pt runs/stage30/best_model.pt --episodes 25 --device mps
        dojo eval runs/ saved_models/0.7352_stage3_best_model.pt --models=best,final --device=cpu  --episodes=10 --procs=10 --num-pellets=3 --ignore-files-like=/best_model,fail
    """
    timestamp = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    models = [m.strip() for m in models.split(",") if m.strip()]

    output_path = "evaluation_summary.txt"
    with open(output_path, "w") as f:
        f.write(f"📊 Dojo Evaluation Summary ({timestamp})\n")
        f.write(f"Device: {device}\n")
        f.write(f"Checkpoint types: {', '.join(models)}\n")
        f.write(f"Episodes per test: {episodes}\n\n")

    # --- collect all model paths ---
    # derive model_ignores from ignore_files_like
    # print("IGNORE FILES LIKE:", ignore_files_like)
    # input()

    model_ignores = ignore_files_like.split(",") if ignore_files_like else []
    model_files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for file in files:

                    model_file = os.path.join(root, file)
                    if any(ignore in model_file for ignore in model_ignores):
                        print("Ignoring:", model_file)
                        continue
                    if any(file == f"{m}_model.pt" for m in models):
                        model_files.append(os.path.join(root, file))
        elif p.endswith(".pt"):
            if any(ignore in p for ignore in model_ignores):
                print("Ignoring:", p)
                continue
            model_files.append(p)
        else:
            print(f"⚠️ Skipping unsupported path: {p}")

    if not model_files:
        print("❌ No model files found to evaluate.")
        return

    print(f"Evaluating {len(model_files)} model(s)...")
    for mf in model_files:
        print(" -", mf)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(procs, len(model_files))) as pool, open(
        output_path, "a"
    ) as f:
        futures = [
            pool.submit(_evaluate_single_model, mp, episodes, delay, fps, device, num_pellets) for mp in model_files
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                (w, h), sc, pth = res
                line = f"✅ {w}x{h:<3} {sc:8.4f}  {pth}\n"
                print(line.strip())
                f.write(line)
                f.flush()
            except Exception as e:
                print(f"❌ Error evaluating model: {e}")

    # --- summarize results ---
    results.sort(key=lambda x: (x[0][0], -x[1]))
    summary = {}
    for (w, h), sc, pth in results:
        summary.setdefault((w, h), []).append((sc, pth))

    with open(output_path, "a") as f:
        f.write("\n📈 Sorted Results by Size:\n\n")
        print("\n📈 Sorted Results by Size:\n")
        for (w, h), items in summary.items():
            f.write(f"{w}x{h}\n")
            print(f"{w}x{h}")
            for sc, pth in sorted(items, key=lambda x: x[0], reverse=True)[:top_n]:
                f.write(f"    {sc:8.4f}  {pth}\n")
                print(f"    {sc:8.4f}  {pth}")
            f.write("\n")
            print()

        # Top 10 overall
        all_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        f.write("🏆 Top 10 Overall Models\n")
        f.write("-----------------------\n")
        top_paths = []
        for (w, h), sc, pth in all_sorted:
            label = f"{w}x{h}"
            f.write(f"    {label:<6} {sc:8.4f}  {pth}\n")
            print(f"    {label:<6} {sc:8.4f}  {pth}")
            top_paths.append(pth)

        rerun_cmd = "dojo eval " + " ".join(top_paths) + " --models=best,final --device=cpu --procs=10 --episodes=25 "
        f.write("\n💡 Tip: To re-evaluate the top 10 with more episodes, run:\n")
        f.write(f"    {rerun_cmd}\n")
        print("\n💡 Tip: To re-evaluate the top 10 with more episodes, run:\n")
        print(f"    {rerun_cmd}\n")

    print(f"\n✅ Evaluation complete. Results written to {output_path}")


if __name__ == "__main__":
    eval_cmd()
