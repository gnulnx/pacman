# dojo_eval.py
import click

from eval_agent import (
    evaluate_cross_size,
    evaluate_full_model_random_pacman_start_same_size_map,
    evaluate_random_start_same_size_map,
)


@click.command("eval")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--episodes", default=10, help="Number of episodes per test.")
def eval_cmd(model_path, episodes):
    print("🔎 Evaluating model:", model_path)
    from pacman_env import MazeSpec

    spec = MazeSpec(width=8, height=8, pellet_mode="full", include_ghosts=False, surround_walls=True)

    r1 = evaluate_full_model_random_pacman_start_same_size_map(model_path, spec, episodes=episodes)
    r2 = evaluate_random_start_same_size_map(model_path, spec, episodes=episodes)
    r3 = evaluate_cross_size(model_path, spec, episodes_per_size=episodes)
    final = (r1 + r2 + sum(r3.values())) / (2 + len(r3))
    print(f"\n🏁 Final score: {final:.3f}")
