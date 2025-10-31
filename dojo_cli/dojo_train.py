# dojo_train.py
import os

import click

from dojo_train_impl import train_stage
from pacman_env import MazeSpec

from .dojo_core import DojoTrainer
from .dojo_curricula import CURRICULA
from .dojo_eval import eval_cmd, eval_curiosity_cmd
from .dojo_samplers import (  # noqa
    cluster_sampler,
    lattice_cluster_sampler,
    random_episode_sampler,
)


@click.group()
def cli():
    """Dojo training and evaluation CLI."""
    pass


@cli.command("train")
@click.argument("curriculum", type=click.Choice(CURRICULA.keys()))
@click.option("--episodes", default=5000, help="Max episodes per stage.")
@click.option("--width", default=8, help="Maze width for training (default: 8).")
@click.option("--height", default=8, help="Maze height for training (default: 8).")
@click.option("--output-dir", default="runs", help="Directory to save output models.")
@click.option("--prev-model-dir", default=None, help="Path to previous model directory for replay buffer mixing.")
@click.option("--replay-buffer-size", default=50000, help="Size of the replay buffer.")
@click.option("--replay-batch-size", default=64, help="Batch size for replay learning.")
@click.option("--eps-start", default=0.8, help="Starting epsilon for epsilon-greedy.")
@click.option("--eps-end", default=0.05, help="Ending epsilon for epsilon-greedy.")
@click.option("--densities", default="0.1,0.5,1.0", help="Comma-separated list of pellet densities to train on.")
@click.option(
    "--tbarl-mode",
    default=1,
    type=int,
    help="TBARL mode: 1) Sample from previous buffers for regression stop AND add those samples to the current buffer  0) Sample from previous buffers for regression stop ONLY",
)
@click.option(
    "--sampler-type",
    default="cluster",
    type=click.Choice(["random", "cluster", "lattice_cluster"]),
    help='Type of sampler to use: "random", "cluster", or "lattice_cluster".',
)
def train(
    curriculum,
    episodes,
    width,
    height,
    output_dir,
    prev_model_dir,
    replay_buffer_size,
    replay_batch_size,
    eps_start,
    eps_end,
    densities,
    tbarl_mode,
    sampler_type,
):
    """Train a model using a predefined curriculum."""

    densities = [float(x.strip()) for x in densities.split(",")]

    # ensrue prev_model is a directory
    if prev_model_dir and not os.path.isdir(prev_model_dir):
        raise ValueError(f"prev_model_dir {prev_model_dir} is not a valid directory.")

    # Always use the final_model with the replay_buffer as they are written together.
    replay_buffer = os.path.join(prev_model_dir, "replay_buffer.pkl") if prev_model_dir else None
    prev_model = os.path.join(prev_model_dir, "final_model.pt") if prev_model_dir else None

    print("Previous model directory:", prev_model_dir)
    print("Using replay buffer from:", replay_buffer)
    print("Using previous model from:", prev_model)

    if prev_model_dir and not (os.path.exists(replay_buffer) and os.path.exists(prev_model)):
        raise ValueError(f"prev_model_dir {prev_model_dir} does not contain required model files.")

    # print("Using previous model directory:", prev_model_dir)
    # print("Using replay buffer from:", replay_buffer)
    # print("Using previous model from:", prev_model)

    if sampler_type == "random":
        sampler_func = random_episode_sampler
    if sampler_type == "lattice_cluster":
        sampler_func = lattice_cluster_sampler
    else:  # cluster sampler
        sampler_func = cluster_sampler

    prev_final = None or prev_model
    agent = None

    for density in densities:
        print(
            f"{width}x{height} - running  density=({density}) on densities=({densities}) using sampler={sampler_type}"
        )
        sampler = sampler_func(
            MazeSpec(width=width, height=height, pellet_mode="custom", surround_walls=True),
            randomize_pacman=True,
            pellet_density=density,
        )

        agent = train_stage(
            stage_name=f"density_{density}",
            maze_spec=MazeSpec(width=width, height=height, pellet_mode="custom", surround_walls=True),
            pretrained_path=prev_final,
            spec_sampler=sampler,
            eps_start=eps_start,
            eps_end=eps_end,
            episodes=episodes,
            agent=agent,
            replay_batch_size=replay_batch_size,
            replay_buffer_size=replay_buffer_size,
            output_dir=output_dir,
            prev_replay_buffers={replay_buffer: 1.0},  # prop of 1.0 since only one buffer
            TBARLMode=tbarl_mode,
        )
        prev_final = f"{output_dir}/density_{density}/final_model.pt"
        replay_buffer = f"{output_dir}/density_{density}/replay_buffer.pkl"
        print("completed density:", density)

    print("✅ Training complete.")


@cli.command("finetune")
@click.option("--model", "model_path", required=True, help="Path to model .pt")
@click.option("--failures", "failure_file", default="failed_runs/final_model.pt.jsonl")
@click.option("--episodes", "total_episodes", default=10000)
@click.option("--failure-prob", default=0.25, help="Fraction of episodes sampled from failure set.")
@click.option("--width", default=8, help="Maze width for training (default: 8).")
@click.option("--height", default=8, help="Maze height for training (default: 8).")
@click.option("--eps-start", default=0.3, help="Starting epsilon for epsilon-greedy.")
@click.option("--eps-end", default=0.05, help="Ending epsilon for epsilon-greedy.")
@click.option("--output-dir", default="runs/failure_mix", help="Directory to save output models.")
def finetune(model_path, failure_file, total_episodes, failure_prob, width, height, eps_start, eps_end, output_dir):
    """
    Fine-tune a model by mixing normal random episodes with recorded failures.

    Examples:
    dojo finetune --model saved_models/0.7352_stage3_best_model.pt \
        --failures failed_runs/final_model.pt.jsonl \
        --episodes 15000 \
        --failure-prob 0.3 \
        --width 8 --height 8 \
        --eps-start=0.31
    """
    trainer = DojoTrainer()
    trainer.train_with_failure_mix(
        model_path=model_path,
        failure_file=failure_file,
        total_episodes=total_episodes,
        failure_prob=failure_prob,
        width=width,
        height=height,
        eps_start=eps_start,
        eps_end=eps_end,
        output_dir=output_dir,
    )
    print("✅ Mixed fine-tune complete.")


cli.add_command(eval_cmd)
cli.add_command(eval_curiosity_cmd)

if __name__ == "__main__":
    cli()
