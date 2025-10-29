# dojo_train.py
import click

from .dojo_core import DojoConfig, DojoTrainer
from .dojo_curricula import CURRICULA
from .dojo_eval import eval_cmd  # for subcommand registration


@click.group()
def cli():
    """Dojo training and evaluation CLI."""
    pass


@cli.command()
@click.argument("curriculum", type=click.Choice(CURRICULA.keys()))
@click.option("--episodes", default=5000, help="Max episodes per stage.")
def train(curriculum, episodes):
    """Train a model using a predefined curriculum."""
    trainer = DojoTrainer(DojoConfig())
    curriculum_fn = CURRICULA[curriculum]
    trainer.run(curriculum_fn)
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

if __name__ == "__main__":
    cli()
