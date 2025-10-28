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


cli.add_command(eval_cmd)

if __name__ == "__main__":
    cli()
