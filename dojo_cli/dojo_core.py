# dojo_core.py
from dataclasses import dataclass
from typing import Callable, Optional

from dojo_train_impl import train_stage
from pacman_env import MazeSpec


@dataclass
class DojoConfig:
    base_dir: str = "runs"
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 10000
    device: str = "mps"
    render_every: int = 100


class DojoTrainer:
    """
    Orchestrates multi-stage training curricula.
    """

    def __init__(self, config: Optional[DojoConfig] = None):
        self.cfg = config or DojoConfig()

    def train_stage(self, stage_name: str, maze_spec: MazeSpec, **kwargs):
        print(f"🏋️ Training stage {stage_name} ({maze_spec.width}x{maze_spec.height})")
        train_stage(stage_name, maze_spec, **kwargs)

    def run(self, curriculum_fn: Callable):
        """
        Executes a curriculum generator that yields dicts
        containing stage_name, maze_spec, and other kwargs.
        """
        stage_counter = 1
        prev_model = None
        for params in curriculum_fn():
            params.setdefault("pretrained_path", prev_model)
            self.train_stage(**params)
            prev_model = f"runs/{params['stage_name']}/final_model.pt"
            stage_counter += 1
        print("✅ Curriculum finished. Final model:", prev_model)
        return prev_model
