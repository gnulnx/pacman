# dojo_samplers.py
import random
from dataclasses import replace
from typing import Callable, Optional

from pacman_env import MazeSpec


def random_episode_sampler(
    base_spec: MazeSpec,
    *,
    randomize_pacman: bool = True,
    randomize_single_target: bool = True,
    pellet_density: Optional[float] = None
) -> Callable[[], MazeSpec]:
    coords = [(x, y) for x in range(base_spec.width) for y in range(base_spec.height)]

    def sampler() -> MazeSpec:
        pacman_start = random.choice(coords) if randomize_pacman else base_spec.pacman_start
        pellet_positions = base_spec.pellet_positions

        if base_spec.pellet_mode == "single":
            if randomize_single_target:
                candidates = [pos for pos in coords if pos != pacman_start]
                pellet_positions = [random.choice(candidates)] if candidates else [pacman_start]
        elif pellet_density is not None:
            n_pellets = max(1, int(pellet_density * base_spec.width * base_spec.height))
            pellet_positions = random.sample(coords, n_pellets)

        return replace(
            base_spec,
            pacman_start=pacman_start,
            pellet_positions=pellet_positions,
            random_seed=random.randint(0, 10**9),
        )

    return sampler
