# dojo_curricula.py
from pacman_env import MazeSpec

from .dojo_samplers import random_episode_sampler


def curriculum_4x4_random():
    """
    Progressive 4×4 curriculum:
      1️⃣ Full pellets with varied starts
      2️⃣ Random single-pellet rehearsal
      3️⃣ Random density (25%)
      4️⃣ Random full-pellet review
    """
    stage = 1
    prev = None

    coords = [(x, y) for x in range(4) for y in range(4)]
    for start_x, start_y in coords:
        yield dict(
            stage_name=f"stage{stage}",
            maze_spec=MazeSpec(
                width=4, height=4, pellet_mode="full", pacman_start=(start_x, start_y), surround_walls=True
            ),
            pretrained_path=prev,
            eps_start=0.8,
            eps_end=0.05,
        )
        prev = f"runs/stage{stage}/final_model.pt"
        stage += 1

    single_spec = MazeSpec(width=4, height=4, pellet_mode="single", include_ghosts=False, surround_walls=True)
    single_sampler = random_episode_sampler(single_spec, randomize_pacman=True, randomize_single_target=True)
    yield dict(
        stage_name=f"stage{stage}",
        maze_spec=single_spec,
        spec_sampler=single_sampler,
        pretrained_path=prev,
        eps_start=0.8,
        eps_end=0.05,
    )
    prev = f"runs/stage{stage}/final_model.pt"
    stage += 1

    density_spec = MazeSpec(width=4, height=4, pellet_mode="custom", include_ghosts=False, surround_walls=True)
    density_sampler = random_episode_sampler(density_spec, randomize_pacman=True, pellet_density=0.25)
    yield dict(
        stage_name=f"stage{stage}",
        maze_spec=density_spec,
        spec_sampler=density_sampler,
        pretrained_path=prev,
        eps_start=0.8,
        eps_end=0.05,
    )
    prev = f"runs/stage{stage}/final_model.pt"
    stage += 1

    review_spec = MazeSpec(width=4, height=4, pellet_mode="full", include_ghosts=False, surround_walls=True)
    full_sampler = random_episode_sampler(review_spec, randomize_pacman=True)
    yield dict(
        stage_name=f"stage{stage}",
        maze_spec=review_spec,
        spec_sampler=full_sampler,
        pretrained_path=prev,
        eps_start=0.6,
        eps_end=0.10,
    )


CURRICULA = {
    "4x4_random": curriculum_4x4_random,
    "density": lambda: [],  # Density curriculum is handled directly in dojo_train.py
}
