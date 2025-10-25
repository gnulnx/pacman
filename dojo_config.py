from pacman_env import MazeSpec

STAGES = {
    "stage1": MazeSpec(width=2, height=2, include_ghosts=False, pellet_mode="single", pellet_positions=[(1, 1)]),
    "stage2": MazeSpec(width=4, height=4, include_ghosts=False, pellet_mode="stripe_h"),
    "stage3": MazeSpec(width=8, height=8, include_ghosts=False, pellet_mode="full", surround_walls=True),
    "stage4": MazeSpec(width=8, height=8, include_ghosts=True, ghost_positions=[(4, 4)], include_power_pellets=True),
}
