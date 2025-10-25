import random
import sys
import types
from typing import Tuple

import numpy as np

# Provide a lightweight pygame stub when the real dependency is unavailable.
if "pygame" not in sys.modules:
    pygame_stub = types.ModuleType("pygame")
    pygame_stub.init = lambda: None
    pygame_stub.get_init = lambda: False
    pygame_stub.quit = lambda: None
    pygame_stub.QUIT = 0
    pygame_stub.K_UP = pygame_stub.K_DOWN = pygame_stub.K_LEFT = pygame_stub.K_RIGHT = 0
    pygame_stub.event = types.SimpleNamespace(get=lambda: [])
    pygame_stub.time = types.SimpleNamespace(Clock=lambda: types.SimpleNamespace(tick=lambda self, fps: None))
    pygame_stub.Surface = lambda size: object()
    pygame_stub.Rect = lambda x, y, w, h: (x, y, w, h)
    pygame_stub.draw = types.SimpleNamespace(rect=lambda *args, **kwargs: None, circle=lambda *args, **kwargs: None)
    pygame_stub.display = types.SimpleNamespace(
        set_mode=lambda size: None, set_caption=lambda title: None, flip=lambda: None
    )
    pygame_stub.surfarray = types.SimpleNamespace(array3d=lambda surface: np.zeros((1, 1, 3), dtype=np.uint8))
    sys.modules["pygame"] = pygame_stub

from pacman_env import Maze, MazeSpec, generate_rect_layout


def _to_grid(pos: Tuple[int, int], surround_walls: bool) -> Tuple[int, int]:
    """Convert interior coordinates to grid coordinates considering border walls."""
    if surround_walls:
        return pos[0] + 1, pos[1] + 1
    return pos


def _pellet_set(maze: Maze) -> set:
    """Return pellet coordinates as (x, y) tuples for easier comparison."""
    return {(int(col), int(row)) for row, col in np.argwhere(maze.pellets == 1)}


def test_pacman_start_and_pellet_positions_are_respected():
    spec = MazeSpec(
        width=2,
        height=2,
        include_ghosts=False,
        pellet_mode="single",
        pacman_start=(1, 1),
        pellet_positions=[(1, 0)],
        include_power_pellets=False,
        surround_walls=True,
    )
    layout = generate_rect_layout(spec)
    maze = Maze(layout)

    expected_spawn = _to_grid(spec.pacman_start, spec.surround_walls)
    expected_pellet = _to_grid(spec.pellet_positions[0], spec.surround_walls)

    assert maze.pacman_spawn == expected_spawn
    assert _pellet_set(maze) == {expected_pellet}
    assert maze.pellets[expected_spawn[1], expected_spawn[0]] == 0


def test_reset_restores_pellets_without_respawning_on_pacman():
    spec = MazeSpec(
        width=2,
        height=2,
        include_ghosts=False,
        pellet_mode="single",
        pacman_start=(0, 0),
        pellet_positions=[(1, 1)],
        include_power_pellets=False,
        surround_walls=True,
    )
    layout = generate_rect_layout(spec)
    maze = Maze(layout)

    spawn = _to_grid(spec.pacman_start, spec.surround_walls)
    pellet = _to_grid(spec.pellet_positions[0], spec.surround_walls)

    # Eat the pellet and ensure it is gone.
    assert maze.consume_pellet(pellet) is True
    assert pellet not in _pellet_set(maze)

    # Reset should restore original pellet layout and keep spawn pellet-free.
    maze.reset()
    assert pellet in _pellet_set(maze)
    assert maze.pellets[spawn[1], spawn[0]] == 0


def test_default_single_pellet_does_not_overlap_with_pacman_spawn():
    spec = MazeSpec(
        width=2,
        height=2,
        include_ghosts=False,
        pellet_mode="single",
        pacman_start=(0, 0),
        pellet_positions=None,
        include_power_pellets=False,
        surround_walls=True,
    )
    layout = generate_rect_layout(spec)
    maze = Maze(layout)

    spawn = _to_grid(spec.pacman_start, spec.surround_walls)
    pellets = _pellet_set(maze)
    assert spawn not in pellets
    assert len(pellets) == 1


def test_randomized_layout_respects_specified_start_and_pellets():
    rng = random.Random(12345)
    for _ in range(20):
        width = rng.randint(2, 5)
        height = rng.randint(2, 5)
        surround = rng.choice([True, False])
        pacman_start = (rng.randrange(width), rng.randrange(height))

        available = [(x, y) for y in range(height) for x in range(width) if (x, y) != pacman_start]
        rng.shuffle(available)
        pellet_count = rng.randint(0, len(available))
        pellet_positions = available[:pellet_count]

        spec = MazeSpec(
            width=width,
            height=height,
            include_ghosts=False,
            pellet_mode="custom",
            pacman_start=pacman_start,
            pellet_positions=pellet_positions,
            include_power_pellets=False,
            surround_walls=surround,
        )

        layout = generate_rect_layout(spec)
        maze = Maze(layout)

        expected_spawn = _to_grid(pacman_start, surround)
        expected_pellets = {_to_grid(pos, surround) for pos in pellet_positions}

        assert maze.pacman_spawn == expected_spawn
        assert _pellet_set(maze) == expected_pellets
        assert maze.pellets[expected_spawn[1], expected_spawn[0]] == 0
