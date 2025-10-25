# Minimal Pac-Man RL Environment

This project provides a compact, modular Pac-Man implementation suitable for manual play and reinforcement learning experiments.

- pygame-based rendering and keyboard controls
- Gym-style `PacmanEnv` with `reset`, `step`, and `render`
- Randomly generated mazes plus handcrafted layouts via a dataclass-driven builder (`MazeSpec`)
- Configurable ghosts, pellets, and curriculum-friendly layouts through `Config`
- Optional headless rendering (`mode="rgb_array"`) for training loops
- Lightweight trajectory recording helper for imitation learning workflows

## Installation

```bash
pip install -r requirements.txt
```

## Play the Game

```bash
python pacman_env.py
```

Use the arrow keys to guide Pac-Man. The round restarts automatically when you win or a ghost catches you.

## Use as an RL Environment

```python
import numpy as np
from pacman_env import Config, MazeSpec, PacmanEnv, ACTIONS

# Default braided maze
config = Config(maze_size=21, random_seed=42)
env = PacmanEnv(config, human_mode=False, headless=True)
state = env.reset()

done = False
total_reward = 0.0
env.record_trajectory(True)

while not done:
    action = int(np.random.choice(list(ACTIONS.keys())))  # replace with an actual policy
    state, reward, done, info = env.step(action)
    total_reward += reward

trajectory = env.record_trajectory(False)
env.close()
```

To connect the environment to Stable-Baselines3:

1. Wrap `PacmanEnv` with `gym.Env` compatible glue (e.g. inherit from `gym.Env` or use a simple adapter).
2. Set `human_mode=False` and `headless=True`.
3. Use discrete action space (0=up, 1=down, 2=left, 3=right).
4. Tune maze structure via `MazeSpec` for curriculum learning (examples below).
5. Optionally fix the maze shape via `random_seed` for reproducible training runs.

## Curriculum-Friendly Layouts

The `MazeSpec` helper makes it easy to build progressively harder environments.

```python
from pacman_env import Config, MazeSpec, PacmanEnv

# Stage 1: 2x2 grid, one pellet, no ghosts.
spec_stage1 = MazeSpec(
    width=2,
    height=2,
    include_ghosts=False,
    pellet_mode="single",
    pellet_positions=[(1, 1)],
    include_power_pellets=False,
)
env_stage1 = PacmanEnv(Config(maze_spec=spec_stage1), human_mode=False, headless=True)

# Stage 2: horizontal stripe of pellets, still no ghosts.
spec_stage2 = MazeSpec(
    width=4,
    height=4,
    include_ghosts=False,
    pellet_mode="stripe_h",
)
env_stage2 = PacmanEnv(Config(maze_spec=spec_stage2), human_mode=False, headless=True)

# Stage 3: classic 8x8 maze with pellets everywhere.
spec_stage3 = MazeSpec(
    width=8,
    height=8,
    include_ghosts=False,
    pellet_mode="full",
    surround_walls=True,
)
env_stage3 = PacmanEnv(Config(maze_spec=spec_stage3), human_mode=False, headless=True)

# Stage 4: introduce ghosts and power pellets.
spec_stage4 = MazeSpec(
    width=8,
    height=8,
    include_ghosts=True,
    ghost_positions=[(4, 4)],
    include_power_pellets=True,
)
env_stage4 = PacmanEnv(Config(maze_spec=spec_stage4), human_mode=False, headless=True)
```

Feel free to mix in your own `pellet_positions`, `ghost_positions`, or completely custom `maze_layout` strings for more advanced scenarios.

### Handcrafted Mazes

You can also plug in a traditional ASCII maze directly:

```python
maze = (
    "################",
    "#P..#......#..G#",
    "#.#.#.####.#.###",
    "#.#.#....#.#..##",
    "#.#.####.#.##.##",
    "#.#......#....##",
    "################",
)

config = Config(maze_layout=maze)
env = PacmanEnv(config, human_mode=False, headless=True)
```

Ensure every row has the same length so the layout forms a clean rectangular grid.

## Next Steps

- Extend `MazeSpec` or supply handcrafted layouts to shape more complex curricula.
- Add smarter ghost policies or multiple ghost variations.
- Use `record_trajectory` to capture state-action-reward tuples and seed imitation learning datasets.
