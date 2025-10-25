# Minimal Pac-Man RL Environment

This project provides a compact, modular Pac-Man implementation suitable for manual play and reinforcement learning experiments.

- pygame-based rendering and keyboard controls
- Gym-style `PacmanEnv` with `reset`, `step`, and `render`
- Randomly generated, fully traversable square mazes with braided corridors (size configurable)
- Configurable grid, colors, and frame rate via a `Config` dataclass
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
from pacman_env import Config, PacmanEnv, ACTIONS

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
4. Optionally fix the maze shape via `random_seed` for reproducible training runs.

## Next Steps

- Extend `Config.maze_layout` to build larger or curriculum-style mazes.
- Add smarter ghost policies or multiple ghost variations.
- Use `record_trajectory` to capture state-action-reward tuples and seed imitation learning datasets.
