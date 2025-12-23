# 🧠 Dojo — Modular Reinforcement Learning Framework (Pac-Man Curriculum Trainer)

Dojo is a modular reinforcement learning (RL) framework designed around **curriculum-based DQN training**, built from scratch using **PyTorch + NumPy + Pygame**.  
The default environment is a grid-based **Pac-Man simulator**, but the codebase is structured to support custom environments and agents later.

---

## 🚀 Project Overview

The Dojo system is organized into three main layers:

| Layer | Description |
|:------|:-------------|
| **dojo_cli/** | Command-line interface (Click-based). Handles commands like `dojo train` and `dojo eval`. |
| **dojo_train_impl.py** | Core training loop and model management. Used by both the CLI and Python API. |
| **pacman_env.py** | The RL environment and game simulation (Pygame + NumPy). |
| **eval_agent.py** | Evaluation utilities and scoring logic (cross-size, random start, etc.). |
| **failed_run_recorder.py** | Saves failed trajectories for analysis and retraining. |
| **dojo_agent.py** | DQN agent definition, neural net, and optimizer setup. |
| **dojo_config.py** | Hyperparameters and runtime configuration. |

---

## 🧩 Installation

Clone and install the project in editable mode:

```bash
git clone https://github.com/yourname/dojo.git
cd dojo
pip install -e .
```

### 🧱 Dependencies
All dependencies are declared in `pyproject.toml` and automatically installed:

```
click>=8.1
torch>=2.0
numpy>=1.20
pygame>=2.1
pillow>=12.0.0
pytest>=8.4.2
pyclean>=3.2.0
```

---

## ⚙️ CLI Usage

Once installed, `dojo` is available globally.

### 🔹 Show help
```bash
dojo --help
```

### 🔹 Train a model
Train using a predefined curriculum (progressive mazes and pellet configurations):

```bash
dojo train
```

This runs through staged training in `dojo_train_impl.py`, saving checkpoints under `runs/`.

Each stage builds on the previous one — starting from small mazes (e.g., 4×4 single pellet) up to large “full” mazes.

### 🔹 Evaluate a trained model
Run the evaluation suite (random starts, random pellet layouts, and cross-size testing):

```bash
dojo eval
```

This runs:

1. `evaluate_full_model_random_pacman_start_same_size_map`
2. `evaluate_random_start_same_size_map`
3. `evaluate_cross_size`

Results are printed as normalized performance scores and combined into a **final evaluation score**.

Example:
```
🏁 Cross-size results: {4: 0.007, 8: 0.059, 12: -0.009}
results_1 (full same size): 0.677
results_2 (random same size): 0.027
Final evaluation score: 0.15
```

---

## 🧠 Architecture Overview

### 🧩 `pacman_env.py`

Implements a self-contained environment following a **Gym-like API**:

```python
env = PacmanEnv(Config(maze_spec=spec))
state = env.reset()
state, reward, done, info = env.step(action)
```

Key components:
- `MazeSpec`: Describes maze geometry and layout (pellets, ghosts, walls, etc.).
- `Maze`: Grid manager that handles pellet placement and wall validation.
- `Pacman` and `Ghost`: Movable entities with simple behavior.
- `PacmanEnv`: Top-level environment class with `.reset()`, `.step()`, `.render()`, and `.close()`.

Supports both **headless training** and **human play** modes.

### 🧩 `dojo_agent.py`

Defines the DQN agent, replay buffer, and model architecture.  
Includes:
- Experience replay
- Epsilon-greedy exploration
- Target-network synchronization
- Reward normalization and stability tricks

### 🧩 `dojo_train_impl.py`

Contains the core `train_stage` function:

```python
def train_stage(stage_name, maze_spec, episodes=10000, pretrained_path=None, spec_sampler=None, ...):
    ...
```

Handles:
- Stage-wise curriculum training
- Early stopping / convergence detection
- Learning rate decay and epsilon scheduling
- Model saving and checkpointing

### 🧩 `dojo_cli` package

Implements the Click-based CLI commands:

- `dojo train` — trains a model via curriculum progression
- `dojo eval` — evaluates trained models across conditions

Each subcommand maps directly to an imported Python function inside the Dojo framework.

---

## 🔬 Evaluation Strategy

Dojo includes a robust multi-stage evaluation designed to test both **memorization** and **generalization**.

### Evaluation Modes
| Mode | Description |
|------|--------------|
| `evaluate_full_model_random_pacman_start_same_size_map` | Randomizes Pac-Man’s start on a full pellet map (same size). |
| `evaluate_random_start_same_size_map` | Randomizes both start position and pellet layout. |
| `evaluate_cross_size` | Tests model on different maze sizes (e.g., 4×4, 8×8, 12×12). |

### Final Score Calculation
The CLI combines all results into a single composite score:

```python
final_score = (results_1 + results_2 + sum(results_3.values())) / (2 + len(results_3))
```

Higher is better. A strong model should maintain positive performance across all maze sizes.

---

## 💾 Project Structure

```bash
.
├── dojo.py                     # CLI entrypoint (installed as `dojo`)
├── dojo_train_impl.py          # Core training loop
├── dojo_agent.py               # DQN agent implementation
├── dojo_config.py              # Config + hyperparameters
├── eval_agent.py               # Evaluation logic
├── failed_run_recorder.py      # Records failed trajectories
├── pacman_env.py               # Game environment
├── dojo_cli/                   # CLI command definitions
│   ├── dojo_core.py
│   ├── dojo_curricula.py
│   ├── dojo_eval.py
│   ├── dojo_samplers.py
│   └── dojo_train.py
├── evaluated_models/           # Output dir for evaluations
├── failed_runs/                # Stored failure data
├── tests/                      # Unit tests
├── pyproject.toml              # Build + dependency manifest
└── README_NEW.md               # This file
```

---

## 🧰 Developer Workflow

### Clean environment
```bash
pyclean .
```

### Run unit tests
```bash
pytest -v
```

### Linting (optional)
```bash
ruff check .
```

### Reinstall after edits
```bash
pip uninstall -y dojo
pip install -e .
```

---

## 🧠 Extending Dojo

You can extend Dojo in multiple directions:

### 1️⃣ New Environment
Add a new environment (e.g., `grid_env.py`) following the same `reset/step/render` interface.

### 2️⃣ New Agent Architecture
Replace the DQN in `dojo_agent.py` with PPO, A3C, or custom attention-based architectures.

### 3️⃣ New Evaluation Metrics
Add scoring functions to `eval_agent.py` for more detailed performance metrics.

### 4️⃣ Curriculum Design
Extend `dojo_cli/dojo_curricula.py` to define new progressive training paths.

---

## 🧪 Example: Human Play Mode

```python
from pacman_env import Game

if __name__ == "__main__":
    Game().run()
```

This launches a playable Pac-Man instance using the same environment logic.

---

## 🧾 License

MIT License © 2025 John Furr

---

## 🧭 Next Steps

- [ ] Refactor for generic environment support (`EnvBase` abstraction)
- [ ] Package release to PyPI
- [ ] Add full test coverage for evaluation logic
- [ ] Integrate WandB or TensorBoard logging
