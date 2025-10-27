import os
import random
import time
from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import torch

from dojo_agent import Agent
from pacman_env import ACTIONS, Config, MazeSpec, PacmanEnv


def preprocess_state(state):
    """Convert raw environment state dict to stacked numpy array for DQN input."""
    pellets = state["pellets"].astype(np.float32)
    pac = np.zeros_like(pellets)
    px, py = state["pacman"]
    pac[py, px] = 1.0
    ghosts = np.zeros_like(pellets)
    for gx, gy in state["ghosts"]:
        if 0 <= gx < pellets.shape[1] and 0 <= gy < pellets.shape[0]:
            ghosts[gy, gx] = 1.0
    stacked = np.stack([pellets, pac, ghosts], axis=0)
    return stacked


def _estimate_max_reward(env: PacmanEnv) -> float:
    """Approximate best-case episodic reward given the current layout."""
    pellet_count = int(env.maze.pellets.sum())
    # Each pellet yields +1, with a small step penalty; assume optimal path keeps penalties minimal.
    return max(1, pellet_count) * 1.0


def train_stage(
    stage_name,
    maze_spec: MazeSpec,
    episodes=10000,
    pretrained_path=None,
    spec_sampler: Optional[Callable[[], MazeSpec]] = None,
):
    """
    Train a DQN agent on a given Pac-Man maze with robust convergence detection.
    Automatically stops when performance stabilizes and approaches optimal reward.
    When `spec_sampler` is provided, a fresh MazeSpec is sampled every episode to vary starts/pellets.
    """

    # --- Environment setup ---
    current_spec = spec_sampler() if spec_sampler is not None else maze_spec
    env = PacmanEnv(Config(maze_spec=current_spec), human_mode=False, headless=True)
    sample_state = preprocess_state(env.reset())
    obs_shape = sample_state.shape
    n_actions = len(ACTIONS)
    agent = Agent(obs_shape, n_actions)

    # --- Optional weight transfer ---
    if pretrained_path and os.path.exists(pretrained_path):
        agent.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        agent.target.load_state_dict(agent.model.state_dict())
        print(f"✅ Loaded pretrained weights from {pretrained_path}")

        # 🧠 Reset optimizer & exploration to adapt to new stage dynamics
        agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.lr)  # clear old momentum
        agent.eps_start = 0.5  # restart exploration higher for the new stage
        agent.eps = agent.eps_start
    agent.steps = 0  # reset epsilon decay counter

    # --- Replay buffer scaling ---
    buffer_size = 2000 if current_spec.width <= 4 else 5000
    agent.memory = agent.memory.__class__(maxlen=buffer_size)

    os.makedirs(f"runs/{stage_name}", exist_ok=True)
    print(f"🚀 Starting training for {stage_name} ({current_spec.width}x{current_spec.height})")

    # --- Logging and convergence tracking ---
    recent_rewards = []
    recent_success = []
    best_mean = -float("inf")
    best_model_path = f"runs/{stage_name}/best_model.pt"
    no_improve_counter = 0
    total_steps = 0
    last_flush = time.time()

    # Estimate maximum achievable reward from the actual layout
    max_possible = _estimate_max_reward(env)
    min_train_episodes = max(500, current_spec.width * current_spec.height * 50)
    state = sample_state

    # Setup training hyperparameters
    eps_start = 1.0
    eps_end = 0.1
    eps_decay = 10000
    epsilon = eps_start

    for ep in range(episodes):
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-1.0 * total_steps / eps_decay)
        done = False
        total_reward = 0
        steps_in_ep = 0

        render_this_episode = ep % 100 == 0
        while not done:
            action = agent.select_action(state, epsilon)
            raw_next, reward, done, info = env.step(action)
            next_state = preprocess_state(raw_next)

            agent.remember((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps_in_ep += 1
            total_steps += 1

            # Learn periodically
            if total_steps % 10 == 0:
                agent.replay(batch_size=32)

            if render_this_episode:
                env.render("human")

        # --- Reward tracking ---
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        pellets_remaining = info.get("pellets_remaining", 0)
        recent_success.append(1.0 if pellets_remaining == 0 else 0.0)
        if len(recent_success) > 200:
            recent_success.pop(0)

        avg = np.mean(recent_rewards)
        std = np.std(recent_rewards)
        rel_std = std / (abs(avg) + 1e-8)
        progress = min(avg / max_possible, 1.0)
        success_rate = np.mean(recent_success) if recent_success else 0.0

        # --- Target update ---
        if ep % 20 == 0:
            agent.update_target()

        # --- Logging + checkpoint ---
        if ep % 50 == 0:
            eps_val = eps_end + (eps_start - eps_end) * np.exp(-1.0 * total_steps / eps_decay)
            torch.save(agent.model.state_dict(), f"runs/{stage_name}/model.pt")
            print(
                f"Episode {ep:4d} | reward={total_reward:6.2f} | avg={avg:6.2f} | std={std:5.2f} "
                f"| rel_std={rel_std*100:4.2f}% | eps={eps_val:.3f} | progress={progress*100:5.1f}%"
            )

            # --- Best model tracking ---
            avg_change = abs(avg - best_mean)
            if avg > best_mean + 0.01:
                best_mean = avg
                torch.save(agent.model.state_dict(), best_model_path)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # --- Early stopping logic ---
            if len(recent_rewards) == 100 and ep >= min_train_episodes:
                # 1️⃣ Solved: clears maze reliably with stable high reward
                # if success_rate >= 0.98 and avg >= 0.98 * max_possible and std <= 0.02 * max_possible:
                #     print(
                #         f"✅ Early stopping: solved (success={success_rate*100:.1f}%, avg={avg:.2f}, std={std:.2f}) at ep {ep}"
                #     )
                #     torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                #     env.close()
                #     return

                # 2️⃣ Plateaued: high performance but no improvement for a while
                if success_rate >= 0.90 and std <= 0.05 * max_possible and no_improve_counter > 20:
                    print(
                        f"🟡 Plateau detected: stopping (success={success_rate*100:.1f}%, avg={avg:.2f}, Δavg={avg_change:.3f})"
                    )
                    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                    env.close()
                    return

                # 3️⃣ Converged mean: flat trend with low variability
                if avg_change < 0.005 and std <= 0.03 * max_possible:
                    print(f"🟢 Converged mean: avg={avg:.2f}, Δavg={avg_change:.3f}, std={std:.2f} at ep {ep}")
                    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                    env.close()
                    return

        # Prepare next episode (potentially with fresh layout)
        if spec_sampler is not None:
            env.close()
            current_spec = spec_sampler()
            env = PacmanEnv(Config(maze_spec=current_spec), human_mode=False, headless=True)
            max_possible = _estimate_max_reward(env)
        state = preprocess_state(env.reset())

        # --- MPS/Metal cache maintenance ---
        if torch.backends.mps.is_available() and time.time() - last_flush > 60:
            torch.mps.empty_cache()
            last_flush = time.time()

    # --- Training complete fallback ---
    env.close()
    print(f"✅ Training complete for {stage_name}")
    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")


def _single_pellet_position(width: int, height: int, start: tuple[int, int]) -> tuple[int, int]:
    """Pick the first interior coordinate that does not overlap Pac-Man's start."""
    for y in range(height):
        for x in range(width):
            if (x, y) != start:
                return x, y
    raise ValueError("No valid pellet locations available.")


def _train_curriculum_for_grid(
    stage_counter: int, prev_model: str | None, size: int, modes: list[str], subset_length: int = 10
) -> tuple[int, str]:
    """Shared helper to run a sequence of curricula for a given grid size."""
    coords = [(x, y) for x in range(size) for y in range(size)]
    stage_path = prev_model

    for pellet_mode in modes:
        subset = random.sample(coords, min(subset_length, len(coords)))
        for start_x, start_y in subset:
            stage_name = f"stage{stage_counter}"
            single_positions = None
            if pellet_mode == "single":
                single_positions = [_single_pellet_position(size, size, (start_x, start_y))]
            maze_spec = MazeSpec(
                width=size,
                height=size,
                include_ghosts=False,
                pellet_mode=pellet_mode,
                pacman_start=(start_x, start_y),
                pellet_positions=single_positions,
                include_power_pellets=False,
                surround_walls=True,
            )
            print(
                f"\n=== Training {stage_name} [{size}x{size} | mode={pellet_mode}] (start={start_x},{start_y}) pellet_positions={single_positions} ==="
            )
            train_stage(stage_name, maze_spec, pretrained_path=stage_path)
            stage_path = f"runs/{stage_name}/final_model.pt"
            stage_counter += 1
    return stage_counter, stage_path


def _random_episode_sampler(
    base_spec: MazeSpec, *, randomize_pacman: bool, randomize_single_target: bool
) -> Callable[[], MazeSpec]:
    """Build a callable that returns fresh MazeSpecs with randomized starts/pellets each episode."""
    coords = [(x, y) for x in range(base_spec.width) for y in range(base_spec.height)]

    def sampler() -> MazeSpec:
        pacman_start = base_spec.pacman_start
        if randomize_pacman:
            pacman_start = random.choice(coords)

        pellet_positions = base_spec.pellet_positions
        if base_spec.pellet_mode == "single":
            if randomize_single_target:
                candidates = [pos for pos in coords if pos != pacman_start]
                if candidates:
                    pellet_positions = [random.choice(candidates)]
                else:
                    pellet_positions = [pacman_start]
            else:
                if pellet_positions and pellet_positions[0] == pacman_start:
                    candidates = [pos for pos in coords if pos != pacman_start]
                    if candidates:
                        pellet_positions = [random.choice(candidates)]
        # For full or other modes, pellet placement is implied by mode; keep existing configuration.

        return replace(
            base_spec,
            pacman_start=pacman_start,
            pellet_positions=pellet_positions,
            random_seed=random.randint(0, 10**9),
        )

    return sampler


if __name__ == "__main__":

    stage = 1
    prev_final = None

    two_by_two_coords = [(x, y) for x in range(2) for y in range(2)]

    # 2x2 curriculum – single pellet: enumerate every pacman/pellet pairing
    for start_x, start_y in two_by_two_coords:
        for pellet_x, pellet_y in two_by_two_coords:
            if (start_x, start_y) == (pellet_x, pellet_y):
                continue
            stage_name = f"stage{stage}"
            maze_spec = MazeSpec(
                width=2,
                height=2,
                include_ghosts=False,
                pellet_mode="single",
                pacman_start=(start_x, start_y),
                pellet_positions=[(pellet_x, pellet_y)],
                include_power_pellets=False,
                surround_walls=True,
            )
            print(
                f"\n=== Training {stage_name} [2x2 | single] "
                f"(start={start_x},{start_y} → pellet={pellet_x},{pellet_y}) ==="
            )
            train_stage(stage_name, maze_spec, pretrained_path=prev_final)
            prev_final = f"runs/{stage_name}/final_model.pt"
            stage += 1

    # 2x2 curriculum – full pellet: sweep every pacman start
    for start_x, start_y in two_by_two_coords:
        stage_name = f"stage{stage}"
        maze_spec = MazeSpec(
            width=2,
            height=2,
            include_ghosts=False,
            pellet_mode="full",
            pacman_start=(start_x, start_y),
            include_power_pellets=False,
            surround_walls=True,
        )
        print(f"\n=== Training {stage_name} [2x2 | full] (start={start_x},{start_y}) ===")
        train_stage(stage_name, maze_spec, pretrained_path=prev_final)
        prev_final = f"runs/{stage_name}/final_model.pt"
        stage += 1

    # 4x4 curricula (single then full pellets)
    stage, prev_final = _train_curriculum_for_grid(
        stage,
        prev_final,
        size=4,
        modes=["full"],
    )

    # 4x4 random rehearsal runs (full)
    review_full_spec = MazeSpec(
        width=4,
        height=4,
        include_ghosts=False,
        pellet_mode="full",
        pacman_start=(0, 0),
        include_power_pellets=False,
        surround_walls=True,
    )
    full_sampler = _random_episode_sampler(
        review_full_spec,
        randomize_pacman=True,
        randomize_single_target=False,
    )
    stage_name = f"stage{stage}"
    print(f"\n=== Training {stage_name} [4x4 | full | **random episodes**] ===")
    train_stage(stage_name, review_full_spec, pretrained_path=prev_final, spec_sampler=full_sampler)
    prev_final = f"runs/{stage_name}/final_model.pt"
    stage += 1

    # 8x8 curricula (single then full pellets)
    stage, prev_final = _train_curriculum_for_grid(
        stage,
        prev_final,
        size=8,
        subset_length=20,
        modes=["full"],
    )

    # 8x8 random rehearsal runs (full)
    review_full_spec_8 = MazeSpec(
        width=8,
        height=8,
        include_ghosts=False,
        pellet_mode="full",
        pacman_start=(0, 0),
        include_power_pellets=False,
        surround_walls=True,
    )
    full_sampler_8 = _random_episode_sampler(review_full_spec_8, randomize_pacman=True, randomize_single_target=False)
    stage_name = f"stage{stage}"
    print(f"\n=== Training {stage_name} [8x8 | full | **random episodes**] ===")
    train_stage(stage_name, review_full_spec_8, pretrained_path=prev_final, spec_sampler=full_sampler_8)
    prev_final = f"runs/{stage_name}/final_model.pt"
    stage += 1
