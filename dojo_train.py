import os
import random
import time

import numpy as np
import torch

from dojo_agent import Agent
from pacman_env import ACTIONS, Config, MazeSpec, PacmanEnv


def preprocess_state(state):
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


from dojo_train import (
    preprocess_state,  # if preprocess_state is in same file, remove this import
)


def train_stage(stage_name, maze_spec, episodes=2000, pretrained_path=None):
    """
    Train a DQN agent on a given Pac-Man maze with robust convergence detection.
    Automatically stops when performance stabilizes and approaches optimal reward.
    """

    # --- Environment setup ---
    env = PacmanEnv(Config(maze_spec=maze_spec), human_mode=False, headless=True)
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
    buffer_size = 2000 if maze_spec.width <= 4 else 5000
    agent.memory = agent.memory.__class__(maxlen=buffer_size)

    os.makedirs(f"runs/{stage_name}", exist_ok=True)
    print(f"🚀 Starting training for {stage_name} ({maze_spec.width}x{maze_spec.height})")

    # --- Logging and convergence tracking ---
    recent_rewards = []
    best_mean = -float("inf")
    best_model_path = f"runs/{stage_name}/best_model.pt"
    no_improve_counter = 0
    total_steps = 0
    last_flush = time.time()

    # Estimate rough maximum possible reward
    max_possible = (maze_spec.width * maze_spec.height) * 1.0  # ≈ one pellet per tile

    for ep in range(episodes):
        state = preprocess_state(env.reset())
        done = False
        total_reward = 0
        steps_in_ep = 0

        render_this_episode = ep % 100 == 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state)

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

        avg = np.mean(recent_rewards)
        std = np.std(recent_rewards)
        rel_std = std / (abs(avg) + 1e-8)
        progress = min(avg / max_possible, 1.0)

        # --- Target update ---
        if ep % 20 == 0:
            agent.update_target()

        # --- Logging + checkpoint ---
        if ep % 50 == 0:
            eps_val = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-1.0 * agent.steps / agent.eps_decay)
            torch.save(agent.model.state_dict(), f"runs/{stage_name}/model.pt")
            print(
                f"Episode {ep:4d} | reward={total_reward:6.2f} | avg={avg:6.2f} | std={std:5.2f} "
                f"| rel_std={rel_std*100:4.2f}% | eps={eps_val:.3f} | progress={progress*100:5.1f}%"
            )

            # --- Best model tracking ---
            avg_change = abs(avg - best_mean)
            if avg > best_mean + 0.05:
                best_mean = avg
                torch.save(agent.model.state_dict(), best_model_path)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # --- Early stopping logic ---
            if len(recent_rewards) == 100:
                # 1️⃣ Solved: high performance + stable
                if progress > 0.95 and rel_std < 0.02:
                    print(
                        f"✅ Early stopping: solved (avg={avg:.2f}, rel_std={rel_std*100:.2f}%, "
                        f"progress={progress*100:.1f}%) at ep {ep}"
                    )
                    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                    env.close()
                    return

                # 2️⃣ Plateaued too long (no improvement + stable)
                if no_improve_counter > 250 and rel_std < 0.05:
                    print(
                        f"🟡 Plateau detected: stopping after {no_improve_counter} episodes "
                        f"(avg={avg:.2f}, rel_std={rel_std*100:.2f}%)"
                    )
                    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                    env.close()
                    return

                # 3️⃣ Converged mean (flat avg, stable)
                if avg_change < 0.01 and rel_std < 0.03:
                    print(
                        f"🟢 Converged mean: avg={avg:.2f}, Δavg={avg_change:.3f}, "
                        f"rel_std={rel_std*100:.2f}% at ep {ep}"
                    )
                    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                    env.close()
                    return

        # --- MPS/Metal cache maintenance ---
        if torch.backends.mps.is_available() and time.time() - last_flush > 60:
            torch.mps.empty_cache()
            last_flush = time.time()

    # --- Training complete fallback ---
    env.close()
    print(f"✅ Training complete for {stage_name}")
    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")


if __name__ == "__main__":

    spec_stage1 = MazeSpec(
        width=2,
        height=2,
        include_ghosts=False,
        pellet_mode="single",
        pacman_start=(0, 0),
        pellet_positions=[(1, 1)],
        include_power_pellets=False,
        surround_walls=True,
    )
    train_stage("stage1", spec_stage1)

    spec_stage2 = MazeSpec(
        width=2,
        height=2,
        include_ghosts=False,
        pellet_mode="single",
        pacman_start=(1, 1),
        pellet_positions=[(0, 0)],
        include_power_pellets=False,
        surround_walls=True,
    )
    train_stage("stage2", spec_stage2, pretrained_path="runs/stage1/final_model.pt")

    spec_stage3 = MazeSpec(
        width=2,
        height=2,
        include_ghosts=False,
        pellet_mode="single",
        pacman_start=(1, 0),
        pellet_positions=[(0, 1)],
        include_power_pellets=False,
        surround_walls=True,
    )
    train_stage("stage3", spec_stage3, pretrained_path="runs/stage2/final_model.pt")

    spec_stage3 = MazeSpec(
        width=2,
        height=2,
        include_ghosts=False,
        pellet_mode="single",
        pacman_start=(0, 1),
        pellet_positions=[(0, 1)],
        include_power_pellets=False,
        surround_walls=True,
    )
    train_stage("stage4", spec_stage3, pretrained_path="runs/stage3/final_model.pt")

    # 4x4 with single pellet and random start
    coords = [(x, y) for x in range(4) for y in range(4)]  # 0–3 inclusive
    subset = random.sample(coords, 10)
    for n, (start_x, start_y) in zip(range(5, 10), subset):  # stages 5 → 9
        stage_name = f"stage{n}"
        maze_spec = MazeSpec(
            width=4,
            height=4,
            pacman_start=(start_x, start_y),
            include_ghosts=False,
            pellet_mode="single",
            surround_walls=True,
        )

        # Load previous stage’s final model (e.g., stage4 → stage5)
        prev_stage = n - 1
        pretrained_path = f"runs/stage{prev_stage}/final_model.pt"

        print(f"\n=== Training {stage_name} (start={start_x},{start_y}) ===")
        train_stage(stage_name, maze_spec, pretrained_path=pretrained_path)

    # 4x4 with full pellet and random start
    subset = random.sample(coords, 10)
    for n, (start_x, start_y) in zip(range(10, 15), subset):  # stages 10 → 14
        stage_name = f"stage{n}"

        maze_spec = MazeSpec(
            width=4,
            height=4,
            pacman_start=(start_x, start_y),
            include_ghosts=False,
            pellet_mode="full",
            surround_walls=True,
        )

        # Load previous stage’s final model (e.g., stage4 → stage5)
        prev_stage = n - 1
        pretrained_path = f"runs/stage{prev_stage}/final_model.pt"

        print(f"\n=== Training {stage_name} (start={start_x},{start_y}) ===")
        train_stage(stage_name, maze_spec, pretrained_path=pretrained_path)

    # 8x8 with single pellet and random start
    coords = [(x, y) for x in range(8) for y in range(8)]  # 0–7 inclusive

    # Pick a random subset of 10 unique start positions
    subset = random.sample(coords, 10)

    for n, (start_x, start_y) in zip(range(15, 25), subset):  # stages 15 → 24
        stage_name = f"stage{n}"

        maze_spec = MazeSpec(
            width=8,
            height=8,
            pacman_start=(start_x, start_y),
            include_ghosts=False,
            pellet_mode="single",
            surround_walls=True,
        )

        # Load previous stage’s final model (e.g., stage4 → stage5)
        prev_stage = n - 1
        pretrained_path = f"runs/stage{prev_stage}/final_model.pt"

        print(f"\n=== Training {stage_name} (start={start_x},{start_y}) ===")
        train_stage(stage_name, maze_spec, pretrained_path=pretrained_path)

    # 8x8 with single pellet and random start
    subset = random.sample(coords, 10)
    for n, (start_x, start_y) in zip(range(25, 35), subset):  # stages 25 → 34
        stage_name = f"stage{n}"

        maze_spec = MazeSpec(
            width=8,
            height=8,
            pacman_start=(start_x, start_y),
            include_ghosts=False,
            pellet_mode="full",
            surround_walls=True,
        )

        # Load previous stage’s final model (e.g., stage4 → stage5)
        prev_stage = n - 1
        pretrained_path = f"runs/stage{prev_stage}/final_model.pt"

        print(f"\n=== Training {stage_name} (start={start_x},{start_y}) ===")
        train_stage(stage_name, maze_spec, pretrained_path=pretrained_path)
