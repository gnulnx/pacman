import os
import pickle
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


def save_state(
    state_dict: dict,
    stage_name: str,
    current_spec: MazeSpec,
    pretrained_path,
    episodes,
    max_possible,
    success_rate,
    avg,
    std,
    model_name="final_model.pt",
):
    """
    Save the model state and training configuration to disk.
    """
    os.makedirs(f"runs/{stage_name}", exist_ok=True)
    torch.save(state_dict, f"runs/{stage_name}/{model_name}")
    #  Also save input params as a pickle for easy loading later
    with open(f"runs/{stage_name}/config.pkl", "wb") as f:
        pickle.dump(
            {
                "maze_spec": current_spec,
                "stage_name": stage_name,
                "pretrained_path": pretrained_path,
                "episodes": episodes,
                "max_possible": max_possible,
                "success_rate": success_rate,
                "avg_reward": avg,
                "std_reward": std,
            },
            f,
        )


def train_stage(
    stage_name,
    maze_spec: MazeSpec,
    episodes=10000,
    pretrained_path=None,
    spec_sampler: Optional[Callable[[], MazeSpec]] = None,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay=10000,
):
    """
    Train a DQN agent on a given Pac-Man maze with robust convergence detection.
    Automatically stops when performance stabilizes and approaches optimal reward.
    When `spec_sampler` is provided, a fresh MazeSpec is sampled every episode to vary starts/pellets.
    """

    # --- Environment setup ---
    current_spec = spec_sampler() if spec_sampler is not None else maze_spec
    max_steps = current_spec.width * current_spec.height * 10  # simple heuristic
    env = PacmanEnv(
        Config(maze_spec=current_spec, max_steps=max_steps, fps=2000),
        human_mode=False,
        headless=True,
    )
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
    no_improve_counter = 0
    total_steps = 0
    last_flush = time.time()

    # Estimate maximum achievable reward from the actual layout
    max_possible = _estimate_max_reward(env)
    min_train_episodes = max(500, current_spec.width * current_spec.height * 50)
    state = sample_state

    # Setup training hyperparameters
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
            save_state(
                agent.model.state_dict(),
                stage_name,
                current_spec,
                pretrained_path,
                episodes,
                max_possible,
                success_rate,
                avg,
                std,
                model_name="model.pt",
            )
            print(
                f"Episode {ep:4d} | reward={total_reward:6.2f} | avg={avg:6.2f} | std={std:5.2f} "
                f"| rel_std={rel_std*100:4.2f}% | eps={eps_val:.3f} | progress={progress*100:5.1f}%"
            )

            # --- Best model tracking ---
            avg_change = abs(avg - best_mean)
            if avg > best_mean + 0.01:
                best_mean = avg
                save_state(
                    agent.model.state_dict(),
                    stage_name,
                    current_spec,
                    pretrained_path,
                    episodes,
                    max_possible,
                    success_rate,
                    avg,
                    std,
                    model_name="best_model.pt",
                )
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
                #     save_state(
                #         agent.model.state_dict(),
                #         stage_name,
                #         current_spec,
                #         pretrained_path,
                #         episodes,
                #         max_possible,
                #         success_rate,
                #         avg,
                #         std,
                #         model_name="final_model.pt",
                #     )

                #     env.close()
                #     return

                # 2️⃣ Plateaued: high performance but no improvement for a while
                if success_rate >= 0.90 and std <= 0.05 * max_possible and no_improve_counter > 20:
                    print(
                        f"🟡 Plateau detected: stopping (success={success_rate*100:.1f}%, avg={avg:.2f}, Δavg={avg_change:.3f})"
                    )
                    save_state(
                        agent.model.state_dict(),
                        stage_name,
                        current_spec,
                        pretrained_path,
                        episodes,
                        max_possible,
                        success_rate,
                        avg,
                        std,
                        model_name="final_model.pt",
                    )

                    env.close()
                    return

                # 3️⃣ Converged mean: flat trend with low variability
                if avg_change < 0.005 and std <= 0.03 * max_possible:
                    print(f"🟢 Converged mean: avg={avg:.2f}, Δavg={avg_change:.3f}, std={std:.2f} at ep {ep}")
                    save_state(
                        agent.model.state_dict(),
                        stage_name,
                        current_spec,
                        pretrained_path,
                        episodes,
                        max_possible,
                        success_rate,
                        avg,
                        std,
                        model_name="final_model.pt",
                    )

                    env.close()
                    return

        # Prepare next episode (potentially with fresh layout)
        if spec_sampler is not None:
            env.close()
            current_spec = spec_sampler()
            max_steps = current_spec.width * current_spec.height * 10  # simple heuristic
            env = PacmanEnv(
                Config(maze_spec=current_spec, max_steps=max_steps, fps=2000),
                human_mode=False,
                headless=True,
            )
            max_possible = _estimate_max_reward(env)
        state = preprocess_state(env.reset())

        # --- MPS/Metal cache maintenance ---
        if torch.backends.mps.is_available() and time.time() - last_flush > 60:
            torch.mps.empty_cache()
            last_flush = time.time()

    # --- Training complete fallback ---
    env.close()
    print(f"✅ Training complete for {stage_name}")
    save_state(
        agent.model.state_dict(),
        stage_name,
        current_spec,
        pretrained_path,
        episodes,
        max_possible,
        success_rate,
        avg,
        std,
        model_name="final_model.pt",
    )


def _single_pellet_position(width: int, height: int, start: tuple[int, int]) -> tuple[int, int]:
    """Pick the first interior coordinate that does not overlap Pac-Man's start."""
    for y in range(height):
        for x in range(width):
            if (x, y) != start:
                return x, y
    raise ValueError("No valid pellet locations available.")


def _train_curriculum_for_grid(
    stage_counter: int,
    prev_model: str | None,
    size: int,
    modes: list[str],
    subset_length: int = 10,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
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
            train_stage(stage_name, maze_spec, pretrained_path=stage_path, eps_end=eps_end, eps_start=eps_start)
            stage_path = f"runs/{stage_name}/final_model.pt"
            stage_counter += 1
    return stage_counter, stage_path


# def _random_episode_sampler(
#     base_spec: MazeSpec, *, randomize_pacman: bool, randomize_single_target: bool
# ) -> Callable[[], MazeSpec]:
#     """
#     Build a callable that returns fresh MazeSpecs with randomized starts/pellets each episode.
#     Each call will return a new maze spec with randomized Pac-Man start position and/or single pellet position.

#     """
#     coords = [(x, y) for x in range(base_spec.width) for y in range(base_spec.height)]

#     def sampler() -> MazeSpec:
#         pacman_start = base_spec.pacman_start
#         if randomize_pacman:
#             pacman_start = random.choice(coords)

#         pellet_positions = base_spec.pellet_positions
#         if base_spec.pellet_mode == "single":
#             if randomize_single_target:
#                 candidates = [pos for pos in coords if pos != pacman_start]
#                 if candidates:
#                     pellet_positions = [random.choice(candidates)]
#                 else:
#                     pellet_positions = [pacman_start]
#             else:
#                 if pellet_positions and pellet_positions[0] == pacman_start:
#                     candidates = [pos for pos in coords if pos != pacman_start]
#                     if candidates:
#                         pellet_positions = [random.choice(candidates)]
#         # For full or other modes, pellet placement is implied by mode; keep existing configuration.

#         return replace(
#             base_spec,
#             pacman_start=pacman_start,
#             pellet_positions=pellet_positions,
#             random_seed=random.randint(0, 10**9),
#         )

#     return sampler


def _random_episode_sampler(
    base_spec: MazeSpec,
    *,
    randomize_pacman: bool = True,
    randomize_single_target: bool = True,
    pellet_density: Optional[float] = None,
) -> Callable[[], MazeSpec]:
    """
    Build a callable that returns fresh MazeSpecs with randomized starts and/or pellet layouts each episode.

    This sampler enables randomized environment generation for curriculum or generalization training.
    Each call to the returned function produces a new MazeSpec instance based on `base_spec`, with
    randomized Pac-Man start position and/or pellet distribution depending on mode and parameters.

    ---
    **Parameters**
    - `base_spec` (MazeSpec): The reference MazeSpec defining maze size and static options (walls, ghosts, etc.).
    - `randomize_pacman` (bool): If True, Pac-Man's start position is randomized each episode.
    - `randomize_single_target` (bool): If True *and* `pellet_mode="single"`, the pellet position will also be randomized
      each episode (ensuring it does not overlap Pac-Man's start).
    - `pellet_density` (float | None): Optional float between 0.0 and 1.0 controlling the fraction of cells that will
      contain pellets for "custom" or "full" pellet modes.
        - If None → uses the base_spec’s default pellet layout.
        - If set (e.g., 0.25 → 25% of cells filled), randomizes pellet positions each episode.

    ---
    **Behavior Summary**
    - `pellet_mode="single"`:
        * If `randomize_single_target=True`: places exactly one pellet at a random location not equal to Pac-Man’s start.
        * If `randomize_single_target=False`: retains base_spec’s pellet position (unless overlapping Pac-Man, then re-randomizes).
    - `pellet_mode="full"` or `"custom"`:
        * If `pellet_density` is None: retains existing pellet distribution.
        * If `pellet_density` ∈ (0, 1]: randomly populates that fraction of grid cells with pellets each episode.
    - In all cases, Pac-Man’s start position and pellet positions are independent and reproducible per episode.

    ---
    **Usage Examples**

    1️⃣ *Randomize only Pac-Man’s start on full grid*
    ```python
    sampler = _random_episode_sampler(base_spec, randomize_pacman=True, randomize_single_target=False)
    ```

    2️⃣ *Random single-pellet navigation task*
    ```python
    sampler = _random_episode_sampler(base_spec, randomize_pacman=True, randomize_single_target=True)
    ```

    3️⃣ *Partial pellet density (e.g., 40% coverage)*
    ```python
    sampler = _random_episode_sampler(base_spec, randomize_pacman=True, pellet_density=0.4)
    ```

    4️⃣ *Fixed Pac-Man, new random pellets each episode*
    ```python
    sampler = _random_episode_sampler(base_spec, randomize_pacman=False, pellet_density=0.3)
    ```

    ---
    **Returns**
    - `Callable[[], MazeSpec]`: A function that produces a new randomized MazeSpec on each call.
    """
    coords = [(x, y) for x in range(base_spec.width) for y in range(base_spec.height)]

    def sampler() -> MazeSpec:
        pacman_start = base_spec.pacman_start
        if randomize_pacman:
            pacman_start = random.choice(coords)

        pellet_positions = base_spec.pellet_positions

        # --- Single pellet mode ---
        if base_spec.pellet_mode == "single":
            if randomize_single_target:
                candidates = [pos for pos in coords if pos != pacman_start]
                pellet_positions = [random.choice(candidates)] if candidates else [pacman_start]
            elif pellet_positions and pellet_positions[0] == pacman_start:
                candidates = [pos for pos in coords if pos != pacman_start]
                pellet_positions = [random.choice(candidates)] if candidates else [pacman_start]

        # --- Full/custom mode with pellet density ---
        elif pellet_density is not None and 0 < pellet_density <= 1.0:
            n_pellets = max(1, int(pellet_density * base_spec.width * base_spec.height))
            pellet_positions = random.sample(coords, n_pellets)

        # --- Otherwise: retain original configuration ---
        else:
            pellet_positions = base_spec.pellet_positions

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

    # === 2x2 curriculum – single pellet enumeration ===
    two_by_two_coords = [(x, y) for x in range(2) for y in range(2)]
    init = True
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
                f"\n=== Training {stage_name} [2x2 | single] (start={start_x},{start_y} → pellet={pellet_x},{pellet_y}) ==="
            )
            train_stage(
                stage_name,
                maze_spec,
                pretrained_path=prev_final,
                eps_start=0.9 if not init else 1.0,
                eps_end=0.05,
            )
            init = False
            prev_final = f"runs/{stage_name}/final_model.pt"
            stage += 1

    # === 2x2 curriculum – full pellets (4 starts) ===
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
        train_stage(
            stage_name,
            maze_spec,
            pretrained_path=prev_final,
            eps_start=0.8,
            eps_end=0.05,
        )
        prev_final = f"runs/{stage_name}/final_model.pt"
        stage += 1

    # === 4x4 curricula 1: full pellets with varied starts ===
    coords = [(x, y) for x in range(4) for y in range(4)]
    for start_x, start_y in coords:
        stage_name = f"stage{stage}"
        maze_spec = MazeSpec(
            width=4,
            height=4,
            include_ghosts=False,
            pellet_mode="full",
            pacman_start=(start_x, start_y),
            include_power_pellets=False,
            surround_walls=True,
        )
        print(f"\n=== Training {stage_name} [4x4 | full] (start={start_x},{start_y}) ===")
        train_stage(
            stage_name,
            maze_spec,
            pretrained_path=prev_final,
            eps_start=0.7,
            eps_end=0.05,
        )
        prev_final = f"runs/{stage_name}/final_model.pt"
        stage += 1

    # === 4x4 curricula 2: randomized single-pellet rehearsal ===
    single_spec_4x4 = MazeSpec(
        width=4,
        height=4,
        pellet_mode="single",
        include_ghosts=False,
        surround_walls=True,
    )
    single_sampler = _random_episode_sampler(
        single_spec_4x4,
        randomize_pacman=True,
        randomize_single_target=True,
    )
    stage_name = f"stage{stage}"
    print(f"\n=== Training {stage_name} [4x4 | single | **random episodes**] ===")
    train_stage(
        stage_name,
        single_spec_4x4,
        pretrained_path=prev_final,
        spec_sampler=single_sampler,
        eps_start=0.8,
        eps_end=0.05,
    )
    prev_final = f"runs/{stage_name}/final_model.pt"
    stage += 1

    # === 4x4 curricula 3: random varied pellet density ===
    density_spec_4x4 = MazeSpec(
        width=4,
        height=4,
        pellet_mode="custom",
        include_ghosts=False,
        surround_walls=True,
    )
    density_sampler = _random_episode_sampler(
        density_spec_4x4,
        randomize_pacman=True,
        pellet_density=0.25,
    )
    stage_name = f"stage{stage}"
    print(f"\n=== Training {stage_name} [4x4 | custom | **random density 25%**] ===")
    train_stage(
        stage_name,
        density_spec_4x4,
        pretrained_path=prev_final,
        spec_sampler=density_sampler,
        eps_start=0.8,
        eps_end=0.05,
    )
    prev_final = f"runs/{stage_name}/final_model.pt"
    stage += 1

    # === 4x4 curricula 4: randomized full-pellet rehearsal ===
    review_full_spec = MazeSpec(
        width=4,
        height=4,
        include_ghosts=False,
        pellet_mode="full",
        include_power_pellets=False,
        surround_walls=True,
    )
    full_sampler = _random_episode_sampler(
        review_full_spec,
        randomize_pacman=True,
    )
    stage_name = f"stage{stage}"
    print(f"\n=== Training {stage_name} [4x4 | full | **random episodes**] ===")
    train_stage(
        stage_name,
        review_full_spec,
        pretrained_path=prev_final,
        spec_sampler=full_sampler,
        eps_start=0.6,
        eps_end=0.10,
    )
    prev_final = f"runs/{stage_name}/final_model.pt"
    stage += 1

    print("✅ 4x4 phases complete. Last model:", prev_final)

    # === 8x8 curricula ===
    stage, prev_final = _train_curriculum_for_grid(
        stage,
        prev_final,
        size=8,
        subset_length=20,
        modes=["full"],
        eps_start=0.5,
        eps_end=0.05,
    )

    # === 8x8 random rehearsal (full) ===
    review_full_spec_8 = MazeSpec(
        width=8,
        height=8,
        include_ghosts=False,
        pellet_mode="full",
        include_power_pellets=False,
        surround_walls=True,
    )
    full_sampler_8 = _random_episode_sampler(
        review_full_spec_8,
        randomize_pacman=True,
    )
    stage_name = f"stage{stage}"
    print(f"\n=== Training {stage_name} [8x8 | full | **random episodes**] ===")
    train_stage(
        stage_name,
        review_full_spec_8,
        pretrained_path=prev_final,
        spec_sampler=full_sampler_8,
        eps_start=0.5,
        eps_end=0.05,
    )
    prev_final = f"runs/{stage_name}/final_model.pt"
    stage += 1


# if __name__ == "__main__":

#     stage = 1
#     prev_final = None

#     two_by_two_coords = [(x, y) for x in range(2) for y in range(2)]
#     init = True
#     # 2x2 curriculum – single pellet: enumerate every pacman/pellet pairing
#     for start_x, start_y in two_by_two_coords:
#         for pellet_x, pellet_y in two_by_two_coords:
#             if (start_x, start_y) == (pellet_x, pellet_y):
#                 continue
#             stage_name = f"stage{stage}"
#             maze_spec = MazeSpec(
#                 width=2,
#                 height=2,
#                 include_ghosts=False,
#                 pellet_mode="single",
#                 pacman_start=(start_x, start_y),
#                 pellet_positions=[(pellet_x, pellet_y)],
#                 include_power_pellets=False,
#                 surround_walls=True,
#             )
#             print(
#                 f"\n=== Training {stage_name} [2x2 | single] "
#                 f"(start={start_x},{start_y} → pellet={pellet_x},{pellet_y}) ==="
#             )
#             if init:
#                 init = False
#                 train_stage(stage_name, maze_spec, pretrained_path=prev_final)
#             else:
#                 train_stage(
#                     stage_name,
#                     maze_spec,
#                     pretrained_path=prev_final,
#                     eps_start=0.9,
#                     eps_end=0.05,
#                 )
#             prev_final = f"runs/{stage_name}/final_model.pt"
#             stage += 1

#     # 2x2 curriculum – full pellet: sweep every pacman start
#     for start_x, start_y in two_by_two_coords:
#         stage_name = f"stage{stage}"
#         maze_spec = MazeSpec(
#             width=2,
#             height=2,
#             include_ghosts=False,
#             pellet_mode="full",
#             pacman_start=(start_x, start_y),
#             include_power_pellets=False,
#             surround_walls=True,
#         )
#         print(f"\n=== Training {stage_name} [2x2 | full] (start={start_x},{start_y}) ===")
#         train_stage(
#             stage_name,
#             maze_spec,
#             pretrained_path=prev_final,
#             eps_start=0.8,
#             eps_end=0.05,
#         )
#         prev_final = f"runs/{stage_name}/final_model.pt"
#         stage += 1

#     # 4x4 curricula 1: full pellets with varied pacman starts
#     coords = [(x, y) for x in range(4) for y in range(4)]
#     for start_x, start_y in coords:
#         stage_name = f"stage{stage}"
#         single_positions = None
#         maze_spec = MazeSpec(
#             width=4,
#             height=4,
#             include_ghosts=False,
#             pellet_mode="full",
#             pacman_start=(start_x, start_y),
#             pellet_positions=single_positions,
#             include_power_pellets=False,
#             surround_walls=True,
#         )
#         print(
#             f"\n=== Training {stage_name} [4x4 | mode=full] (start={start_x},{start_y}) pellet_positions={single_positions} ==="
#         )
#         train_stage(
#             stage_name,
#             maze_spec,
#             pretrained_path=prev_final,
#             eps_start=0.7,
#             eps_end=0.05,
#         )
#         prev_final = f"runs/{stage_name}/final_model.pt"
#         stage += 1

#     # 4x4 curricula 2: random rehearsal runs (single pellet)
#     single_spec = MazeSpec(
#         width=4,
#         height=4,
#         pellet_mode="single",
#         include_ghosts=False,
#         surround_walls=True,
#     )

#     single_sampler = _random_episode_sampler(
#         single_spec,
#         randomize_pacman=True,
#         randomize_single_target=True,  # crucial
#     )
#     stage_name = f"stage{stage}"

#     train_stage(
#         stage_name,
#         single_spec,
#         pretrained_path=prev_final,
#         spec_sampler=single_sampler,
#         eps_start=0.6,
#         eps_end=0.05,
#     )
#     prev_final = f"runs/{stage_name}/final_model.pt"
#     stage += 1

#     # 4x4 curricula 3: random varied pellet density
#     single_spec = MazeSpec(
#         width=4,
#         height=4,
#         pellet_mode="custom",
#         include_ghosts=False,
#         surround_walls=True,
#     )

#     single_sampler = _random_episode_sampler(
#         single_spec,
#         randomize_pacman=True,
#         pellet_density=0.25,
#     )
#     stage_name = f"stage{stage}"

#     train_stage(
#         stage_name,
#         single_spec,
#         pretrained_path=prev_final,
#         spec_sampler=single_sampler,
#         eps_start=0.5,
#         eps_end=0.05,
#     )
#     prev_final = f"runs/{stage_name}/final_model.pt"
#     stage += 1

#     # 4x4 curricula 3: random episodes with full pellets
#     review_full_spec = MazeSpec(
#         width=4,
#         height=4,
#         include_ghosts=False,
#         pellet_mode="full",
#         include_power_pellets=False,
#         surround_walls=True,
#     )
#     # full_sampler() will return a new MazeSpec each episode with randomized pacman start and full pellets
#     full_sampler = _random_episode_sampler(
#         review_full_spec,
#         randomize_pacman=True,
#         # randomize_single_target=True,
#     )
#     stage_name = f"stage{stage}"
#     print(f"\n=== Training {stage_name} [4x4 | full | **random episodes**] ===")
#     train_stage(
#         stage_name,
#         review_full_spec,
#         pretrained_path=prev_final,
#         spec_sampler=full_sampler,
#         eps_start=0.4,
#         eps_end=0.1,
#     )
#     prev_final = f"runs/{stage_name}/final_model.pt"
#     stage += 1

#     print("4x4 is done.  last model:", prev_final)

#     # 8x8 curricula (single then full pellets)
#     # This is still starting from epsilon=1
#     stage, prev_final = _train_curriculum_for_grid(
#         stage,
#         prev_final,
#         size=8,
#         subset_length=20,
#         modes=["full"],
#         eps_start=0.3,
#         eps_end=0.05,
#     )

#     # 8x8 random rehearsal runs (full)
#     review_full_spec_8 = MazeSpec(
#         width=8,
#         height=8,
#         include_ghosts=False,
#         pellet_mode="full",
#         include_power_pellets=False,
#         surround_walls=True,
#     )
#     full_sampler_8 = _random_episode_sampler(review_full_spec_8, randomize_pacman=True, randomize_single_target=False)
#     stage_name = f"stage{stage}"
#     print(f"\n=== Training {stage_name} [8x8 | full | **random episodes**] ===")
#     train_stage(
#         stage_name,
#         review_full_spec_8,
#         pretrained_path=prev_final,
#         spec_sampler=full_sampler_8,
#         eps_start=0.20,
#         eps_end=0.05,
#     )
#     prev_final = f"runs/{stage_name}/final_model.pt"
#     stage += 1
