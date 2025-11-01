import os
import pickle
import random
import sys  # noqa
import time
from collections import deque
from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dojo_agent import Agent, ReplayBuffer  # noqa
from pacman_env import ACTIONS, Config, CuriousPacmanEnv, MazeSpec, PacmanEnv  # noqa


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

    # visit = state.get("visitation", np.zeros_like(pellets))
    # stacked = np.stack([pellets, pac, ghosts, visit], axis=0)
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
    output_dir=None,
    model_name="final_model.pt",
    replay_buffer: Optional[deque] = None,  # 👈 new
):
    """
    Save the model state, replay buffer, and training configuration to disk.
    """
    # Legacy behavior: default to runs/stage_name if no output_dir provided
    # if not output_dir:
    #     output_dir = f"runs/{stage_name}"

    os.makedirs(output_dir, exist_ok=True)

    # --- Save model ---
    torch.save(state_dict, f"{output_dir}/{model_name}")

    # --- Save training metadata ---
    with open(f"{output_dir}/config.pkl", "wb") as f:
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

    # --- Save replay buffer (optional) ---
    if replay_buffer is not None:
        # keep a small subset if buffer is huge
        buffer_path = f"{output_dir}/replay_buffer.pkl"
        with open(buffer_path, "wb") as f:
            pickle.dump(list(replay_buffer), f, protocol=pickle.HIGHEST_PROTOCOL)


def rolling_slope(series, window=500):
    if len(series) < 50:  # need at least 50 points to be meaningful
        return None
    w = min(window, len(series))
    xs = np.arange(w)
    ys = np.array(series[-w:])
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = np.sum(xs**2)
    if denom < 1e-6:
        return 0.0
    return np.sum(xs * ys) / denom


def train_stage(
    stage_name,
    maze_spec: MazeSpec,
    episodes=10000,
    pretrained_path=None,
    spec_sampler: Optional[Callable[[], MazeSpec]] = None,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay=10000,
    agent: Optional[Agent] = None,  # only pass agent if your maze_spec is the same size
    replay_batch_size=64,
    replay_buffer_size=100000,
    max_steps_per_episode: Optional[int] = None,
    prev_replay_buffer: Optional[deque] = None,
    output_dir="runs",
    decay_mode="exponential",  # "linear" or "exponential"
    Env=PacmanEnv,
    TBARL=None,  # 1) Sample from previous buffers for regression stop AND add those samples to the current buffer  0) Sample from previous buffers for regression stop ONLY
):
    """
    Train a DQN agent on a given Pac-Man maze with robust convergence detection.
    Automatically stops when performance stabilizes and approaches optimal reward.
    When `spec_sampler` is provided, a fresh MazeSpec is sampled every episode to vary starts/pellets.
    """

    # 📝 Initialize TensorBoard writer for this training stage
    output_dir = os.path.join(output_dir, stage_name)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))

    # Normalize and load any previous buffers
    loaded_prev_buffers = []
    print("Loading replay buffers")
    start = time.time()

    # --- Load previous buffer only if TBARL active and file provided ---
    loaded_prev_buffer = None
    if TBARL and prev_replay_buffer:
        path = prev_replay_buffer
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded_prev_buffer = pickle.load(f)
            print(f"Loaded teacher buffer: {path} ({len(loaded_prev_buffer)} samples)")

    labels = [label for *_, label in loaded_prev_buffers]
    print(f"Loaded replay buffers: {labels}")
    print("TBARL:", TBARL)

    # --- Environment setup ---
    current_spec = spec_sampler() if spec_sampler is not None else maze_spec
    if max_steps_per_episode is not None:
        max_steps = max_steps_per_episode
    else:
        max_steps = current_spec.width * current_spec.height * 10  # simple heuristic

    print("max_steps_per_episode:", max_steps)

    env = Env(
        Config(maze_spec=current_spec, max_steps=max_steps, fps=2000),
        human_mode=False,
        headless=True,
    )
    sample_state = preprocess_state(env.reset())
    print("Pacman at:", env.pacman.position)
    print("Maze shape:", env.maze.width, env.maze.height)

    obs_shape = sample_state.shape
    n_actions = len(ACTIONS)

    # --- Setup the agent with appropriate memory size ---
    if agent is None:
        agent = Agent(obs_shape, n_actions, memory_size=replay_buffer_size)
    else:
        # Since you are reusing the agent it's important to reset optimizer state
        agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.lr)

    # --- Optional weight transfer ---
    if pretrained_path and os.path.exists(pretrained_path):
        agent.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        agent.target.load_state_dict(agent.model.state_dict())
        print(f"✅ Loaded pretrained weights from {pretrained_path}")

        # 🧊 Optional: freeze conv layers if model was trained on a larger maze
        try:
            # detect smaller environment vs pretrained map size
            with open(os.path.join(os.path.dirname(pretrained_path), "config.pkl"), "rb") as f:
                prev_cfg = pickle.load(f)
            prev_spec = prev_cfg.get("maze_spec", None)

            if prev_spec and (prev_spec.width > current_spec.width or prev_spec.height > current_spec.height):
                print(f"🧠 Detected smaller maze ({current_spec.width}x{current_spec.height}); freezing conv layers")
                for name, param in agent.model.named_parameters():
                    if "conv" in name:
                        param.requires_grad = False
        except Exception as e:
            print(f"⚠️ Could not read previous spec for auto-freeze: {e}")

    # --- Adjust replay buffer size if needed ---
    if replay_buffer_size != agent.memory.maxlen:
        print(f"🔄 Growing replay buffer from {agent.memory.maxlen} → {replay_buffer_size}")
        new_rb = ReplayBuffer(replay_buffer_size)
        for t in agent.memory:  # preserves (s,a,r,s',done[,origin])
            new_rb.push(t if len(t) == 6 else t, origin=(t[5] if len(t) == 6 else "self"))
        agent.memory = new_rb

    os.makedirs(f"runs/{stage_name}", exist_ok=True)
    print(f"🚀 Starting training for {stage_name} ({current_spec.width}x{current_spec.height})")

    # --- Logging and convergence tracking ---
    recent_rewards = []
    recent_success = []
    reward_history = []
    success_history = []
    best_mean = -float("inf")
    no_improve_counter = 0
    total_steps = 0
    last_flush = time.time()

    # Estimate maximum achievable reward from the actual layout
    max_possible = _estimate_max_reward(env)
    # min_train_episodes = max(500, current_spec.width * current_spec.height * 50)
    min_train_episodes = 500
    state = sample_state

    # Setup training hyperparameters
    epsilon = eps_start

    for ep in range(episodes):
        episode_start = time.time()
        # Decay epsilon over time
        # epsilon = eps_end + (eps_start - eps_end) * np.exp(-1.0 * total_steps / eps_decay)

        # Linear epsilon decay
        warmup_episodes = 50
        if ep < warmup_episodes:
            epsilon = eps_start
        else:
            if decay_mode == "linear":
                # Linear epsilon decay
                epsilon = max(
                    eps_end, eps_start - ((ep - warmup_episodes) / (episodes - warmup_episodes)) * (eps_start - eps_end)
                )

            elif decay_mode == "exponential":
                # Exponential epsilon decay
                # eps = eps_end + (eps_start - eps_end) * exp(-k * (ep - warmup))
                # where k chosen so eps ≈ eps_end at final episode
                decay_rate = np.log(eps_start / eps_end) / (episodes - warmup_episodes)
                epsilon = max(eps_end, eps_start * np.exp(-decay_rate * (ep - warmup_episodes)))

        done = False
        total_reward = 0
        steps_in_ep = 0
        prior_replay_samples = 0
        replay_total_time = 0
        replay_calls = 0

        render_this_episode = ep % 100 == 0
        while not done:
            action = agent.select_action(state, epsilon)
            raw_next, reward, done, info = env.step(action)
            next_state = preprocess_state(raw_next)

            agent.remember((state, action, reward, next_state, done))
            # if info.get("pellets_remaining", 1) == 0:  # bias towards cleared mazes.  Prolly should be a param
            #     agent.remember((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward
            steps_in_ep += 1
            total_steps += 1

            # Learn periodically
            # --- TBARL: percent time to cross-sample from teacher replay buffers ---
            if loaded_prev_buffers and random.random() < TBARL:  # 25% chance to activate TBARL this step
                samples = random.sample(loaded_prev_buffers, min(len(loaded_prev_buffers), replay_batch_size))
                replay_total_time = agent.replay(batch_size=replay_batch_size, buffer=samples)
                if replay_total_time:
                    replay_total_time += replay_total_time
                    replay_calls += 1
            else:
                # Standard self replay
                replay_total_time = agent.replay(batch_size=replay_batch_size)
                if replay_total_time:
                    replay_total_time += replay_total_time
                    replay_calls += 1

            if render_this_episode and not env.headless:
                print("Rendering episode", ep)
                env.render("human")

        # --- Reward tracking ---
        recent_rewards.append(total_reward)
        reward_history.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        if len(reward_history) > 2000:
            reward_history.pop(0)

        pellets_remaining = info.get("pellets_remaining", 0)
        recent_success.append(1.0 if pellets_remaining == 0 else 0.0)
        success_history.append(1.0 if pellets_remaining == 0 else 0.0)
        if len(recent_success) > 200:
            recent_success.pop(0)
        if len(success_history) > 2000:
            success_history.pop(0)

        avg = np.mean(recent_rewards)
        std = np.std(recent_rewards)
        rel_std = std / (abs(avg) + 1e-8)
        progress = min(avg / max_possible, 1.0)

        # This should trend towards 1.0 as agent learns to clear mazes reliably
        success_rate = np.mean(recent_success) if recent_success else 0.0

        # --- Target update ---
        if ep > 0 and ep % 20 == 0:
            agent.update_target()

        if ep > 0 and ep % 50 == 0:
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
                model_name="last_model.pt",
                replay_buffer=agent.memory,
                output_dir=f"{output_dir}",
            )

            total_episode_time = time.time() - episode_start
            replay_buffer_report = agent.memory.report()
            avg_replay_time = replay_total_time / max(replay_calls, 1)
            print(
                f"Episode {ep:4d} | reward={total_reward:6.2f} | success_rate_100={success_rate*100:5.1f}% | reward/avg_100={avg:6.2f} | reward/std_100={std:5.2f} "
                f"| rel_std={rel_std*100:4.2f}% | eps={epsilon:.3f} | progress={progress*100:5.1f}% | total_steps={total_steps:,} "
                f"| episode_time={total_episode_time:.2f}s | len(recent_rewards)={len(recent_rewards)} | replay_buffer_report={replay_buffer_report} |  replay_avg={avg_replay_time:.4f}s | replay_total={replay_total_time:.4f}s"
            )

            # --- Best model tracking ---
            avg_change = abs(avg - best_mean)
            if avg > best_mean + 0.01:
                print("saving for new best model with avg reward:", avg)
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
                    replay_buffer=agent.memory,
                    output_dir=f"{output_dir}",
                )
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # --- Early stopping logic ---
            if len(recent_rewards) >= 200 and ep >= min_train_episodes:
                ready_to_save = False

                slope_reward = rolling_slope(recent_rewards, 1000)
                slope_success = rolling_slope(recent_success, 1000)
                mean_r = np.mean(recent_rewards)
                std_r = np.std(recent_rewards)
                success_rate = np.mean(recent_success)

                # --- Normalized metrics ---
                rel_mean = mean_r / (max_possible + 1e-8)
                rel_std = std_r / (max_possible + 1e-8)
                slope_r_norm = (slope_reward or 0) / (max_possible + 1e-8)

                if success_rate >= 0.97 and rel_mean >= 0.90 and rel_std <= 0.12:
                    print(f"✅ Solved: succ={success_rate:.3f}, rel_mean={rel_mean:.2f}, rel_std={rel_std:.2f}")
                    ready_to_save = True

                elif success_rate >= 0.90 and abs(slope_r_norm) < 1e-4 and rel_std <= 0.15:
                    print(f"🟡 Plateau: slope={slope_r_norm:.6f}, succ={success_rate:.3f}, rel_std={rel_std:.2f}")
                    ready_to_save = True

                elif abs(slope_r_norm) < 1e-4 and (slope_success is not None and abs(slope_success) < 1e-4):
                    print(f"🟢 Converged: slope_r={slope_r_norm:.5f}, slope_s={slope_success:.5f}")
                    ready_to_save = True

                if ready_to_save:
                    save_state(
                        agent.model.state_dict(),
                        stage_name,
                        current_spec,
                        pretrained_path,
                        episodes,
                        max_possible,
                        success_rate,
                        mean_r,
                        std_r,
                        model_name="final_model.pt",
                        replay_buffer=agent.memory,
                        output_dir=output_dir,
                    )
                    env.close()
                    return agent

        if ep % 10 == 0:  # or every 20, whatever granularity you prefer
            writer.add_scalar("reward/episode", total_reward, ep)
            writer.add_scalar("reward/avg_100", np.mean(recent_rewards), ep)
            writer.add_scalar("reward/std_100", np.std(recent_rewards), ep)
            writer.add_scalar("exploration/epsilon", epsilon, ep)
            writer.add_scalar("memory/fill_ratio", len(agent.memory) / agent.memory.maxlen, ep)
            writer.add_scalar("success/rate", success_rate, ep)
            writer.add_scalar("replay/prior_samples", prior_replay_samples, ep)

        # Prepare next episode (potentially with fresh layout)
        if spec_sampler is not None:
            env.close()
            current_spec = spec_sampler()
            max_steps = current_spec.width * current_spec.height * 10  # simple heuristic
            env = Env(
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
        replay_buffer=agent.memory,
        output_dir=f"{output_dir}",
    )
    writer.close()
    return agent


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
    output_dir = "curiosity_runs"
    # # prev_final = "saved_models/0.7352_stage3_best_model.pt"

    # Playign with new CuriousPacmanEnv that is a bit faster for just training agents
    agent = None
    stage_name = "curiousity_pretrain"

    width = 4
    height = 4

    sampler = _random_episode_sampler(
        MazeSpec(width=width, height=height, pellet_mode="custom", surround_walls=True),
        randomize_pacman=True,
        pellet_density=0,  # no pellets in curious agent
    )
    print(f"\n=== Training {stage_name} [{width}x{height} | curiosity] (random start) ===")
    agent = train_stage(
        stage_name,
        spec_sampler=sampler,
        maze_spec=MazeSpec(width=width, height=height, pellet_mode="custom", surround_walls=True),
        pretrained_path=prev_final,
        eps_start=0.8,
        eps_end=0.05,
        agent=agent,
        episodes=5000,
        replay_buffer_size=5000,
        output_dir=output_dir,
        Env=CuriousPacmanEnv,
        max_steps_per_episode=20,
    )
    prev_final = f"{output_dir}/{stage_name}/final_model.pt"
    print("Curiousity agent trains and saved at:", f"{output_dir}/{stage_name}/final_model.pt")
    stage += 1
    sys.exit(1)

    # # === 2x2 curriculum – single pellet enumeration ===
    # stages 1 through 8
    two_by_two_coords = [(x, y) for x in range(2) for y in range(2)]
    init = True
    agent = None
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
            agent = train_stage(
                stage_name,
                maze_spec,
                pretrained_path=prev_final,
                eps_start=0.9 if init else 0.5,
                eps_end=0.05,
                agent=agent,
                episodes=3000,
                replay_buffer_size=5000,
                output_dir=output_dir,
            )
            init = False
            prev_final = f"{output_dir}/{stage_name}/final_model.pt"
            stage += 1

    # # === 2x2 curriculum – full pellets (4 starts) ===
    # stages 9 through 12
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
        agent = train_stage(
            stage_name,
            maze_spec,
            pretrained_path=prev_final,
            eps_start=0.8,
            eps_end=0.05,
            agent=agent,
            episodes=5000,
            replay_buffer_size=5000,
            output_dir=output_dir,
        )
        prev_final = f"{output_dir}/{stage_name}/final_model.pt"
        stage += 1

    # === 4x4 density curricula ===
    # stages 16 through 22
    for density in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print("4x4 - running density:", density)
        sampler = _random_episode_sampler(
            MazeSpec(width=4, height=4, pellet_mode="custom", surround_walls=True),
            randomize_pacman=True,
            pellet_density=density,
        )
        agent = train_stage(
            f"stage{stage}",
            MazeSpec(width=4, height=4, pellet_mode="custom", surround_walls=True),
            pretrained_path=prev_final,
            spec_sampler=sampler,
            eps_start=0.5,
            eps_end=0.01,
            episodes=20000,
            agent=agent,
            replay_batch_size=32,
            replay_buffer_size=20000,
            output_dir=output_dir,
        )
        prev_final = f"{output_dir}/stage{stage}/final_model.pt"
        print("completed density:", density)
        stage += 1

    # prev_final = "saved_models/0.7352_stage3_best_model.pt"
    # agent = None
    # === 8x8 density curricula ===
    # stages 23 through 31
    for density in [0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
        print("8x8 - running density:", density)
        sampler = _random_episode_sampler(
            MazeSpec(width=8, height=8, pellet_mode="custom", surround_walls=True),
            randomize_pacman=True,
            pellet_density=density,
        )
        agent = train_stage(
            f"stage{stage}",
            MazeSpec(width=8, height=8, pellet_mode="custom", surround_walls=True),
            pretrained_path=prev_final,
            spec_sampler=sampler,
            eps_start=0.5,
            eps_end=0.05,
            # eps_decay=200000,  # using linear decay now
            episodes=20000,
            agent=agent,
            replay_batch_size=64,
            replay_buffer_size=50000,
            output_dir=output_dir,
        )
        prev_final = f"{output_dir}/stage{stage}/final_model.pt"
        print("completed density:", density)
        stage += 1
