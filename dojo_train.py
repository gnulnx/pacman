import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import replace
from typing import Callable, List, Optional

import numpy as np
import torch

from dojo_agent import Agent
from pacman_env import ACTIONS, Config, MazeSpec, PacmanEnv

# dojo_train.py


TRAIN_MIN_EPISODES = 200
MEMORY = deque(maxlen=50000)


def set_global_seed(seed=None):
    """
    Sets random seeds for reproducibility and logs the seed used.
    If no seed is provided, generates one from current time.
    """
    if seed is None:
        seed = int(time.time()) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🧩 Using random seed: {seed}")
    return seed


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


def train_stage(
    stage_name,
    maze_spec: MazeSpec,
    episodes=10000,
    pretrained_path=None,
    spec_sampler: Optional[Callable[[], MazeSpec]] = None,
    eps_start=0.8,
    eps_decay=3000,
    warmup_episodes=50,
    lr_factor=1.0,
    max_steps=1000,
):
    """
    Train a DQN agent on a given Pac-Man maze with robust convergence control.
    This version:
      - Uses high-variance reward scaling (0–100) to maintain gradient signal.
      - Updates the target network slowly (every 200 episodes).
      - Keeps strong exploration early to avoid collapse.
      - Retains early-stop logic but with smoother tolerances.
    """

    def build_env(spec: MazeSpec) -> PacmanEnv:
        return PacmanEnv(Config(maze_spec=spec, max_steps=max_steps), human_mode=False, headless=True)

    # --- environment / agent setup ---
    current_spec = spec_sampler() if spec_sampler is not None else maze_spec
    env = build_env(current_spec)
    sample_state = preprocess_state(env.reset())
    obs_shape = sample_state.shape
    n_actions = len(ACTIONS)

    eps_decay = max(int(episodes / 3), eps_decay)
    print(f"training with eps_decay = {eps_decay}")
    agent = Agent(
        obs_shape,
        n_actions,
        eps_decay=eps_decay,
        eps_start=eps_start,
        eps_end=0.01,
        memory=MEMORY,
        # memory_size=50000 if current_spec.width <= 4 else 100000,
    )

    # reward scaling parameters — high-variance restored
    alpha = 0.5
    c = 100.0

    # --- load pretrained model (if provided) ---
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location="cpu")
        agent.model.load_state_dict(state_dict)
        agent.target.load_state_dict(agent.model.state_dict())
        print(f"✅ Loaded pretrained weights from {pretrained_path}")
        agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.lr * lr_factor)
        agent.set_epsilon(eps_start)

    os.makedirs(f"runs/{stage_name}", exist_ok=True)
    print(f"🚀 Starting training for {stage_name} ({current_spec.width}x{current_spec.height})")

    # --- bookkeeping ---
    recent_rewards, recent_success = [], []
    best_mean = -float("inf")
    best_model_path = f"runs/{stage_name}/best_model.pt"
    no_improve_counter, total_steps = 0, 0
    last_flush = time.time()
    min_train_episodes = TRAIN_MIN_EPISODES
    state = sample_state
    current_epsilon = getattr(agent, "epsilon", eps_start)
    agent.set_epsilon(current_epsilon)

    eval_interval = 100
    eval_episodes = 100
    eval_success_target = 1.0
    eval_step_target = min(15, max_steps)

    def run_greedy_evaluation() -> tuple[float, float]:
        """Run deterministic episodes to verify learning progress."""
        successes = 0
        total_steps_eval = 0
        for _ in range(eval_episodes):
            eval_spec = spec_sampler() if spec_sampler is not None else maze_spec
            eval_env = build_env(eval_spec)
            state_eval = preprocess_state(eval_env.reset())
            done_eval = False
            steps_used = 0
            info_eval: dict[str, int] = {}
            while not done_eval and steps_used < max_steps:
                action_eval = agent.select_action(state_eval, epsilon=0.0)
                next_state_eval, _, done_eval, info_eval = eval_env.step(action_eval)
                state_eval = preprocess_state(next_state_eval)
                steps_used += 1
            eval_env.close()
            if info_eval.get("pellets_remaining", 0) == 0:
                successes += 1
            total_steps_eval += steps_used
        return successes / eval_episodes, total_steps_eval / eval_episodes

    for ep in range(episodes):
        epsilon_for_episode = current_epsilon
        agent.set_epsilon(epsilon_for_episode)
        pellets_initial = int(env.maze.pellets.sum())
        episode_scale = (max(1, pellets_initial) ** (1 - alpha)) * c
        done, total_reward_scaled, steps_in_ep = False, 0.0, 0
        render_this_episode = ep % 100 == 0
        in_warmup = ep < warmup_episodes

        while not done:
            action = agent.select_action(state, epsilon=epsilon_for_episode)
            raw_next, reward_raw, done, info = env.step(action)

            # keep reward variance high
            reward_scaled = float((reward_raw / (max(1, pellets_initial) ** alpha)) * c)

            next_state = preprocess_state(raw_next)
            agent.remember((state, action, reward_scaled, next_state, done))
            state = next_state
            total_reward_scaled += reward_scaled
            steps_in_ep += 1
            total_steps += 1

            if not in_warmup and total_steps % 4 == 0:
                agent.replay(batch_size=32)
            if render_this_episode:
                env.render("human")

        # episode summary
        recent_rewards.append(total_reward_scaled)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        pellets_remaining = info.get("pellets_remaining", 0)
        recent_success.append(1.0 if pellets_remaining == 0 else 0.0)
        if len(recent_success) > 200:
            recent_success.pop(0)

        avg = np.mean(recent_rewards)
        std = np.std(recent_rewards)
        rel_std = std / (abs(avg) + 1e-8)
        success_rate = np.mean(recent_success) if recent_success else 0.0
        progress = min(avg / episode_scale, 1.0)

        # slower target updates — stability
        if ep % 200 == 0 and not in_warmup:
            agent.update_target()

        if ep % 50 == 0:
            eps_val = epsilon_for_episode
            torch.save(agent.model.state_dict(), f"runs/{stage_name}/model.pt")
            pellets_collected = (pellets_initial - pellets_remaining) / max(1, pellets_initial)

            if ep == 0:
                print(
                    f"Episode {ep:4d} | reward={total_reward_scaled:6.2f} | avg={avg:6.2f} | std={std:5.2f} "
                    f"| rel_std={rel_std*100:4.2f}% | eps={eps_val:.3f} | progress={progress*100:5.1f}% "
                    f"| success_rate={success_rate*100:5.1f}% | pellets={pellets_collected*100:5.1f}%"
                    + (" [WARM-UP]" if in_warmup else "")
                )

            avg_change = abs(avg - best_mean)
            if avg > best_mean + 0.01:
                best_mean = avg
                torch.save(agent.model.state_dict(), best_model_path)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # --- early stopping ---
            # print(
            #     "warmup:",
            #     in_warmup,
            #     "len(recent_rewards):",
            #     len(recent_rewards),
            #     "ep:",
            #     ep,
            #     "min_train_episodes:",
            #     min_train_episodes,
            # )
            # if not in_warmup and len(recent_rewards) == 100 and ep >= min_train_episodes:
            #     if success_rate >= 0.98 and rel_std <= 0.02:
            #         print(f"✅ Early stopping: solved at ep {ep}")
            #         torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
            #         env.close()
            #         return
            #     if success_rate >= 0.90 and rel_std <= 0.05 and no_improve_counter > 20:
            #         print(f"🟡 Plateau detected: stopping at ep {ep}")
            #         torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
            #         env.close()
            #         return
            #     if avg_change < 0.005 and rel_std <= 0.03:
            #         print(f"🟢 Converged mean at ep {ep}")
            #         torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
            #         env.close()
            #         return

        if ep % eval_interval == 0 and ep >= warmup_episodes and not in_warmup:
            greedy_success, greedy_steps = run_greedy_evaluation()
            print(
                f"Episode {ep:4d} | reward={total_reward_scaled:6.2f} | avg={avg:6.2f} | std={std:5.2f} "
                f"| rel_std={rel_std*100:4.2f}% | eps={eps_val:.3f} | progress={progress*100:5.1f}% "
                f"| 🔍 Greedy eval → success={greedy_success*100:5.1f}% | avg_steps={greedy_steps:4.1f}"
            )
            # print(f"   🔍 Greedy eval → success={greedy_success*100:5.1f}% | avg_steps={greedy_steps:4.1f}\n")
            if greedy_success >= eval_success_target and greedy_steps <= eval_step_target:
                print(
                    f"✅ Deterministic evaluation passed at episode {ep}: "
                    f"success={greedy_success*100:.1f}% avg_steps={greedy_steps:.1f}"
                )
                torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                env.close()
                return

        # --- resample new maze if needed ---
        if spec_sampler is not None:
            env.close()
            current_spec = spec_sampler()
            env = build_env(current_spec)
        state = preprocess_state(env.reset())

        # --- periodic MPS flush ---
        if torch.backends.mps.is_available() and time.time() - last_flush > 60:
            torch.mps.empty_cache()
            last_flush = time.time()

    env.close()
    print(f"✅ Training complete for {stage_name}")
    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")


def _single_pellet_position(width: int, height: int, start: tuple[int, int]) -> tuple[int, int]:
    for y in range(height):
        for x in range(width):
            if (x, y) != start:
                return x, y
    raise ValueError("No valid pellet locations available.")


def _random_episode_sampler(
    base_spec: MazeSpec,
    *,
    randomize_pacman: bool,
    randomize_single_target: bool,
    max_target_count: int = 32,
) -> Callable[[], MazeSpec]:
    """
    Build a callable that produces ``MazeSpec`` copies with optional randomisation.

    - For ``pellet_mode == "single"`` we either sample new pellet targets (between 1 and
      ``max_target_count``) or reuse the supplied list.
    - For structured modes (``"full"``, stripes, etc.) and ``"custom"`` we keep the pellet layout
      exactly as described in the base specification.
    """
    coords = [(x, y) for x in range(base_spec.width) for y in range(base_spec.height)]

    def choose_unique_positions(exclude: set[tuple[int, int]], count: int) -> List[tuple[int, int]]:
        available = [pos for pos in coords if pos not in exclude]
        if not available:
            return []
        count = min(count, len(available))
        return random.sample(available, count)

    def sampler() -> MazeSpec:
        pacman_start = random.choice(coords) if randomize_pacman else base_spec.pacman_start

        mode = base_spec.pellet_mode
        pellets: Optional[List[tuple[int, int]]]

        if mode == "single":
            base_positions = list(base_spec.pellet_positions or [])
            if randomize_single_target or not base_positions:
                target_count = max(1, max_target_count)
                pellet_count = random.randint(1, target_count)
                pellets = choose_unique_positions({pacman_start}, pellet_count)
                if not pellets:
                    pellets = [_single_pellet_position(base_spec.width, base_spec.height, pacman_start)]
            else:
                pellets = [pos for pos in base_positions if pos != pacman_start]
                desired = max(1, max_target_count)
                seen = set(pellets)
                while len(pellets) < desired:
                    extra = choose_unique_positions(seen | {pacman_start}, 1)
                    if not extra:
                        break
                    pellets.extend(extra)
                    seen.update(extra)
                if not pellets:
                    pellets = [_single_pellet_position(base_spec.width, base_spec.height, pacman_start)]
        else:
            pellets = list(base_spec.pellet_positions) if base_spec.pellet_positions else None

        spec = replace(
            base_spec,
            pacman_start=pacman_start,
            pellet_positions=pellets,
            random_seed=random.randint(0, 10**9),
        )

        # print(f"   → episode layout: start={spec.pacman_start}, pellets={spec.pellet_positions}")
        return spec

    return sampler


def hash_layout(spec: MazeSpec) -> str:
    # Compactly represent the maze layout (ignore pellet order)
    pellets_sorted = sorted(spec.pellet_positions or [])
    key = json.dumps((spec.pacman_start, pellets_sorted))
    return hashlib.md5(key.encode()).hexdigest()


def build_balanced_sampler(
    base_spec, randomize_pacman=True, randomize_single_target=True, max_target_count=32, pre_samples=50000
):
    # Step 1: Generate a large random set
    raw_sampler = _random_episode_sampler(
        base_spec,
        randomize_pacman=randomize_pacman,
        randomize_single_target=randomize_single_target,
        max_target_count=max_target_count,
    )

    counts = Counter()
    cached = []
    for _ in range(pre_samples):
        s = raw_sampler()
        h = hash_layout(s)
        counts[h] += 1
        cached.append((h, s))

    # Step 2: Compute inverse frequency weights
    total = sum(counts.values())
    weights = {h: (total / counts[h]) for h in counts}

    # Step 3: Build weighted sampling array
    population = [s for (h, s) in cached]
    probs = [weights[h] for (h, s) in cached]
    total_w = sum(probs)
    probs = [p / total_w for p in probs]

    def sampler():
        # pick one biased toward rarer configurations
        return random.choices(population, weights=probs, k=1)[0]

    # print(
    #     f"Balanced sampler built with {len(counts)} unique layouts across {pre_samples} samples. Total population: {len(population)}"
    # )
    # print(population)
    # input()
    return sampler


def build_stratified_sampler(
    base_spec: MazeSpec,
    *,
    randomize_pacman: bool = True,
    randomize_single_target: bool = True,
    max_target_count: int = 16,
    pre_samples: int = 5000,
    distance_bins: Optional[List[int]] = None,
) -> Callable[[], MazeSpec]:
    """
    Create a callable sampler that balances maze layouts by 'difficulty' strata:
      - Distance from Pac-Man start to nearest pellet
      - Number of pellets

    It first generates many random layouts (like a dry run), buckets them into
    distance ranges, and samples evenly from each bucket.

    Args:
        base_spec: Base MazeSpec to clone from
        randomize_pacman: Randomize Pac-Man start positions
        randomize_single_target: Randomize pellet positions when in 'single' mode
        max_target_count: Max pellets to sample for multi-target episodes
        pre_samples: Number of random layouts to precompute
        distance_bins: Custom distance breakpoints; defaults to quartiles
    """
    coords = [(x, y) for x in range(base_spec.width) for y in range(base_spec.height)]

    def choose_unique_positions(exclude: set[tuple[int, int]], count: int) -> List[tuple[int, int]]:
        available = [pos for pos in coords if pos not in exclude]
        if not available:
            return []
        count = min(count, len(available))
        return random.sample(available, count)

    # --- Phase 1: generate random layouts ---
    samples = []
    for _ in range(pre_samples):
        pacman_start = random.choice(coords) if randomize_pacman else base_spec.pacman_start
        if randomize_single_target:
            pellet_count = random.randint(1, max_target_count)
            pellets = choose_unique_positions({pacman_start}, pellet_count)
        else:
            pellets = base_spec.pellet_positions or choose_unique_positions({pacman_start}, 1)

        # Compute minimum Manhattan distance to any pellet
        px, py = pacman_start
        dist = min(abs(px - x) + abs(py - y) for (x, y) in pellets)
        samples.append((dist, pacman_start, pellets))

    # --- Phase 2: define difficulty bins ---
    dists = [s[0] for s in samples]
    max_d = max(dists)
    if distance_bins is None:
        # Default: quartile-like cutoffs
        distance_bins = [1, max(2, max_d // 3), max(3, 2 * max_d // 3), max_d + 1]
    bins = defaultdict(list)
    for dist, start, pellets in samples:
        for i, cutoff in enumerate(distance_bins):
            if dist <= cutoff:
                bins[i].append((dist, start, pellets))
                break

    print(f"Stratified sampler built with {len(samples)} layouts, {len(bins)} distance strata.")

    # --- Phase 3: uniform bucket sampler ---
    def sampler() -> MazeSpec:
        if not bins:
            raise RuntimeError("No buckets built for stratified sampler")
        bucket_id = random.choice(list(bins.keys()))
        dist, pac_start, pellets = random.choice(bins[bucket_id])
        spec = replace(
            base_spec,
            pacman_start=pac_start,
            pellet_positions=pellets,
            random_seed=random.randint(0, 10**9),
        )
        return spec

    return sampler


if __name__ == "__main__":

    seed = set_global_seed()  # call this FIRST
    # input("Press Enter to begin training...")

    stage = 1
    prev_final = None

    # --- 2x2 curriculum: exhaustive enumeration ---
    two_by_two_coords = [(x, y) for x in range(2) for y in range(2)]

    # Single-pellet cases
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
            train_stage(
                stage_name,
                maze_spec,
                pretrained_path=prev_final,
                eps_start=0.5,
                eps_decay=2000,
                max_steps=10,
            )
            prev_final = f"runs/{stage_name}/final_model.pt"
            stage += 1

    # print("Finish 2x2 single-pellet cases")
    # input()

    # Full-pellet cases
    # stage = 13
    # stage_name = f"stage{stage}"
    # prev_final = "runs/stage12/final_model.pt"
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
            eps_start=0.01,
            # eps_decay=2000,
            max_steps=3,  # very short for 2x2 full
        )
        prev_final = f"runs/{stage_name}/final_model.pt"
        stage += 1

    # print("\n~~ check modells ~~")
    # input()

    # --- 4x4 curriculum: structured starts ---
    # stage = 16
    # stage_name = f"stage{stage}"
    # prev_final = "runs/stage17/final_model.pt"
    coords_4 = [(x, y) for x in range(4) for y in range(4)]
    four_by_four_coords = [(x, y) for x in range(4) for y in range(4)]
    for start_x, start_y in four_by_four_coords:
        for pellet_x, pellet_y in four_by_four_coords:
            if (start_x, start_y) == (pellet_x, pellet_y):
                continue
            stage_name = f"stage{stage}"
            maze_spec = MazeSpec(
                width=4,
                height=4,
                include_ghosts=False,
                pellet_mode="single",
                pacman_start=(start_x, start_y),
                pellet_positions=[(pellet_x, pellet_y)],
                include_power_pellets=False,
                surround_walls=True,
            )
            print(
                f"\n=== Training {stage_name} [4x4 | single] "
                f"(start={start_x},{start_y} → pellet={pellet_x},{pellet_y}) ==="
            )
            train_stage(
                stage_name,
                maze_spec,
                pretrained_path=prev_final,
                eps_start=0.2,
                # eps_decay=2000,
                max_steps=32,
            )
            prev_final = f"runs/{stage_name}/final_model.pt"
            stage += 1

    # stage = 256
    # stage_name = f"stage{stage}"
    # prev_final = "runs/stage255/final_model.pt"
    coords_4 = [(x, y) for x in range(4) for y in range(4)]
    full_subset_4 = random.sample(coords_4, min(16, len(coords_4)))
    for start_x, start_y in full_subset_4:
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
            # eps_decay=4000,
            eps_start=0.2,
            max_steps=32,
        )
        prev_final = f"runs/{stage_name}/final_model.pt"
        stage += 1

    print("final model saves as:", prev_final)

    # # --- 4x4 random rehearsal ---
    # review_full_spec_4 = MazeSpec(
    #     width=4,
    #     height=4,
    #     include_ghosts=False,
    #     pellet_mode="full",
    #     # pacman_start=(0, 0),
    #     include_power_pellets=False,
    #     surround_walls=True,
    # )
    # full_sampler_4 = _random_episode_sampler(review_full_spec_4, randomize_pacman=True, randomize_single_target=False)
    # stage_name = f"stage{stage}"
    # print(f"\n=== Training {stage_name} [4x4 | full | random episodes] ===")
    # train_stage(stage_name, review_full_spec_4, pretrained_path=prev_final, spec_sampler=full_sampler_4)
    # prev_final = f"runs/{stage_name}/final_model.pt"
    # stage += 1

    # --- 4x4 random rehearsal ---
    # stage = 33
    # stage_name = f"stage{stage}"
    # prev_final = "runs/stage32/final_model.pt"
    # prev_final = "saved_models/4_x_4_mastered/final_model.pt"
    # review_full_spec_4 = MazeSpec(
    #     width=4,
    #     height=4,
    #     include_ghosts=False,
    #     pellet_mode="single",
    #     include_power_pellets=False,
    #     surround_walls=True,
    # )
    # full_sampler_4 = build_balanced_sampler(
    #     review_full_spec_4, randomize_pacman=True, randomize_single_target=False, max_target_count=16
    # )
    # train_stage(
    #     stage_name,
    #     review_full_spec_4,
    #     pretrained_path=prev_final,
    #     spec_sampler=full_sampler_4,
    #     eps_decay=5000,  # slower decay for more exploration
    #     episodes=20000,  # more episodes for mastery
    #     max_steps=50,  # force training to fail early so we optimize for faster solutions
    #     eps_start=0.1,
    # )
    # print(f"\n=== Training {stage_name} [4x4 | single | random episodes] ===")
    # prev_final = f"runs/{stage_name}/final_model.pt"
    # stage += 1

    # --- 8x8 curriculum: structured starts ---
    # prev_final = "saved_models/4_x_4_mastered/final_model.pt"
    # coords_8 = [(x, y) for x in range(8) for y in range(8)]

    # full_subset_8 = random.sample(coords_8, min(64, len(coords_8)))
    # for start_x, start_y in full_subset_8:
    #     stage_name = f"stage{stage}"
    #     maze_spec = MazeSpec(
    #         width=8,
    #         height=8,
    #         include_ghosts=False,
    #         pellet_mode="full",
    #         pacman_start=(start_x, start_y),
    #         include_power_pellets=False,
    #         surround_walls=True,
    #     )
    #     print(f"\n=== Training {stage_name} [8x8 | full] (start={start_x},{start_y}) ===")
    #     train_stage(stage_name, maze_spec, pretrained_path=prev_final)
    #     prev_final = f"runs/{stage_name}/final_model.pt"
    #     stage += 1

    # --- 8x8 random rehearsal ---
    # review_full_spec_8 = MazeSpec(
    #     width=8,
    #     height=8,
    #     include_ghosts=False,
    #     pellet_mode="full",
    #     pacman_start=(0, 0),
    #     include_power_pellets=False,
    #     surround_walls=True,
    # )
    # full_sampler_8 = _random_episode_sampler(review_full_spec_8, randomize_pacman=True, randomize_single_target=False)
    # stage_name = f"stage{stage}"
    # print(f"\n=== Training {stage_name} [8x8 | full | random episodes] ===")
    # train_stage(stage_name, review_full_spec_8, pretrained_path=prev_final, spec_sampler=full_sampler_8)
    # prev_final = f"runs/{stage_name}/final_model.pt"
    # stage += 1

    # prev_final = "saved_models/8_x_8_11262025/final_model.pt"
    # review_full_spec_8 = MazeSpec(
    #     width=8,
    #     height=8,
    #     include_ghosts=False,
    #     pellet_mode="single",
    #     # pacman_start=(0, 0),
    #     include_power_pellets=False,
    #     surround_walls=True,
    # )
    # full_sampler_8 = _random_episode_sampler(
    #     review_full_spec_8, randomize_pacman=True, randomize_single_target=False, max_target_count=64
    # )
    # stage_name = f"stage{stage}"
    # print(
    #     f"\n=== Training {stage_name} [8x8 | single | random episodes] pellet_positions {review_full_spec_8.pellet_positions}==="
    # )
    # train_stage(stage_name, review_full_spec_8, pretrained_path=prev_final, spec_sampler=full_sampler_8)
    # prev_final = f"runs/{stage_name}/final_model.pt"
    # stage += 1
