import os
import random
import time
from dataclasses import replace
from typing import Callable, List, Optional

import numpy as np
import torch

from dojo_agent import Agent
from pacman_env import ACTIONS, Config, MazeSpec, PacmanEnv

TRAIN_MIN_EPISODES = 500


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
    episodes=2000,
    pretrained_path=None,
    spec_sampler: Optional[Callable[[], MazeSpec]] = None,
    eps_start=0.5,
    warmup_episodes=25,
    lr_factor=0.25,
):
    """
    Train a DQN agent on a given Pac-Man maze with adaptive epsilon decay,
    early-stop criteria, and short warm-up. If a pretrained model is already
    near-optimal, it skips retraining entirely.
    """

    def build_env(spec: MazeSpec) -> PacmanEnv:
        return PacmanEnv(Config(maze_spec=spec), human_mode=False, headless=True)

    # --- environment / agent setup ---
    current_spec = spec_sampler() if spec_sampler is not None else maze_spec
    env = build_env(current_spec)
    sample_state = preprocess_state(env.reset())
    obs_shape = sample_state.shape
    n_actions = len(ACTIONS)

    eps_decay = max(int(episodes / 4), 1000)
    agent = Agent(obs_shape, n_actions, eps_decay=eps_decay, eps_start=eps_start)

    pellet_total = float(env.maze.pellets.sum())
    alpha, c = 0.5, 100.0
    max_possible = (pellet_total ** (1 - alpha)) * c  # expected max episodic return

    # --- load pretrained model ---
    if pretrained_path and os.path.exists(pretrained_path):
        agent.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        agent.target.load_state_dict(agent.model.state_dict())
        print(f"✅ Loaded pretrained weights from {pretrained_path}")
        agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.lr * lr_factor)
        agent.eps_start = eps_start
        agent.eps = eps_start

        # --- quick evaluation of pretrained model ---
        eval_env = build_env(maze_spec)
        total_eval_reward = 0
        for _ in range(5):
            s = preprocess_state(eval_env.reset())
            done = False
            while not done:
                a = agent.select_action(s, steps=99999)  # greedy eval
                nxt, r, done, _ = eval_env.step(a)
                s = preprocess_state(nxt)
                total_eval_reward += (r / (pellet_total**alpha)) * c
        eval_env.close()
        avg_eval = total_eval_reward / 5.0
        if avg_eval >= 0.9 * max_possible:
            print(f"⚡ Pretrained model already near-optimal ({avg_eval:.1f}); skipping retrain.")
            os.makedirs(f"runs/{stage_name}", exist_ok=True)
            torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
            return

    agent.steps = 0
    buffer_size = 10000 if current_spec.width <= 4 else 20000
    agent.memory = agent.memory.__class__(maxlen=buffer_size)

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

    # --- training loop ---
    for ep in range(episodes):
        done, total_reward, steps_in_ep = False, 0, 0
        render_this_episode = ep % 100 == 0
        in_warmup = ep < warmup_episodes

        while not done:
            action = agent.select_action(state, steps=ep)
            raw_next, reward, done, info = env.step(action)

            # --- reward scaling ---
            reward = float((reward / (pellet_total**alpha)) * c)

            next_state = preprocess_state(raw_next)
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps_in_ep += 1
            total_steps += 1

            if not in_warmup and total_steps % 10 == 0:
                agent.replay(batch_size=32)
            if render_this_episode:
                env.render("human")

        # --- metrics ---
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

        if ep % 20 == 0 and not in_warmup:
            agent.update_target()

        if ep % 50 == 0:
            eps_val = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-1.0 * ep / agent.eps_decay)
            torch.save(agent.model.state_dict(), f"runs/{stage_name}/model.pt")
            print(
                f"Episode {ep:4d} | reward={total_reward:6.2f} | avg={avg:6.2f} | std={std:5.2f} "
                f"| rel_std={rel_std*100:4.2f}% | eps={eps_val:.3f} | progress={progress*100:5.1f}% "
                f"| decay_rate={agent.eps_decay} | success_rate={success_rate*100:5.1f}%"
                + (" [WARM-UP]" if in_warmup else "")
            )

            avg_change = abs(avg - best_mean)
            if avg > best_mean + 0.01:
                best_mean = avg
                torch.save(agent.model.state_dict(), best_model_path)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # --- early-stop criteria (restored) ---
            if not in_warmup and len(recent_rewards) == 100 and ep >= min_train_episodes:
                if success_rate >= 0.98 and avg >= 0.98 * max_possible and std <= 0.02 * max_possible:
                    print(f"✅ Early stopping: solved at ep {ep}")
                    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                    env.close()
                    return
                if success_rate >= 0.90 and std <= 0.05 * max_possible and no_improve_counter > 20:
                    print(f"🟡 Plateau detected: stopping at ep {ep}")
                    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                    env.close()
                    return
                if avg_change < 0.005 and std <= 0.03 * max_possible:
                    print(f"🟢 Converged mean at ep {ep}")
                    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
                    env.close()
                    return

        # --- resample new maze if needed ---
        if spec_sampler is not None:
            env.close()
            current_spec = spec_sampler()
            env = build_env(current_spec)
            pellet_total = float(env.maze.pellets.sum())
            max_possible = (pellet_total ** (1 - alpha)) * c
        state = preprocess_state(env.reset())

        # --- occasional cache clear for MPS ---
        if torch.backends.mps.is_available() and time.time() - last_flush > 60:
            torch.mps.empty_cache()
            last_flush = time.time()

    env.close()
    print(f"✅ Training complete for {stage_name}")
    torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")


# def train_stage(
#     stage_name,
#     maze_spec: MazeSpec,
#     episodes=10000,
#     pretrained_path=None,
#     spec_sampler: Optional[Callable[[], MazeSpec]] = None,
#     eps_start=0.5,
# ):
#     """
#     Train a DQN agent on a given Pac-Man maze with robust convergence detection.
#     Automatically stops when performance stabilizes and approaches optimal reward.
#     When `spec_sampler` is provided, a fresh MazeSpec is sampled every episode.
#     """

#     def build_env(spec: MazeSpec) -> PacmanEnv:
#         return PacmanEnv(Config(maze_spec=spec), human_mode=False, headless=True)

#     current_spec = spec_sampler() if spec_sampler is not None else maze_spec
#     env = build_env(current_spec)
#     sample_state = preprocess_state(env.reset())
#     obs_shape = sample_state.shape
#     n_actions = len(ACTIONS)
#     agent = Agent(obs_shape, n_actions, eps_decay=10000, eps_start=eps_start)
#     pellet_total = float(env.maze.pellets.sum())
#     max_possible = 1.0  # rewards are normalised by pellet_total

#     if pretrained_path and os.path.exists(pretrained_path):
#         agent.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
#         agent.target.load_state_dict(agent.model.state_dict())
#         print(f"✅ Loaded pretrained weights from {pretrained_path}")
#         agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.lr)
#         agent.eps_start = 0.5
#         agent.eps = agent.eps_start
#     agent.steps = 0

#     # buffer_size = 2000 if current_spec.width <= 4 else 5000
#     buffer_size = 10000 if current_spec.width <= 4 else 20000
#     agent.memory = agent.memory.__class__(maxlen=buffer_size)

#     os.makedirs(f"runs/{stage_name}", exist_ok=True)
#     print(f"🚀 Starting training for {stage_name} ({current_spec.width}x{current_spec.height})")

#     recent_rewards = []
#     recent_success = []
#     best_mean = -float("inf")
#     best_model_path = f"runs/{stage_name}/best_model.pt"
#     no_improve_counter = 0
#     total_steps = 0
#     last_flush = time.time()

#     min_train_episodes = TRAIN_MIN_EPISODES
#     state = sample_state

#     for ep in range(episodes):
#         done = False
#         total_reward = 0
#         steps_in_ep = 0

#         render_this_episode = ep % 100 == 0
#         while not done:
#             action = agent.select_action(state, steps=ep)
#             raw_next, reward, done, info = env.step(action)

#             # Normalize the rewards for different maze sizes and pellet counts
#             reward = float(reward / pellet_total)

#             next_state = preprocess_state(raw_next)

#             agent.remember((state, action, reward, next_state, done))
#             state = next_state
#             total_reward += reward
#             steps_in_ep += 1
#             total_steps += 1

#             if total_steps % 10 == 0:
#                 agent.replay(batch_size=32)
#             if render_this_episode:
#                 env.render("human")

#         recent_rewards.append(total_reward)
#         if len(recent_rewards) > 100:
#             recent_rewards.pop(0)

#         pellets_remaining = info.get("pellets_remaining", 0)
#         recent_success.append(1.0 if pellets_remaining == 0 else 0.0)
#         if len(recent_success) > 200:
#             recent_success.pop(0)

#         avg = np.mean(recent_rewards)
#         std = np.std(recent_rewards)
#         rel_std = std / (abs(avg) + 1e-8)
#         progress = min(avg / max_possible, 1.0)
#         success_rate = np.mean(recent_success) if recent_success else 0.0

#         if ep % 20 == 0:
#             agent.update_target()

#         if ep % 50 == 0:
#             eps_val = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-1.0 * ep / agent.eps_decay)
#             torch.save(agent.model.state_dict(), f"runs/{stage_name}/model.pt")
#             print(
#                 f"Episode {ep:4d} | reward={total_reward:6.2f} | avg={avg:6.2f} | std={std:5.2f} "
#                 f"| rel_std={rel_std*100:4.2f}% | eps={eps_val:.3f} | progress={progress*100:5.1f}% | decay_rate={agent.eps_decay} | success_rate={success_rate*100:5.1f}%"
#             )

#             avg_change = abs(avg - best_mean)
#             if avg > best_mean + 0.01:
#                 best_mean = avg
#                 torch.save(agent.model.state_dict(), best_model_path)
#                 no_improve_counter = 0
#             else:
#                 no_improve_counter += 1

#             if len(recent_rewards) == 100 and ep >= min_train_episodes:
#                 if success_rate >= 0.98 and avg >= 0.98 * max_possible and std <= 0.02 * max_possible:
#                     print(
#                         f"✅ Early stopping: solved (success={success_rate*100:.1f}%, avg={avg:.2f}, std={std:.2f}) at ep {ep}"
#                     )
#                     torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
#                     env.close()
#                     return
#                 if success_rate >= 0.90 and std <= 0.05 * max_possible and no_improve_counter > 20:
#                     print(
#                         f"🟡 Plateau detected: stopping (success={success_rate*100:.1f}%, avg={avg:.2f}, Δavg={avg_change:.3f})"
#                     )
#                     torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
#                     env.close()
#                     return
#                 if avg_change < 0.005 and std <= 0.03 * max_possible:
#                     print(f"🟢 Converged mean: avg={avg:.2f}, Δavg={avg_change:.3f}, std={std:.2f} at ep {ep}")
#                     torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")
#                     env.close()
#                     return

#         if spec_sampler is not None:
#             env.close()
#             current_spec = spec_sampler()
#             env = build_env(current_spec)
#             pellet_total = float(env.maze.pellets.sum())
#             max_possible = 1.0
#         state = preprocess_state(env.reset())

#         if torch.backends.mps.is_available() and time.time() - last_flush > 60:
#             torch.mps.empty_cache()
#             last_flush = time.time()

#     env.close()
#     print(f"✅ Training complete for {stage_name}")
#     torch.save(agent.model.state_dict(), f"runs/{stage_name}/final_model.pt")


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


# def _random_episode_sampler(
#     base_spec: MazeSpec,
#     *,
#     randomize_pacman: bool,
#     randomize_single_target: bool,
#     max_target_count: int = 32,
# ) -> Callable[[], MazeSpec]:
#     """Return a sampler that emits fresh MazeSpecs with randomized start/pellet layouts.

#     The sampler clones the base specification each time, optionally randomizes Pac-Man's
#     starting tile, and chooses between 1 and `max_target_count` unique pellet positions.
#     For single-pellet curricula this allows rehearsal with varying pellet counts.
#     """

#     coords = [(x, y) for x in range(base_spec.width) for y in range(base_spec.height)]

#     def choose_unique_positions(exclude: set[tuple[int, int]], count: int) -> List[tuple[int, int]]:
#         """Helper that samples up to `count` unique coordinates avoiding `exclude`."""
#         available = [pos for pos in coords if pos not in exclude]
#         if not available:
#             return []
#         count = min(count, len(available))
#         return random.sample(available, count)

#     def sampler() -> MazeSpec:
#         # --- Pac-Man start ---
#         pacman_start = base_spec.pacman_start
#         if randomize_pacman:
#             pacman_start = random.choice(coords)

#         # --- Pellet selection ---
#         pellets: List[tuple[int, int]]
#         if randomize_single_target or not base_spec.pellet_positions:
#             # Randomize pellet count between 1 and max_target_count (inclusive).
#             target_count = max(1, max_target_count)
#             pellet_count = random.randint(1, target_count)
#             pellets = choose_unique_positions({pacman_start}, pellet_count)
#         else:
#             # Start from provided pellets; ensure uniqueness and avoid the start tile.
#             pellets = [pos for pos in base_spec.pellet_positions if pos != pacman_start]
#             seen = set(pellets)
#             target_count = max(1, max_target_count)
#             while len(pellets) < target_count:
#                 extra = choose_unique_positions(seen | {pacman_start}, 1)
#                 if not extra:
#                     break
#                 pellets.extend(extra)
#                 seen.update(extra)

#         spec = replace(
#             base_spec,
#             pacman_start=pacman_start,
#             pellet_positions=pellets,
#             random_seed=random.randint(0, 10**9),
#         )
#         # Debug print statement for visibility; comment out if too noisy.
#         print(f"   → episode layout: start={spec.pacman_start}, pellets={spec.pellet_positions}")
#         return spec

#     return sampler


if __name__ == "__main__":

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
            train_stage(stage_name, maze_spec, pretrained_path=prev_final)
            prev_final = f"runs/{stage_name}/final_model.pt"
            stage += 1

    # Full-pellet cases
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

    # --- 4x4 curriculum: structured starts ---
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
        train_stage(stage_name, maze_spec, pretrained_path=prev_final)
        prev_final = f"runs/{stage_name}/final_model.pt"
        stage += 1

    # --- 4x4 random rehearsal ---
    # stage = 17
    # stage_name = f"stage{stage}"
    # prev_final = "runs/stage16/final_model.pt"
    review_full_spec_4 = MazeSpec(
        width=4,
        height=4,
        include_ghosts=False,
        pellet_mode="full",
        pacman_start=(0, 0),
        include_power_pellets=False,
        surround_walls=True,
    )
    full_sampler_4 = _random_episode_sampler(review_full_spec_4, randomize_pacman=True, randomize_single_target=False)
    stage_name = f"stage{stage}"
    print(f"\n=== Training {stage_name} [4x4 | full | random episodes] ===")
    train_stage(stage_name, review_full_spec_4, pretrained_path=prev_final, spec_sampler=full_sampler_4)
    prev_final = f"runs/{stage_name}/final_model.pt"
    stage += 1

    # --- 4x4 random rehearsal ---
    review_full_spec_4 = MazeSpec(
        width=4,
        height=4,
        include_ghosts=False,
        pellet_mode="single",
        # pacman_start=(0, 0),
        include_power_pellets=False,
        surround_walls=True,
    )
    full_sampler_4 = _random_episode_sampler(
        review_full_spec_4, randomize_pacman=True, randomize_single_target=False, max_target_count=16
    )
    train_stage(stage_name, review_full_spec_4, pretrained_path=prev_final, spec_sampler=full_sampler_4)
    prev_final = f"runs/{stage_name}/final_model.pt"
    stage += 1

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
