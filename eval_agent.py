import copy  # noqa
import math  # noqa
import os  # noqa
import pickle  # noqa
import random  # noqa
import shutil  # noqa
import time  # noqa
from dataclasses import replace  # noqa

import torch  # noqa

from dojo_agent import Agent  # noqa
from dojo_train import preprocess_state  # noqa
from pacman_env import (  # noqa
    ACTIONS,
    Config,
    MazeSpec,
    PacmanEnv,
    generate_rect_layout,
)

# Exact runs each time
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)


# def evaluate(
#     model_path, maze_spec, episodes=5, delay=0.25, fps=100, random_pacman_start=False, show_pacman=False, output=True
# ):
#     # 1) Freeze the layout explicitly from the spec
#     layout = tuple(generate_rect_layout(maze_spec))
#     cfg = Config(maze_layout=layout)  # <- NOT maze_spec
#     env = PacmanEnv(cfg, human_mode=False, headless=False)
#     sample_state = preprocess_state(env.reset())
#     agent = Agent(sample_state.shape, len(ACTIONS))
#     agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     agent.model.eval()

#     if output:
#         print(f"🎯 Evaluating {model_path} on {maze_spec.width}x{maze_spec.height} maze...")
#     total_reward = 0.0
#     total_steps = 0
#     for ep in range(episodes):
#         if random_pacman_start:
#             start_x = random.randint(0, spec.width - 1)
#             start_y = random.randint(0, spec.height - 1)
#             start_x = 1
#             start_y = 1
#             print("Randomizing Pacman start to:", (start_x, start_y))
#             spec.pacman_start = (start_x, start_y)

#             # Reinit layout and env with new pacman start
#             layout = tuple(generate_rect_layout(maze_spec))
#             cfg = Config(maze_layout=layout, max_steps=200, fps=fps)
#             env = PacmanEnv(cfg, human_mode=False, headless=False)

#             env.reset()

#         state = preprocess_state(env.reset())
#         done = False
#         total_episode_reward = 0.0
#         total_episode_steps = 0
#         while not done:
#             with torch.no_grad():
#                 s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
#                 action = int(torch.argmax(agent.model(s)).item())
#             next_state, reward, done, _ = env.step(action)
#             state = preprocess_state(next_state)
#             if show_pacman:
#                 env.render("human")
#             total_episode_reward += reward
#             total_episode_steps += 1

#             time.sleep(delay)
#         total_reward += total_episode_reward
#         total_steps += total_episode_steps

#         if output:
#             print(f"Episode {ep}: total_reward={total_episode_reward:.2f} in {total_episode_steps} steps")
#         time.sleep(0.5)

#     avg_reward = total_reward / episodes
#     avg_steps = total_steps / episodes

#     if output:
#         print(f"=== Average Reward over {episodes} episodes: {avg_reward:.2f} ===")
#         print(f"=== Average Steps over {episodes} episodes: {avg_steps:.2f} ===")
#     env.close()

#     # return a sum of squares for avg_reward and avg_steps
#     return avg_reward / (avg_steps)


# def evaluate(
#     model_path, maze_spec, episodes=5, delay=0.25, fps=100, random_pacman_start=False, show_pacman=False, output=True
# ):
#     # Work on a local copy so we don't mutate caller/global specs
#     base_spec: MazeSpec = copy.deepcopy(maze_spec)

#     # Build first env from the (possibly non-randomized) spec
#     layout = tuple(generate_rect_layout(base_spec))
#     cfg = Config(maze_layout=layout, fps=fps)  # use layout, not maze_spec
#     env = PacmanEnv(cfg, human_mode=False, headless=not show_pacman)

#     sample_state = preprocess_state(env.reset())
#     agent = Agent(sample_state.shape, len(ACTIONS))
#     agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     agent.model.eval()

#     if output:
#         print(f"🎯 Evaluating {model_path} on {base_spec.width}x{base_spec.height} maze...")

#     total_reward = 0.0
#     total_steps = 0

#     for ep in range(episodes):
#         # For each episode, optionally randomize the pacman start ON THE SPEC WE'LL USE
#         if random_pacman_start:
#             start_x = random.randint(0, base_spec.width - 1)
#             start_y = random.randint(0, base_spec.height - 1)
#             start_x = 0
#             start_y = 1
#             if output:
#                 print("Randomizing Pacman start to:", (start_x, start_y))
#             ep_spec = replace(base_spec, pacman_start=(start_x, start_y))
#         else:
#             ep_spec = base_spec

#         # Rebuild layout/env for this episode from the correct spec
#         layout = tuple(generate_rect_layout(ep_spec))
#         print("\n".join(layout))
#         cfg = Config(maze_layout=layout, max_steps=200, fps=fps)
#         # Close previous env before replacing
#         env.close()
#         env = PacmanEnv(cfg, human_mode=False, headless=not show_pacman)

#         state = preprocess_state(env.reset())
#         done = False
#         total_episode_reward = 0.0
#         total_episode_steps = 0

#         while not done:
#             with torch.no_grad():
#                 s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
#                 action = int(torch.argmax(agent.model(s)).item())
#             next_state, reward, done, _ = env.step(action)
#             state = preprocess_state(next_state)
#             if show_pacman:
#                 env.render("human")
#             total_episode_reward += reward
#             total_episode_steps += 1
#             time.sleep(delay)

#         total_reward += total_episode_reward
#         total_steps += total_episode_steps

#         if output:
#             print(f"Episode {ep}: total_reward={total_episode_reward:.2f} in {total_episode_steps} steps")
#         time.sleep(0.5)

#     avg_reward = total_reward / episodes
#     avg_steps = total_steps / episodes

#     if output:
#         print(f"=== Average Reward over {episodes} episodes: {avg_reward:.2f} ===")
#         print(f"=== Average Steps over {episodes} episodes: {avg_steps:.2f} ===")

#     env.close()

#     # Keep your current metric; guard against zero division just in case
#     return avg_reward / (avg_steps if avg_steps != 0 else 1.0)


def evaluate(
    model_path: str,
    maze_spec: MazeSpec,
    episodes: int = 5,
    delay: float = 0.25,
    fps: int = 100,
    random_pacman_start: bool = False,
    randomize_pellets: bool = False,
    num_random_pellets: int = 1,
    show_pacman: bool = False,
    output: bool = True,
) -> float:
    """
    Evaluate a trained Pac-Man agent under various randomization modes.

    Modes:
      1. Spec-exact                      -> random_pacman_start=False, randomize_pellets=False
      2. Random Pac-Man start            -> random_pacman_start=True,  randomize_pellets=False
      3. Random Pac-Man + random pellets -> random_pacman_start=True,  randomize_pellets=True
         (num_random_pellets controls how many pellets are placed)
    """
    base_spec: MazeSpec = copy.deepcopy(maze_spec)

    # Build agent once
    initial_layout = tuple(generate_rect_layout(base_spec))
    env = PacmanEnv(Config(maze_layout=initial_layout, fps=fps), human_mode=False, headless=not show_pacman)
    sample_state = preprocess_state(env.reset())
    agent = Agent(sample_state.shape, len(ACTIONS))
    agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.model.eval()

    if output:
        print(f"🎯 Evaluating {model_path} on {base_spec.width}x{base_spec.height} maze...")

    total_reward, total_steps = 0.0, 0

    for ep in range(episodes):
        # --- Derive episode-specific spec ---
        ep_spec = copy.deepcopy(base_spec)

        # (1) Randomize Pac-Man start
        if random_pacman_start:
            start_x = random.randint(0, ep_spec.width - 1)
            start_y = random.randint(0, ep_spec.height - 1)
            ep_spec = replace(ep_spec, pacman_start=(start_x, start_y))
            if output:
                print(f"  🎲 Pac-Man start: {(start_x, start_y)}")

        # (2) Randomize pellets if requested
        if randomize_pellets:
            pellets = []
            while len(pellets) < num_random_pellets:
                pos = (
                    random.randint(0, ep_spec.width - 1),
                    random.randint(0, ep_spec.height - 1),
                )
                if pos != ep_spec.pacman_start and pos not in pellets:
                    pellets.append(pos)
            ep_spec = replace(ep_spec, pellet_mode="custom", pellet_positions=pellets)
            if output:
                print(f"  🍒 Randomized {len(pellets)} pellet(s): {pellets}")

        # --- Rebuild env for this episode ---
        layout = tuple(generate_rect_layout(ep_spec))
        cfg = Config(maze_layout=layout, max_steps=200, fps=fps)
        env.close()
        env = PacmanEnv(cfg, human_mode=False, headless=not show_pacman)

        # Render first frame right after reset (so Pac-Man is visible before moving)
        state = preprocess_state(env.reset())
        if show_pacman:
            env.render("human")

        done = False
        total_episode_reward, total_episode_steps = 0.0, 0

        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                action = int(torch.argmax(agent.model(s)).item())
            next_state, reward, done, _ = env.step(action)
            state = preprocess_state(next_state)
            if show_pacman:
                env.render("human")
            total_episode_reward += reward
            total_episode_steps += 1
            time.sleep(delay)

        total_reward += total_episode_reward
        total_steps += total_episode_steps

        if output:
            print(f"  ✅ Episode {ep}: reward={total_episode_reward:.2f} steps={total_episode_steps}")
        time.sleep(0.25)

    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes

    if output:
        print(f"=== Average Reward: {avg_reward:.2f} | Average Steps: {avg_steps:.2f} ===")

    env.close()

    # Simple efficiency metric (higher is better)
    return avg_reward / (avg_steps if avg_steps > 0 else 1.0)


if __name__ == "__main__":
    # Base directory containing all your stage runs
    run_dir = "runs"
    run_name = "stage2"

    stage_path = os.path.join(run_dir, run_name)
    model_path = os.path.join(stage_path, "final_model.pt")
    config_path = os.path.join(stage_path, "config.pkl")

    with open(config_path, "rb") as f:
        cfg = pickle.load(f)

    spec: MazeSpec = cfg["maze_spec"]
    # print("Loaded config:", cfg)
    print(f"🧠 Loaded {run_name} ({spec.width}x{spec.height}) with {spec.pellet_mode} mode")
    # layout = tuple(generate_rect_layout(spec))
    # cfg = Config(maze_layout=layout, max_steps=200)
    # env = PacmanEnv(cfg, human_mode=True, headless=False)

    # input("test")
    num_pellets = random.randint(1, (spec.width * spec.height) - 1)
    print("Number of random pellets for evaluation:", num_pellets)

    results = evaluate(
        model_path,
        spec,
        random_pacman_start=True,  # ✅ Random start each episode
        randomize_pellets=True,  # ✅ Random pellet layout each episode
        num_random_pellets=num_pellets,  # ✅ Vary pellet count
        show_pacman=True,
        episodes=10,
        delay=0,
        fps=1000,
    )
    print(f"Final evaluation score: {results:.2f}")

    input("run done")

    # # Optional: loop through all stages later
    # evaluated_dir = "evaluated_models"
    # os.makedirs(evaluated_dir, exist_ok=True)

    ############ NEW METHOD ############
    run_dir = "runs"

    # def _score_from_cfg(cfg: dict):
    #     # Prefer higher success_rate, then higher avg_reward, then lower std_reward
    #     sr  = cfg.get("success_rate")
    #     avg = cfg.get("avg_reward")
    #     std = cfg.get("std_reward")
    #     # Fallbacks ensure comparable tuples even if keys are missing
    #     return (
    #         sr if sr is not None else -1.0,
    #         avg if avg is not None else float("-inf"),
    #         -(std if std is not None else 0.0),
    #     )

    # best_by_size = {}  # size_key -> {"score": tuple, "run_name": str, "stage_path": str}

    for run_name in sorted(os.listdir(run_dir)):
        stage_path = os.path.join(run_dir, run_name)
        cfg_path = os.path.join(stage_path, "config.pkl")
        model_path = os.path.join(stage_path, "final_model.pt")

        # Check it config file exists
        if not os.path.isfile(cfg_path):
            print(f"⚠️  Skipping {run_name}, no config.pkl found.")
            continue

        print("stage_path:", stage_path)
        # Read the config file
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)

        print(cfg)
        spec: generate_rect_layout = cfg["maze_spec"]

        config_class = f"{spec.width}x{spec.height}"
        config_name = f"{run_name}_{config_class}_{spec.pellet_mode}"
        print(f"🧠 Loaded {config_name} in class {config_class}")

        score = evaluate(
            model_path,
            MazeSpec(
                width=spec.width,
                height=spec.height,
                include_ghosts=spec.include_ghosts,
                # ghost_positions=spec.ghost_positions,
                pellet_mode=spec.pellet_mode,
                pellet_positions=spec.pellet_positions,
                include_power_pellets=spec.include_power_pellets,
                surround_walls=spec.surround_walls,
            ),
            random_pacman_start=True,
            show_pacman=True,
            episodes=10,
            # delay=0,
            fps=1000,
            output=False,
        )

        print(f"Final evaluation score for {config_name}: {score:.4f}")
