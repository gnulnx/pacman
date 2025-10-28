import copy  # noqa
import math  # noqa
import os  # noqa
import pickle  # noqa
import random  # noqa
import shutil  # noqa
import sys  # noqa
import time  # noqa
from dataclasses import replace  # noqa

import numpy as np  # noqa
import torch  # noqa

from dojo_agent import Agent  # noqa
from dojo_train import preprocess_state  # noqa
from failed_run_recorder import FailedRunRecorder  # noqa
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


# def visualize_failed_record(env, record, pause=True):
#     """
#     Visually confirm that a saved failure record can be reconstructed.
#     This assumes pygame is still active and 'env' is a PacmanEnv instance.
#     """
#     state = record["final_state"]

#     # --- Restore maze pellets ---
#     env.maze.pellets = np.array(state["pellets"], dtype=np.uint8)

#     # --- Restore Pac-Man ---
#     env.pacman.position = tuple(state["pacman"])

#     # --- Restore ghosts (if any) ---
#     for g, pos in zip(env.ghosts, state["ghosts"]):
#         g.position = tuple(pos)

#     # --- Re-render the restored state ---
#     env.render("human")
#     print(f"🔍 Visualizing failed episode {record['episode']}")
#     print(f"  Pellets remaining: {record['pellets_remaining']}")
#     print(f"  Reward: {record['reward']:.2f} | Steps: {record['steps']}")
#     print(f"  Failure reason: {record['failure_reason']}")

#     if pause:
#         input("🟡 Press Enter to continue...")


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

    print("pellet count: ", env.maze.pellets.sum())

    if output:
        print(f"🎯 Evaluating {model_path} on {base_spec.width}x{base_spec.height} maze...")

    total_reward, total_steps, total_pellets_left = 0.0, 0, 0

    run_name = os.path.basename(model_path)
    recorder = FailedRunRecorder(run_name=run_name)

    for ep in range(episodes):
        # --- Derive episode-specific spec ---
        ep_spec = copy.deepcopy(base_spec)

        # (1) Randomize Pac-Man start
        if random_pacman_start:
            start_x = random.randint(0, ep_spec.width - 1)
            start_y = random.randint(0, ep_spec.height - 1)
            ep_spec = replace(ep_spec, pacman_start=(start_x, start_y))

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
            # print(f"reward={reward} done={done}")
            state = preprocess_state(next_state)
            if show_pacman:
                env.render("human")
            total_episode_reward += reward
            total_episode_steps += 1
            time.sleep(delay)

        total_reward += total_episode_reward
        total_steps += total_episode_steps
        total_pellets_left += env.maze.pellets.sum()

        if env.maze.pellets.sum() != 0:
            # At this point we need to write/save the final layout to retrain on
            recorder.record_failure(
                env=env,
                episode=ep,
                maze_spec=ep_spec,
                final_state=env._get_state(),
                reward=total_episode_reward,
                steps=total_episode_steps,
                pellets_remaining=int(env.maze.pellets.sum()),
                failure_reason="pellets_remaining",
            )
            # visualize_failed_record(env, last_record)
            recorder.save()
            input("checkpoint (press Enter to continue)")

        if output:
            print(
                f"  ✅ Episode {ep}: reward={total_episode_reward:.2f} steps={total_episode_steps} pellets_left={env.maze.pellets.sum()}"
            )
        time.sleep(0.25)

    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes
    avg_pellets_left = total_pellets_left / episodes

    if output:
        print(
            f"=== Average Reward: {avg_reward:.2f} | Average Steps: {avg_steps:.2f} | Average Pellets Left: {avg_pellets_left:.2f} ==="
        )

    env.close()

    # Simple efficiency metric (higher is better)
    return avg_reward / (avg_steps if avg_steps > 0 else 1.0)


if __name__ == "__main__":
    # Base directory containing all your stage runs
    run_dir = "runs"
    run_name = "stage56"

    stage_path = os.path.join(run_dir, run_name)
    model_path = os.path.join(stage_path, "final_model.pt")
    config_path = os.path.join(stage_path, "config.pkl")

    with open(config_path, "rb") as f:
        cfg = pickle.load(f)

    spec: MazeSpec = cfg["maze_spec"]
    print(f"🧠 Loaded {run_name} ({spec.width}x{spec.height}) with {spec.pellet_mode} mode")

    num_pellets = random.randint(1, (spec.width * spec.height) - 1)
    print("Number of random pellets for evaluation:", num_pellets)

    # TODO you want to create seperate methods for ecah evaluation approach. Just set the MazeSpec manually along wtih evaluate(...)
    # you only need the actual configured MazeSpec from the pickle to get width/height right.  And even then you might want to test an
    # 8x8 on a 4x4 just to see how well it generalizes.
    # For now just hardcode a few things to get the eval working.
    # num_pellets = 15
    # spec.pellet_density = 1
    # spec.pellet_mode = "custom"  # Override to full pellets for eval
    # spec.width = 4
    # spec.height = 4
    spec.pacman_start = (0, 0)
    results = evaluate(
        model_path,
        spec,
        random_pacman_start=True,  # ✅ Random start each episode
        randomize_pellets=True,  # ✅ Random pellet layout each episode
        num_random_pellets=num_pellets,  # ✅ Vary pellet count
        show_pacman=True,
        episodes=1000,
        output=True,
        delay=0,
        fps=1000,
    )
    print(f"Final evaluation score: {results:.2f}")

    sys.exit(0)

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
    print("Evaluating all runs in:", run_dir)
    print("-----------------------------------")
    for run_name in sorted(os.listdir(run_dir)):
        stage_path = os.path.join(run_dir, run_name)
        cfg_path = os.path.join(stage_path, "config.pkl")
        model_path = os.path.join(stage_path, "final_model.pt")

        # Check it config file exists
        if not os.path.isfile(cfg_path):
            print(f"⚠️  Skipping {run_name}, no config.pkl found.")
            continue

        # print("stage_path:", stage_path)
        # Read the config file
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)

        # print(cfg)
        spec: generate_rect_layout = cfg["maze_spec"]

        config_class = f"{spec.width}x{spec.height}"
        config_name = f"{run_name}_{config_class}_{spec.pellet_mode}"
        # print(f"🧠 Loaded {config_name} in class {config_class}")
        # print("MazeSpec:", spec)
        spec = MazeSpec(
            width=spec.width,
            height=spec.height,
            include_ghosts=spec.include_ghosts,
            # ghost_positions=spec.ghost_positions,
            pellet_positions=spec.pellet_positions,
            pellet_mode=spec.pellet_mode,
            include_power_pellets=False,
            surround_walls=True,
        )
        # print("MazeSpec:", spec)
        # input()
        score = evaluate(
            model_path,
            spec,
            random_pacman_start=True,
            show_pacman=False,
            episodes=10,
            delay=0,
            fps=1000,
            output=False,
        )

        print(f"Final evaluation score for {config_name}: {score:.4f}")
