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
from jprint import jprint  # noqa

from dojo_agent import Agent  # noqa
from dojo_train_impl import preprocess_state  # noqa
from failed_run_recorder import FailedRunRecorder  # noqa
from pacman_env import (  # noqa
    ACTIONS,
    Config,
    CuriousPacmanEnv,
    MazeSpec,
    PacmanEnv,
    generate_rect_layout,
)


def evaluate_curiosity(
    model_path: str,
    maze_spec: MazeSpec,
    episodes: int = 5,
    delay: float = 0.0,
    fps: int = 200,
    show_pacman: bool = False,
    output: bool = True,
    device: str = "cpu",
    random_pacman_start=True,
) -> float:
    """
    Evaluate a curiosity-pretrained model.

    Score = average fraction of unique tiles visited per episode.
    1.0 means the agent explored every reachable tile at least once.
    """

    print("show_pacman:", show_pacman)
    print("maze_spec:", maze_spec)

    env = CuriousPacmanEnv(
        Config(maze_spec=maze_spec, fps=fps, max_steps=1024),
        human_mode=False,
        headless=False,
    )

    if show_pacman:
        env.render("human")

    sample_state = preprocess_state(env.reset())
    agent = Agent(sample_state.shape, len(ACTIONS), device=device)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()

    if output:
        print(f"🧭 Evaluating curiosity model on {maze_spec.width}x{maze_spec.height} maze...")

    total_fraction, total_reward = 0.0, 0.0

    for ep in range(episodes):
        state = preprocess_state(env.reset())
        done = False
        ep_reward = 0.0

        ep_spec = copy.deepcopy(maze_spec)

        # Randomize starts / pellets
        if random_pacman_start:
            ep_spec = replace(
                ep_spec, pacman_start=(random.randint(0, ep_spec.width - 1), random.randint(0, ep_spec.height - 1))
            )

        layout = tuple(generate_rect_layout(ep_spec))
        # --- Convert to mutable numeric grid ---
        # layout_np = np.array([[1 if c == "#" else 0 for c in row] for row in layout], dtype=np.int8)

        # # --- Add one random interior wall ---
        # x, y = random.randint(1, layout_np.shape[1] - 2), random.randint(1, layout_np.shape[0] - 2)
        # if (x, y) != ep_spec.pacman_start:
        #     layout_np[y, x] = 1  # mark as wall

        # # --- Convert back to string rows ---
        # layout = tuple("".join("#" if v == 1 else "." for v in row) for row in layout_np)

        cfg = Config(maze_layout=layout, max_steps=200, fps=fps)
        env.close()
        env = CuriousPacmanEnv(cfg, human_mode=False, headless=False)
        state = preprocess_state(env.reset())
        if show_pacman:
            env.render("human")

        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                action = int(torch.argmax(agent.model(s)).item())
                # action = random.choice(list(ACTIONS.keys()))
                print("Action chosen:", action)
                # input()
            next_state, reward, done, info = env.step(action)
            state = preprocess_state(next_state)
            ep_reward += reward
            if show_pacman:
                env.render("human")
            if delay:
                time.sleep(delay)

        visited_fraction = info.get("visited_fraction", 0.0)
        total_fraction += visited_fraction
        total_reward += ep_reward

        if output:
            print(f"  ✅ Ep{ep}: visited={visited_fraction:.2f}, reward={ep_reward:.2f}")

    env.close()
    avg_fraction = total_fraction / episodes
    avg_reward = total_reward / episodes

    if output:
        print(f"🌍 Avg visited fraction={avg_fraction:.3f}, Avg reward={avg_reward:.3f}")

    return avg_fraction


def evaluate_curiosity_same_size(model_path, maze_spec, episodes=10, device="cpu"):
    return evaluate_curiosity(
        model_path,
        maze_spec,
        episodes=episodes,
        device=device,
        show_pacman=True,
        output=True,
    )


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
    device: str = "cpu",
    normalize_by_size: bool = True,
) -> float:
    """
    Evaluate a trained Pac-Man agent under various randomization modes.

    Parameters
    ----------
    normalize_by_size : bool
        If True, the final efficiency score is normalized by maze area and
        completion percentage so models of different sizes can be compared fairly.
    """
    base_spec: MazeSpec = copy.deepcopy(maze_spec)

    # --- Build agent and env once ---
    initial_layout = tuple(generate_rect_layout(base_spec))
    env = PacmanEnv(Config(maze_layout=initial_layout, fps=fps), human_mode=False, headless=not show_pacman)
    sample_state = preprocess_state(env.reset())

    agent = Agent(sample_state.shape, len(ACTIONS), device=device)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()

    total_reward, total_steps, total_pellets_left = 0.0, 0, 0
    total_pellets_start = 0.0
    total_pellets = env.maze.pellets.sum()

    if output:
        print(f"🎯 Evaluating {model_path} on {base_spec.width}x{base_spec.height} maze...")

    recorder = FailedRunRecorder(run_name=os.path.basename(model_path))

    for ep in range(episodes):
        ep_spec = copy.deepcopy(base_spec)

        # Randomize starts / pellets
        if random_pacman_start:
            ep_spec = replace(
                ep_spec, pacman_start=(random.randint(0, ep_spec.width - 1), random.randint(0, ep_spec.height - 1))
            )
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

        layout = tuple(generate_rect_layout(ep_spec))
        cfg = Config(maze_layout=layout, max_steps=200, fps=fps)
        env.close()
        env = PacmanEnv(cfg, human_mode=False, headless=not show_pacman)
        state = preprocess_state(env.reset())
        state = preprocess_state(env.reset())
        start_pellets = float(env.maze.pellets.sum())  # <-- add this
        total_pellets_start += start_pellets  # <-- and this
        if show_pacman:
            env.render("human")

        done = False
        ep_reward, ep_steps = 0.0, 0
        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                action = int(torch.argmax(agent.model(s)).item())
            next_state, reward, done, _ = env.step(action)
            state = preprocess_state(next_state)
            if show_pacman:
                env.render("human")
            ep_reward += reward
            ep_steps += 1
            if delay:
                time.sleep(delay)

        pellets_left = env.maze.pellets.sum()
        total_reward += ep_reward
        total_steps += ep_steps
        total_pellets_left += pellets_left

        # Save failed layouts
        if pellets_left > 0:
            recorder.record_failure(
                env=env,
                episode=ep,
                maze_spec=ep_spec,
                final_state=env._get_state(),
                reward=ep_reward,
                steps=ep_steps,
                pellets_remaining=int(pellets_left),
                failure_reason="pellets_remaining",
            )
            recorder.save(verbose=False)

        if output:
            print(f"  ✅ Ep{ep}: reward={ep_reward:.1f} steps={ep_steps} pellets_left={pellets_left}")

    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes
    avg_pellets_left = total_pellets_left / episodes
    avg_pellets_start = total_pellets_start / episodes
    return max(0.0, 1.0 - (avg_pellets_left / max(avg_pellets_start, 1.0)))


def evaluate_full_model_random_pacman_start_same_size_map(
    model_path: str,
    maze_spec: MazeSpec,
    episodes: int = 5,
    delay: float = 0.25,
    fps: int = 100,
    show_pacman: bool = False,
    output: bool = True,
    device: str = "cpu",
):

    return evaluate(
        model_path,
        MazeSpec(
            width=maze_spec.width,
            height=maze_spec.height,
            pellet_mode="full",
            include_ghosts=False,
            include_power_pellets=False,
            surround_walls=True,
        ),
        random_pacman_start=True,  # ✅ Random start each episode
        show_pacman=show_pacman,
        episodes=episodes,
        output=output,
        delay=delay,
        fps=fps,
        device=device,
    )


def evaluate_random_start_same_size_map(
    model_path: str,
    maze_spec: MazeSpec,
    episodes: int = 5,
    delay: float = 0.25,
    fps: int = 100,
    show_pacman: bool = False,
    output: bool = True,
    num_pellets: int = None,
    device: str = "cpu",
):
    if num_pellets is None:
        num_pellets = random.randint(1, (maze_spec.width * maze_spec.height) - 1)

    return evaluate(
        model_path,
        MazeSpec(
            width=maze_spec.width,
            height=maze_spec.height,
            pellet_mode="custom",
            include_ghosts=False,
            include_power_pellets=False,
            surround_walls=True,
        ),
        random_pacman_start=True,  # ✅ Random start each episode
        randomize_pellets=True,  # ✅ Random pellet layout each episode
        num_random_pellets=num_pellets,  # ✅ Vary pellet count
        show_pacman=show_pacman,
        episodes=episodes,
        output=output,
        delay=delay,
        fps=fps,
        device=device,
    )


def evaluate_cross_size(
    model_path: str,
    maze_spec: MazeSpec,
    target_sizes: tuple[tuple[int, int], ...] = ((4, 4), (8, 8), (12, 12)),
    episodes_per_size: int = 5,
    delay: float = 0.0,
    fps: int = 200,
    show_pacman: bool = False,
    output: bool = True,
    device: str = "cpu",
    num_pellets: int = None,
) -> dict[int, float]:
    """
    Evaluate a trained model on *different maze sizes* to measure cross-scale generalization.

    This evaluation mode asks:
        "Can a policy trained on one environment size (e.g. 8x8)
         adapt its strategy to smaller or larger grids (e.g. 4x4, 12x12)?"

    For each target size, this function:
      1. Builds a new MazeSpec of the requested width/height
      2. Uses `pellet_mode="custom"` with randomized pellet layouts
      3. Enables random Pac-Man start position
      4. Runs a small evaluation batch via `evaluate(...)`
      5. Records the model's normalized efficiency score for that size

    Returns
    -------
    dict[int, float]
        A mapping of maze size → performance score, where higher scores
        indicate better transfer/generalization of the trained policy.

    Example
    -------
    >>> results = evaluate_cross_size("runs/stage56/final_model.pt", spec, target_sizes=(4, 8, 12))
    >>> print(results)
    {4: 0.58, 8: 0.61, 12: 0.47}
    """

    scores: dict[int, float] = {}

    if output:
        print("🔍 Cross-size generalization evaluation")
        print("----------------------------------------")

    for width, height in target_sizes:
        # Dynamically adjust maze spec for each test size
        test_spec = MazeSpec(
            width=width,
            height=height,
            include_ghosts=False,
            include_power_pellets=False,
            surround_walls=True,
            pellet_mode="custom",
        )

        max_pellets = (test_spec.width * test_spec.height) - 1

        if num_pellets is None or num_pellets > max_pellets:
            num_pellets = random.randint(1, max_pellets)

        if output:
            print(f"⚙️  Testing {test_spec.width}x{test_spec.height} map with {num_pellets} random pellets...")

        score = evaluate(
            model_path,
            test_spec,
            random_pacman_start=True,
            randomize_pellets=True,
            num_random_pellets=num_pellets,
            show_pacman=show_pacman,
            episodes=episodes_per_size,
            delay=delay,
            fps=fps,
            output=False,
            device=device,
        )

        scores[f"{width}x{height}"] = score

        if output:
            print(f"  ✅ {width}x{height} → Score: {score:.4f}")

    if output:
        print("----------------------------------------")
        print(f"🏁 Cross-size results for : {scores}")

    return scores


if __name__ == "__main__":
    # Base directory containing all your stage runs
    run_dir = "runs/failure_mix/"
    run_name = "best_overall"

    # density/cluster_tbarl_1/density_0.02/last_model.pt

    # runs/stage26/final_model.pt
    run_dir = "density/cluster_tbarl_1/"
    # run_dir = "density/random_tbarl_0/"
    run_name = "density_0.02"
    # "saved_models/0.7352_stage3_best_model.pt"

    stage_path = os.path.join(run_dir, run_name)
    model_path = os.path.join(stage_path, "last_model.pt")

    # model_path = "saved_models/0.7352_stage3_best_model.pt"
    config_path = os.path.join(stage_path, "config.pkl")

    with open(config_path, "rb") as f:
        cfg = pickle.load(f)

    spec: MazeSpec = cfg["maze_spec"]
    print(f"🧠 Loaded {run_name} ({spec.width}x{spec.height}) with {spec.pellet_mode} mode")

    num_pellets = random.randint(1, (spec.width * spec.height) - 1)
    print("Number of random pellets for evaluation:", num_pellets)

    episodes = 25
    results = 0
    num_pellets = 2

    target_sizes = [(8, 8)]

    results = evaluate_cross_size(
        model_path,
        spec,
        target_sizes=target_sizes,
        episodes_per_size=episodes,
        show_pacman=True,
        output=True,
        delay=0.025,
        fps=1000,
        num_pellets=num_pellets,
    )
    final_score = sum(results.values()) / len(results)
    # final_score = (results_1 + results_2 + sum(results_3.values())) / (2 + len(results_3))
    print(f"Results for pellets={num_pellets}(cross size):")
    jprint(results)
    print(f"Final evaluation score: {final_score:.2f}")
