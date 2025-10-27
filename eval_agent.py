import os
import pickle
import random
import time

import torch

from dojo_agent import Agent
from dojo_train import preprocess_state
from pacman_env import ACTIONS, Config, MazeSpec, PacmanEnv, generate_rect_layout

# Exact runs each time
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)


def evaluate(model_path, maze_spec, episodes=5, delay=0.25, fps=100, random_pacman_start=False, show_pacman=False):
    # 1) Freeze the layout explicitly from the spec
    layout = tuple(generate_rect_layout(maze_spec))
    cfg = Config(maze_layout=layout)  # <- NOT maze_spec
    env = PacmanEnv(cfg, human_mode=False, headless=False)
    sample_state = preprocess_state(env.reset())
    agent = Agent(sample_state.shape, len(ACTIONS))
    agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.model.eval()

    print(f"🎯 Evaluating {model_path} on {maze_spec.width}x{maze_spec.height} maze...")
    total_reward = 0.0
    total_steps = 0
    for ep in range(episodes):
        if random_pacman_start:
            start_x = random.randint(0, spec.width - 1)
            start_y = random.randint(0, spec.height - 1)
            spec.pacman_start = (start_x, start_y)

            # Reinit layout and env with new pacman start
            layout = tuple(generate_rect_layout(maze_spec))
            cfg = Config(maze_layout=layout, max_steps=200, fps=fps)
            env = PacmanEnv(cfg, human_mode=False, headless=False)

            env.reset()

        state = preprocess_state(env.reset())
        done = False
        total_episode_reward = 0.0
        total_episode_steps = 0
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
        print(f"Episode {ep}: total_reward={total_episode_reward:.2f} in {total_episode_steps} steps")
        time.sleep(0.5)

    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes
    print(f"=== Average Reward over {episodes} episodes: {avg_reward:.2f} ===")
    print(f"=== Average Steps over {episodes} episodes: {avg_steps:.2f} ===")
    env.close()


if __name__ == "__main__":
    # Base directory containing all your stage runs
    run_dir = "runs"
    run_name = "stage17"

    # Derive model/config paths
    stage_path = os.path.join(run_dir, run_name)
    model_path = os.path.join(stage_path, "final_model.pt")
    config_path = os.path.join(stage_path, "config.pkl")

    # Load stage configuration
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)

    # This is using the same mazespec that was used for training
    spec: MazeSpec = cfg["maze_spec"]
    print(f"🧠 Loaded {run_name} ({spec.width}x{spec.height}) with {spec.pellet_mode} mode")

    # Evaluate the trained model
    evaluate(
        model_path,
        spec,
        random_pacman_start=True,
        show_pacman=True,
        episodes=10,
        delay=0,
        fps=1000,
    )

    # Optional: loop through all stages later
    """
    for run_name in sorted(os.listdir(run_dir)):
        stage_path = os.path.join(run_dir, run_name)
        if not os.path.isdir(stage_path):
            continue
        model_path = os.path.join(stage_path, "final_model.pt")
        config_path = os.path.join(stage_path, "config.pkl")
        if not (os.path.exists(model_path) and os.path.exists(config_path)):
            continue

        with open(config_path, "rb") as f:
            cfg = pickle.load(f)
        spec: MazeSpec = cfg["maze_spec"]
        print(f"🎯 Evaluating {run_name}")
        evaluate(model_path, spec, random_pacman_start=True, show_pacman=False, episodes=5, fps=200)
    """
