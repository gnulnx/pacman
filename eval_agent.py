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

    # # 3) Show the actual layout used by pygame
    # print("=== Layout used by pygame ===")
    # for row in env.config.maze_layout:
    #     print(row)

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
    # ✅ use your trained stage2 model
    size = 8

    spec = MazeSpec(
        width=size,
        height=size,
        include_ghosts=False,
        pellet_mode="full",
        surround_walls=True,
    )
    # 28 works decently well
    evaluate(
        "runs/stage48/final_model.pt",
        spec,
        random_pacman_start=True,
        show_pacman=True,
        episodes=100,
        delay=0,
        fps=1000,
    )
