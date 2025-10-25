import time

import torch

from dojo_agent import Agent
from dojo_train import preprocess_state
from pacman_env import ACTIONS, Config, MazeSpec, PacmanEnv


def watch(stage_name="stage1"):
    # same simple 2×2 layout
    spec = MazeSpec(
        width=2,
        height=2,
        include_ghosts=False,
        pellet_mode="single",
        pellet_positions=[(1, 1)],
        include_power_pellets=False,
        surround_walls=True,
    )
    env = PacmanEnv(Config(maze_spec=spec), human_mode=False, headless=False)

    sample_state = preprocess_state(env.reset())
    agent = Agent(sample_state.shape, len(ACTIONS))
    agent.model.load_state_dict(torch.load(f"runs/{stage_name}/model.pt", map_location="cpu"))
    agent.model.eval()

    print("🎮 Watching trained agent...")
    for ep in range(5):
        state = preprocess_state(env.reset())
        done = False
        total_reward = 0
        while not done:
            # greedy (no epsilon)
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q = agent.model(s)
                action = int(torch.argmax(q).item())
            next_state, reward, done, _ = env.step(action)
            state = preprocess_state(next_state)
            env.render("human")  # 🪟 opens pygame window
            total_reward += reward
            time.sleep(0.25)
        print(f"Episode {ep}: total_reward={total_reward:.2f}")
        time.sleep(1)
    env.close()


if __name__ == "__main__":
    watch("stage1")
