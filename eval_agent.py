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


# def evaluate(model_path, maze_spec, episodes=5, delay=0.25):
#     env = PacmanEnv(Config(maze_spec=maze_spec), human_mode=False, headless=False)

#     # initialize agent + model
#     sample_state = preprocess_state(env.reset())
#     agent = Agent(sample_state.shape, len(ACTIONS))

#     # ✅ Load model weights
#     agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     agent.model.eval()

#     # ✅ Make sure we're on the same device (MPS if available)
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#         print("Using MPS")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("Using CUDA")
#     else:
#         device = torch.device("cpu")
#         print("Using CPU")

#     agent.model.to(device)

#     print(f"🎯 Evaluating {model_path} on {maze_spec.width}x{maze_spec.height} maze...")
#     for ep in range(episodes):
#         state = preprocess_state(env.reset())
#         done = False
#         total_reward = 0
#         while not done:
#             with torch.no_grad():
#                 s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # 👈 ensure same device
#                 q = agent.model(s)
#                 action = int(torch.argmax(q).item())
#             next_state, reward, done, _ = env.step(action)
#             state = preprocess_state(next_state)
#             env.render("human")
#             total_reward += reward
#             time.sleep(delay)
#         print(f"Episode {ep}: total_reward={total_reward:.2f}")
#         time.sleep(1)
#     env.close()


def evaluate(model_path, maze_spec, episodes=5, delay=0.25, randmon_pacman_start=False):
    # 1) Freeze the layout explicitly from the spec
    layout = tuple(generate_rect_layout(maze_spec))

    # 2) Build env from the fixed layout (bypass auto-gen)
    cfg = Config(maze_layout=layout)  # <- NOT maze_spec
    env = PacmanEnv(cfg, human_mode=False, headless=False)

    # 3) Show the actual layout used by pygame
    print("=== Layout used by pygame ===")
    for row in env.config.maze_layout:
        print(row)

    # 4) Set up agent on same device
    sample_state = preprocess_state(env.reset())
    agent = Agent(sample_state.shape, len(ACTIONS))
    agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    agent.model.to(device)

    print(f"🎯 Evaluating {model_path} on {maze_spec.width}x{maze_spec.height} maze...")
    for ep in range(episodes):
        if randmon_pacman_start:
            start_x = random.randint(0, spec.width - 1)
            start_y = random.randint(0, spec.height - 1)
            spec.pacman_start = (start_x, start_y)

            # Reinit layout and env with new pacman start
            layout = tuple(generate_rect_layout(maze_spec))
            cfg = Config(maze_layout=layout)  # <- NOT maze_spec
            env = PacmanEnv(cfg, human_mode=False, headless=False)

            env.reset()

        state = preprocess_state(env.reset())
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(agent.model(s)).item())
            next_state, reward, done, _ = env.step(action)
            state = preprocess_state(next_state)
            env.render("human")
            total_reward += reward
            time.sleep(delay)
        print(f"Episode {ep}: total_reward={total_reward:.2f}")
        time.sleep(0.5)
    env.close()


if __name__ == "__main__":
    # ✅ use your trained stage2 model
    size = 4

    # spec = MazeSpec(
    #     width=4,
    #     height=4,
    #     include_ghosts=False,
    #     pellet_mode="full",
    #     pacman_start=(2, 1),
    #     pellet_positions=[(0, 0)],
    #     include_power_pellets=False,
    #     surround_walls=True,
    # )

    start_x = random.randint(0, size - 1)
    start_y = random.randint(0, size - 1)

    spec = MazeSpec(
        width=size,
        height=size,
        pacman_start=(start_x, start_y),
        include_ghosts=False,
        pellet_mode="full",
        # pellet_positions=[(0, 0)],
        surround_walls=True,
    )
    evaluate("runs/stage38/final_model.pt", spec, randmon_pacman_start=True)
