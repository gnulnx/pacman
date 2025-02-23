import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pygame
import math
import cv2  # OpenCV for image processing

# -----------------------------
# Configuration Parameters
# -----------------------------
MODE = "train"                # "train" or "play"
# Checkpoint file paths:
BEST_CHECKPOINT_PATH = "pacman_dqn_best.pth"     # Always stores the best model (with best_avg_reward and epsilon)
LATEST_CHECKPOINT_PATH = "pacman_dqn_latest.pth"   # Stores the latest model (can be overwritten)

HEADLESS = False              # For evaluation, you might want rendering.
DEBUG = False                 # Extra per-step debug rendering.
RENDER_EVERY = 10             # Render final frame every N episodes during training.
FIXED_MAZE = True             # Use a fixed maze layout for initial episodes.
NUM_EPISODES = 2000           # Total training episodes.
MAX_STEPS_PER_EPISODE = 200   # Maximum steps per episode.
TARGET_UPDATE_FREQ = 1000     # Frequency (in steps) to update target network.

# DQN and training hyperparameters:
INPUT_CHANNELS = 1
ACTION_DIM = 4                # 0 = up, 1 = down, 2 = left, 3 = right.
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_CAPACITY = 10000
EPSILON_START = 0.1
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Pygame screen and game settings:
from settings import ROWS, COLS, TILE_SIZE, BLACK
# For our test, override ROWS and COLS:
ROWS = 12
COLS = 12

# For reproducibility:
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# For headless mode, set the SDL video driver if desired.
if MODE == "train" and HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# -----------------------------
# Pre-generate a fixed maze layout if required.
# -----------------------------
from maze import generate_maze
FIXED_MAZE_LAYOUT = generate_maze(ROWS, COLS)

# -----------------------------
# Import game modules.
# -----------------------------
from pacman import PacMan
from ghost import Ghost  # Future use.
from maze import Maze, get_open_cells

# -----------------------------
# Pacman Environment
# -----------------------------
class PacmanEnv:
    def __init__(self, fixed_maze=False):
        self.fixed_maze = fixed_maze
        self.screen_width = COLS * TILE_SIZE
        self.screen_height = ROWS * TILE_SIZE
        pygame.display.set_caption("Pac-Man RL")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.reset()

    def reset(self):
        if self.fixed_maze:
            maze_layout = FIXED_MAZE_LAYOUT
        else:
            maze_layout = generate_maze(12, 12)
        self.maze_obj = Maze(maze_layout)
        open_cells = get_open_cells(maze_layout)
        if not open_cells:
            raise Exception("No open cells in maze!")
        self.ghosts = []  # Not used currently.
        pac_cell = random.choice(open_cells)
        pac_x = pac_cell[1] * TILE_SIZE + TILE_SIZE // 2
        pac_y = pac_cell[0] * TILE_SIZE + TILE_SIZE // 2
        self.pacman = PacMan(pac_x, pac_y)
        self.done = False
        directions = [
            pygame.math.Vector2(0, -1),
            pygame.math.Vector2(0, 1),
            pygame.math.Vector2(-1, 0),
            pygame.math.Vector2(1, 0)
        ]
        self.last_direction = random.choice(directions)
        self.old_tile = (int(pac_y // TILE_SIZE), int(pac_x // TILE_SIZE))
        return self.get_state()

    def distance_to_nearest_pellet(self):
        if not self.maze_obj.pellets:
            return 0.0
        px, py = self.pacman.x, self.pacman.y
        min_dist = float('inf')
        for (r, c) in self.maze_obj.pellets:
            cx = c * TILE_SIZE + TILE_SIZE // 2
            ry = r * TILE_SIZE + TILE_SIZE // 2
            dist = math.hypot(px - cx, py - ry)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def step(self, action):
        pygame.event.pump()
        dist_before = self.distance_to_nearest_pellet()
        if action == 0:
            self.last_direction = pygame.math.Vector2(0, -1)
        elif action == 1:
            self.last_direction = pygame.math.Vector2(0, 1)
        elif action == 2:
            self.last_direction = pygame.math.Vector2(-1, 0)
        elif action == 3:
            self.last_direction = pygame.math.Vector2(1, 0)
        self.pacman.intended_direction = self.last_direction

        reward = 0.0


        # Pellet and Fruit Rewards
        pre_pellet = len(self.maze_obj.pellets)
        pre_fruits = len(self.maze_obj.fruits)

        self.pacman.update(self.maze_obj)

        # self.pacman.check_for_collection(self.maze_obj)

        post_pellot = len(self.maze_obj.pellets)
        fruit_post = len(self.maze_obj.fruits)

        pellets_collected = pre_pellet - post_pellot
        reward += pellets_collected * 5

        fruits_collected = pre_fruits - fruit_post
        reward += fruits_collected * 10

        # This is useful for small boards or if there is enough time to clear the board
        if post_pellot == 0:
            reward += 50
            self.done = True

        # print(pre_pellet, post_pellot, pellets_collected, reward)
        # input()
        
        # new_tile = (int(self.pacman.y // TILE_SIZE), int(self.pacman.x // TILE_SIZE))
        # if new_tile == self.old_tile:
        #     reward -= 0.2
        # self.old_tile = new_tile
        # dist_after = self.distance_to_nearest_pellet()

        # delta = dist_before - dist_after
        # if delta > 0:
        #     reward += 0.5
        # elif delta < 0:
        #     reward -= 0.5

        if DEBUG and not HEADLESS:
            self.render()
            pygame.time.delay(30)
        else:
            self.draw_offscreen()
        next_state = self.get_state()
        return next_state, reward, self.done, {}

    def draw_offscreen(self):
        self.screen.fill(BLACK)
        self.maze_obj.draw(self.screen)
        self.pacman.draw(self.screen)

    def render(self):
        self.screen.fill(BLACK)
        self.maze_obj.draw(self.screen)
        self.pacman.draw(self.screen)
        pygame.display.flip()

    def get_state(self):
        if HEADLESS:
            self.draw_offscreen()
        else:
            self.render()
        image = pygame.surfarray.array3d(self.screen)
        image = np.transpose(image, (1, 0, 2))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (84, 84))
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def close(self):
        pygame.quit()

# -----------------------------
# DQN Model
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Initialize DQN, Replay Buffer, Optimizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(INPUT_CHANNELS, ACTION_DIM).to(device)
target_net = DQN(INPUT_CHANNELS, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

# Attempt to load checkpoint: Prefer best checkpoint.
epsilon = EPSILON_START  # Default starting epsilon
# Also maintain best_avg_reward across runs:
best_avg_reward = float('-inf')
if os.path.exists(BEST_CHECKPOINT_PATH):
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    best_avg_reward = checkpoint.get("best_avg_reward", float('-inf'))
    print(f"🏆 Resumed from best checkpoint with epsilon {epsilon:.3f} and best_avg_reward {best_avg_reward:.2f}.")
elif os.path.exists(LATEST_CHECKPOINT_PATH):
    checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    print(f"📌 Resumed from latest checkpoint with epsilon {epsilon:.3f}.")
else:
    print("🚨 No checkpoint found. Starting from scratch.")

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(ACTION_DIM)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()

def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------------
# Training Loop
# -----------------------------
def train_dqn():
    global epsilon, best_avg_reward
    env = PacmanEnv(fixed_maze=FIXED_MAZE)
    total_steps = 0
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            train_step()
            total_steps += 1

            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} => Reward: {episode_reward:.2f}, Steps: {steps}, Epsilon: {epsilon:.3f}")

        if episode % RENDER_EVERY == 0 and episode > 0:
            env.render()
            pygame.time.delay(1000)

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode > 0 and episode % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10.0
            print(f"Average reward over last 10 episodes: {avg_reward:.2f}")
            # Save latest checkpoint (always update latest)
            latest_checkpoint = {"model_state": policy_net.state_dict(), "epsilon": epsilon}
            torch.save(latest_checkpoint, LATEST_CHECKPOINT_PATH)
            print(f"📌 Latest checkpoint saved at episode {episode}")
            # Save best model only if current avg_reward exceeds best_avg_reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_checkpoint = {"model_state": policy_net.state_dict(), "epsilon": epsilon, "best_avg_reward": best_avg_reward}
                torch.save(best_checkpoint, BEST_CHECKPOINT_PATH)
                print(f"🏆 New best model saved at episode {episode} with average reward {avg_reward:.2f}")

    env.close()

# -----------------------------
# Evaluation / Play Mode
# -----------------------------
def play_dqn(num_episodes=10):
    # In play mode, we want epsilon to be 0 (fully greedy).
    if os.path.exists(BEST_CHECKPOINT_PATH):
        checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
        policy_net.load_state_dict(checkpoint["model_state"])
        policy_net.eval()
        print("🏆 Loaded trained model from best checkpoint for play mode.")
    else:
        print("🚨 No checkpoint found. Exiting play mode.")
        return

    # Override epsilon to 0 in play mode
    play_epsilon = 0.0

    env = PacmanEnv(fixed_maze=False)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = select_action(state, play_epsilon)  # Always greedy
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            pygame.time.delay(100)
            steps += 1
        print(f"Evaluation Episode {episode} finished in {steps} steps with reward {episode_reward:.2f}.")
    env.close()

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    pygame.init()
    if MODE == "train":
        train_dqn()
    elif MODE == "play":
        play_dqn(num_episodes=5)
    pygame.time.delay(3000)
    pygame.quit()
