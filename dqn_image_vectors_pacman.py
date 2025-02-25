
from pacman import PacMan
from ghost import Ghost  # Future use.
from maze import Maze, get_open_cells, safe_spawn_pacman
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
import csv
from settings import ROWS, COLS, TILE_SIZE, BLACK
from maze import generate_maze


csvfile = open("training_log.csv", "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["episode", "step", "total_frames", "loss", "avg_q", "epsilon", "episode_reward"])

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
NUM_EPISODES = 10000           # Total training episodes.
MAX_STEPS_PER_EPISODE = 2500   # Maximum steps per episode.
TARGET_UPDATE_FREQ = 1000     # Frequency (in steps) to update target network.

# DQN and training hyperparameters:
INPUT_CHANNELS = 1
ACTION_DIM = 4                # 0 = up, 1 = down, 2 = left, 3 = right.
LR = 1e-3
# LR = 0.00025  # original value from Atari paper
GAMMA = 0.99
BATCH_SIZE = 32
INITIAL_BUFFER_SIZE = 25000  # Start training after this many steps.
BUFFER_CAPACITY = 100000  # changed from 1000000
EPSILON_START = 0.9
EPSILON_LOAD_OVERWRITE = True  # If True, will overwrite epsilon from checkpoint.
EPSILON_END = 0.001
EPSILON_DECAY = 0.9999

FRAME_STACK_SIZE = 4  # Number of consecutive frames to stack
INPUT_CHANNELS = FRAME_STACK_SIZE  # Instead of 1, now we have 4 channels


# Pygame screen and game settings:
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
FIXED_MAZE_LAYOUT = generate_maze(ROWS, COLS)


# -----------------------------
# Pacman Environment
# -----------------------------
class PacmanEnv:
    def __init__(self, fixed_maze=False):
        self.fixed_maze = fixed_maze
        self.screen_width = COLS * TILE_SIZE
        self.screen_height = ROWS * TILE_SIZE
        print(f"Screen size: {self.screen_width}x{self.screen_height}")
        pygame.display.set_caption("Pac-Man RL")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.frame_stack = deque(maxlen=FRAME_STACK_SIZE)  # For frame stacking

        self.reset()

    def reset(self):
        if self.fixed_maze:
            maze_layout = FIXED_MAZE_LAYOUT
        else:
            maze_layout = generate_maze(ROWS, COLS)

        self.maze_obj = Maze(maze_layout)
        open_cells = get_open_cells(maze_layout)
        if not open_cells:
            raise Exception("No open cells in maze!")
        
        ghosts = []
        ghost_cells = open_cells[:]  # copy list
        random.shuffle(ghost_cells)
        for _ in range(3):
            if ghost_cells:
                cell = ghost_cells.pop()
                ghost_x = cell[1] * TILE_SIZE + TILE_SIZE // 2
                ghost_y = cell[0] * TILE_SIZE + TILE_SIZE // 2
                ghosts.append(Ghost(ghost_x, ghost_y))
        self.ghosts = ghosts

        pac_cell = random.choice(open_cells)
        pac_x = pac_cell[1] * TILE_SIZE + TILE_SIZE // 2
        pac_y = pac_cell[0] * TILE_SIZE + TILE_SIZE // 2
        pac_x, pac_y = safe_spawn_pacman(maze_layout, ghosts)
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
        # return self.get_state()
    
        # Get the initial frame and fill the frame stack
        initial_frame = self._get_frame()
        self.frame_stack.clear()
        for _ in range(FRAME_STACK_SIZE):
            self.frame_stack.append(initial_frame)
        
        return self._get_stacked_state()

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

    def _action_to_direction(self, action, base_direction):
        # Convert an action (0: forward, 1: left, 2: right, 3: reverse)
        # into a direction vector relative to the current base_direction.
        if action == 0:
            # print("forward")
            return base_direction
        elif action == 1:
            # print("left")
            return pygame.math.Vector2(-base_direction.y, base_direction.x)
        elif action == 2:
            # print("right")
            return pygame.math.Vector2(base_direction.y, -base_direction.x)
        elif action == 3:
            # print("reverse")
            return -base_direction

    def legal_actions(self):
        legal = []
        for action in range(ACTION_DIM):
            candidate = self._action_to_direction(action, self.last_direction)
            new_x = self.pacman.x + candidate.x * self.pacman.speed
            new_y = self.pacman.y + candidate.y * self.pacman.speed
            if not self.pacman.collides_with_wall(new_x, new_y, self.maze_obj):
                legal.append(action)
        return legal

    def step(self, action):
        pygame.event.pump()
        dist_before = self.distance_to_nearest_pellet()
        # legal = self.legal_actions()

        # Convert the chosen action to a candidate direction.
        candidate = self._action_to_direction(action, self.last_direction)
        # new_x = self.pacman.x + candidate.x * self.pacman.speed
        # new_y = self.pacman.y + candidate.y * self.pacman.speed

        # If the candidate move is illegal, force a legal one if available.
        # while action not in legal:
        #     print(f"Illegal action {action} attempted. Legal actions: {legal}")
        #     if self.pacman.collides_with_wall(new_x, new_y, self.maze_obj):
        #         print("Collision detected. Adjusting direction.")
                    
        # if self.pacman.collides_with_wall(new_x, new_y, self.maze_obj):
        #     if legal:
        #         forced_action = random.choice(legal)
        #         candidate = self._action_to_direction(forced_action, self.last_direction)
            # Otherwise, if no legal moves exist, keep current direction.

        # Update last_direction and Pac-Man’s intended_direction with the final candidate.
        self.last_direction = candidate
        self.pacman.intended_direction = candidate

        reward = 0.0
        pre_pellet = len(self.maze_obj.pellets)
        pre_fruits = len(self.maze_obj.fruits)

        self.pacman.update(self.maze_obj)
        for ghost in self.ghosts:
            ghost.vulnerable = (pygame.time.get_ticks() < self.pacman.powerpellet_end)
            ghost.update(self.maze_obj, self.pacman)

        for ghost in self.ghosts:
            distance = math.hypot(self.pacman.x - ghost.x, self.pacman.y - ghost.y)
            if distance < self.pacman.radius + ghost.radius:
                if pygame.time.get_ticks() < self.pacman.powerpellet_end:
                    self.pacman.score += 10
                    ghost_cell = random.choice(get_open_cells(self.maze_obj.layout))
                    ghost.x = ghost_cell[1] * TILE_SIZE + TILE_SIZE // 2
                    ghost.y = ghost_cell[0] * TILE_SIZE + TILE_SIZE // 2
                    ghost.vulnerable = False
                else:
                    self.done = True
                    reward -= 10
                    break

        post_pellet = len(self.maze_obj.pellets)
        fruit_post = len(self.maze_obj.fruits)
        pellets_collected = pre_pellet - post_pellet
        reward += pellets_collected * 5
        fruits_collected = pre_fruits - fruit_post
        reward += fruits_collected * 7.5

        # new_tile = (int(self.pacman.y // TILE_SIZE), int(self.pacman.x // TILE_SIZE))
        # if new_tile == self.old_tile:
        #     reward -= 0.1
        # self.old_tile = new_tile

        if (not self.maze_obj.pellets) and (not self.maze_obj.fruits) and (not self.maze_obj.power_pellets):
            reward += 50   # Bonus for clearing the board.
            self.done = True

        # Distance based learning. off whiel we try frame stacking...
        dist_after = self.distance_to_nearest_pellet()

        if dist_after < dist_before:
            reward += 0.1

        if HEADLESS:
            self.draw_offscreen()
        else:
            self.render()

        new_frame = self._get_frame()
        self.frame_stack.append(new_frame)
        next_state = self._get_stacked_state()
        return next_state, reward, self.done, {}


    def _get_frame(self):
        """Capture the current screen as a grayscale image."""
        image = pygame.surfarray.array3d(self.screen)
        image = np.transpose(image, (1, 0, 2))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (84, 84))
        image = image.astype(np.float32) / 255.0
        return image  # shape: (84, 84)

    def _get_stacked_state(self):
        """Stack the frames along the channel dimension."""
        # Convert the deque to a numpy array with shape (FRAME_STACK_SIZE, 84, 84)
        stacked_state = np.array(self.frame_stack)
        # Optionally add a batch dimension if needed later: (1, FRAME_STACK_SIZE, 84, 84)
        return stacked_state

    def draw_offscreen(self):
        self.screen.fill(BLACK)
        self.maze_obj.draw(self.screen)
        self.pacman.draw(self.screen)

    def render(self):
        self.screen.fill(BLACK)
        self.maze_obj.draw(self.screen)
        self.pacman.draw(self.screen)
        for ghost in self.ghosts:
            ghost.draw(self.screen)
        pygame.display.flip()

    def get_state(self):
        image = pygame.surfarray.array3d(self.screen)
        image = np.transpose(image, (1, 0, 2))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (84, 84))
        image = image.astype(np.float32) / 255.0

        if DEBUG:
            cv2.imshow("State", image)
            input("Check image state. Press Enter to continue...")

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

class DuelingDQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(DuelingDQN, self).__init__()
        # Shared convolutional feature extractor (same as before)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Compute the flattened size after conv layers
        # (Here we assume the output size is 7x7 based on input size 84x84.)
        self.fc_input_dim = 7 * 7 * 64

        # Value stream
        self.value_fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # Advantage stream
        self.advantage_fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        value = self.value_fc(x)  # shape: [batch, 1]
        advantage = self.advantage_fc(x)  # shape: [batch, output_dim]
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")  # This should print "Using device: mps"

# DuelyDQN
policy_net = DuelingDQN(INPUT_CHANNELS, ACTION_DIM).to(device)
target_net = DuelingDQN(INPUT_CHANNELS, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Single DQ
# policy_net = DQN(INPUT_CHANNELS, ACTION_DIM).to(device)
# target_net = DQN(INPUT_CHANNELS, ACTION_DIM).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
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
    if EPSILON_LOAD_OVERWRITE:
        epsilon=EPSILON_START
    best_avg_reward = checkpoint.get("best_avg_reward", float('-inf'))
    print(f"🏆 Resumed from best checkpoint with epsilon {epsilon:.3f} and best_avg_reward {best_avg_reward:.2f}.")
elif os.path.exists(LATEST_CHECKPOINT_PATH):
    checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    if EPSILON_LOAD_OVERWRITE:
        epsilon=EPSILON_START
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

global_train_step = 0  # At module level

def train_step_double_dqn(episode, total_frames, episode_reward):
    global global_train_step
    if len(replay_buffer) < INITIAL_BUFFER_SIZE:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    # Current Q-values for chosen actions:
    q_values = policy_net(states).gather(1, actions)
    
    # --- Double DQN target calculation ---
    # Use policy_net to pick the best next action:
    best_next_actions = policy_net(next_states).argmax(1, keepdim=True)
    # Use target_net to evaluate those best actions:
    next_q_values = target_net(next_states).gather(1, best_next_actions)
    # Compute the target Q-value using the Bellman equation:
    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    # -------------------------------------

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    global_train_step += 1
    if global_train_step % 100 == 0:
        avg_q = q_values.mean().item()
        loss_value = f"{loss.item():.4f}"
        avg_q_value = f"{avg_q:.4f}"
        writer.writerow([episode, global_train_step, total_frames, loss_value, avg_q_value, epsilon, episode_reward])
        csvfile.flush()


def train_step(episode, total_frames, episode_reward):
    global global_train_step
    if len(replay_buffer) < INITIAL_BUFFER_SIZE:
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

    global_train_step += 1
    if global_train_step % 100 == 0:
        avg_q = q_values.mean().item()
        loss_value = f"{loss.item():.4f}"
        avg_q_value = f"{avg_q:.4f}"
        # print(f"Step {global_train_step} => Loss: {loss_value}, Avg Q: {avg_q_value}, Epsilon: {epsilon:.3f}")
        writer.writerow([episode, global_train_step, total_frames, loss_value, avg_q_value, epsilon, episode_reward])
        csvfile.flush()



def get_action_from_direction(candidate, current_direction):
    """
    Given a candidate movement direction (a pygame.Vector2) and the current
    direction (a pygame.Vector2), return a relative action index:
      0: go forward (same as current_direction)
      1: turn left (90° left of current_direction)
      2: turn right (90° right of current_direction)
      3: reverse (180° turn)
    
    If the candidate does not exactly equal one of these, the one with the highest
    dot product is chosen.
    """
    # Normalize current_direction (default to up if zero)
    if current_direction.length() == 0:
        forward = pygame.math.Vector2(0, -1)
    else:
        forward = current_direction.normalize()
    
    # Define the relative directions.
    left = pygame.math.Vector2(-forward.y, forward.x)
    right = pygame.math.Vector2(forward.y, -forward.x)
    reverse = -forward

    # Normalize candidate (default to up if zero)
    if candidate.length() == 0:
        cand = pygame.math.Vector2(0, -1)
    else:
        cand = candidate.normalize()

    # Compute dot products with each of the four directions.
    dots = [
        cand.dot(forward),  # forward
        cand.dot(left),     # left
        cand.dot(right),    # right
        cand.dot(reverse)   # reverse
    ]
    # Return the index of the maximum dot product.
    action = dots.index(max(dots))
    return action


# -----------------------------
# Training Loop
# -----------------------------
def train_dqn():
    global epsilon, best_avg_reward
    env = PacmanEnv(fixed_maze=FIXED_MAZE)
    total_frames = 0
    best_episode_reward = float('-inf')
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        episodes_since_improvement = 0
        best_window_reward = float('-inf')
        patience = 250
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = select_action(state, epsilon)

            # This is where the rewards are calculated
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # if len(replay_buffer) < INITIAL_BUFFER_SIZE:  # Use BFS navigation until buffer fills
            #     env.pacman.auto_navigate(env.maze_obj)  # Use auto-navigation
            #     action = get_action_from_direction(env.pacman.intended_direction, env.last_direction)
            # else:
            #     action = select_action(state, epsilon)  # Use trained policy

            # next_state, reward, done, _ = env.step(action)

            # episode_reward += reward
            # steps += 1
            # replay_buffer.push(state, action, reward, next_state, done)

            train_step_double_dqn(episode, total_frames, episode_reward)
            # train_step(episode, total_frames, episode_reward)
            # print("normal trainig has ensued")
            total_frames += 1

            if total_frames % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if best_episode_reward  < episode_reward:
            best_episode_reward = episode_reward


        episode_rewards.append(episode_reward)
        print(f"Episode {episode} => Reward: {episode_reward:.2f}, Highest Reward: {best_episode_reward:2f} Steps: {steps}, Total Frames: {total_frames}, Replay_Buffer: {len(replay_buffer)}, Epsilon: {epsilon:.3f}")
        # print(f"Episode {episode} => Reward: {episode_reward:.2f}, Steps: {steps}, Epsilon: {epsilon:.3f}")

        # Decay epsilon
        if replay_buffer and len(replay_buffer) >= INITIAL_BUFFER_SIZE:
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode > 0 and episode % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10.0
            print(f"Average reward over last 10 episodes: {avg_reward:.2f} - best so far: {best_avg_reward:.2f}")
            # Save latest checkpoint (always update latest)
            latest_checkpoint = {"model_state": policy_net.state_dict(), "epsilon": epsilon}
            torch.save(latest_checkpoint, LATEST_CHECKPOINT_PATH)
            print(f"📌 Latest checkpoint saved at episode {episode}")
            # Save best model only if current avg_reward exceeds best_avg_reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                episodes_since_improvement = 0
                best_checkpoint = {"model_state": policy_net.state_dict(), "epsilon": epsilon, "best_avg_reward": best_avg_reward}
                torch.save(best_checkpoint, BEST_CHECKPOINT_PATH)
                print(f"🏆 New best model saved at episode {episode} with average reward {avg_reward:.2f}")
            else:
                episodes_since_improvement += 10
            
            if episodes_since_improvement >= patience:
                print("Early stopping: No significant improvement in average reward.")
                break

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

    env = PacmanEnv(fixed_maze=True)
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
