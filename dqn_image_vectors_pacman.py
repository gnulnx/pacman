import csv
import os
import random

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

from train.dueling_dqn import DuelingDQN
from train.env import PacmanEnv
from train.replay_buffer import ReplayBuffer
from train.settings import (
    ACTION_DIM,
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    BUFFER_CAPACITY,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_LOAD_OVERWRITE,
    EPSILON_START,
    FIXED_MAZE,
    GAMMA,
    INITIAL_BUFFER_SIZE,
    INPUT_CHANNELS,
    LATEST_CHECKPOINT_PATH,
    LR,
    MAX_STEPS_PER_EPISODE,
    MODE,
    NUM_EPISODES,
    TARGET_UPDATE_FREQ,
    USE_8BIT,
)

csvfile = open("training_log.csv", "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["episode", "step", "total_frames", "loss", "avg_q", "epsilon", "episode_reward"])


# For reproducibility:
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#     def __init__(self, input_channels, output_dim):
#         super(DuelingDQN, self).__init__()
#         # Shared convolutional feature extractor (same as before)
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#         )
#         # Compute the flattened size after conv layers
#         # (Here we assume the output size is 7x7 based on input size 84x84.)
#         self.fc_input_dim = 7 * 7 * 64

#         # Value stream
#         self.value_fc = nn.Sequential(nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, 1))
#         # Advantage stream
#         self.advantage_fc = nn.Sequential(nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, output_dim))

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)  # flatten
#         value = self.value_fc(x)  # shape: [batch, 1]
#         advantage = self.advantage_fc(x)  # shape: [batch, output_dim]
#         # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
#         q = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         return q


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

# Single DQN
# policy_net = DQN(INPUT_CHANNELS, ACTION_DIM).to(device)
# target_net = DQN(INPUT_CHANNELS, ACTION_DIM).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

# Attempt to load checkpoint: Prefer best checkpoint.
epsilon = EPSILON_START  # Default starting epsilon
# Also maintain best_avg_reward across runs:
best_avg_reward = float("-inf")
if os.path.exists(BEST_CHECKPOINT_PATH):
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    if EPSILON_LOAD_OVERWRITE:
        epsilon = EPSILON_START
    best_avg_reward = checkpoint.get("best_avg_reward", float("-inf"))
    print(f"🏆 Resumed from best checkpoint with epsilon {epsilon:.3f} and best_avg_reward {best_avg_reward:.2f}.")
elif os.path.exists(LATEST_CHECKPOINT_PATH):
    checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    if EPSILON_LOAD_OVERWRITE:
        epsilon = EPSILON_START
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

    if USE_8BIT:
        states = torch.tensor(states, dtype=torch.float32).to(device) / 255.0
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device) / 255.0
    else:
        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)

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
        cand.dot(left),  # left
        cand.dot(right),  # right
        cand.dot(reverse),  # reverse
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
    best_episode_reward = float("-inf")
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        episodes_since_improvement = 0
        best_window_reward = float("-inf")
        patience = 250
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            # 4 frames at a time
            action = select_action(state, epsilon)
            total_reward = 0  # Accumulate rewards over skipped frames

            for _ in range(4):  # Skip 4 frames
                next_state, reward, done, _ = env.step(action)
                total_reward += reward  # Sum rewards from skipped frames
                if done:
                    break  # Stop if episode ends

            # Use only the LAST frame in the state
            replay_buffer.push(state, action, total_reward, next_state, done)
            state = next_state
            episode_reward += total_reward
            steps += 1

            # -- End 4 frames at a time  ---

            # 1 Frame at a time
            # One frame at a time
            # action = select_action(state, epsilon)
            # next_state, reward, done, _ = env.step(action)
            # episode_reward += reward
            # steps += 1

            # replay_buffer.push(state, action, reward, next_state, done)
            # state = next_state

            # -- end 1 frame at a time ---

            # BFS BUFFER FILL
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
            total_frames += 1

            if total_frames % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if best_episode_reward < episode_reward:
            best_episode_reward = episode_reward

        episode_rewards.append(episode_reward)
        print(
            f"Episode {episode} => Reward: {episode_reward:.2f}, Highest Reward: {best_episode_reward:2f} Steps: {steps}, Total Frames: {total_frames}, Replay_Buffer: {len(replay_buffer)}, Epsilon: {epsilon:.3f}"
        )
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
                best_checkpoint = {
                    "model_state": policy_net.state_dict(),
                    "epsilon": epsilon,
                    "best_avg_reward": best_avg_reward,
                }
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
