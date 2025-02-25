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
    BEST_SINGLE_CHECKPOINT_PATH,  # New parameter for best single-episode checkpoint
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
    PLAY_CHECKPOINT,  # New setting to choose which checkpoint to use in play mode
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

# -----------------------------
# Initialize DQN, Replay Buffer, Optimizer
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

policy_net = DuelingDQN(INPUT_CHANNELS, ACTION_DIM).to(device)
target_net = DuelingDQN(INPUT_CHANNELS, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

epsilon = EPSILON_START  # Default starting epsilon
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


def select_action_with_inertia(state, epsilon, last_action=None, inertia=1.0):
    # With probability epsilon, choose a random action
    if random.random() < epsilon:
        return random.randrange(ACTION_DIM)

    # Otherwise, select the best action from the Q-network
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = policy_net(state_tensor).squeeze()

    # If we have a previous action, add a bias to it
    if last_action is not None:
        biased_q = q_values.clone()
        biased_q[last_action] += inertia  # Increase the value of the last action
        return biased_q.argmax().item()
    else:
        return q_values.argmax().item()


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

    q_values = policy_net(states).gather(1, actions)
    best_next_actions = policy_net(next_states).argmax(1, keepdim=True)
    next_q_values = target_net(next_states).gather(1, best_next_actions)
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
        writer.writerow([episode, global_train_step, total_frames, loss_value, avg_q_value, epsilon, episode_reward])
        csvfile.flush()


def get_action_from_direction(candidate, current_direction):
    if current_direction.length() == 0:
        forward = pygame.math.Vector2(0, -1)
    else:
        forward = current_direction.normalize()
    left = pygame.math.Vector2(-forward.y, forward.x)
    right = pygame.math.Vector2(forward.y, -forward.x)
    reverse = -forward
    if candidate.length() == 0:
        cand = pygame.math.Vector2(0, -1)
    else:
        cand = candidate.normalize()
    dots = [cand.dot(forward), cand.dot(left), cand.dot(right), cand.dot(reverse)]
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
    best_single_reward = float("-inf")  # Track the best single-episode reward
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        episodes_since_improvement = 0
        patience = 250
        steps = 0

        # 4 steps
        # while not done and steps < MAX_STEPS_PER_EPISODE:
        #     action = select_action(state, epsilon)
        #     total_reward = 0
        #     for _ in range(4):
        #         next_state, reward, done, _ = env.step(action)
        #         total_reward += reward
        #         if done:
        #             break
        #     replay_buffer.push(state, action, total_reward, next_state, done)
        #     state = next_state
        #     episode_reward += total_reward
        #     steps += 1
        #     train_step_double_dqn(episode, total_frames, episode_reward)
        #     total_frames += 1
        #     if total_frames % TARGET_UPDATE_FREQ == 0:
        #         target_net.load_state_dict(policy_net.state_dict())

        # Single Step
        last_action = None
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = select_action_with_inertia(
                state,
                epsilon,
                last_action=last_action if steps > 0 else None,
                inertia=1,
            )
            last_action = action
            # action = select_action(state, epsilon)
            # print(f"action={action}".strip())
            # input()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            train_step_double_dqn(episode, total_frames, episode_reward)
            total_frames += 1
            if total_frames % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if best_episode_reward < episode_reward:
            best_episode_reward = episode_reward

        # Save best single-episode model if this episode beats previous best
        if episode_reward > best_single_reward:
            best_single_reward = episode_reward
            best_single_checkpoint = {
                "model_state": policy_net.state_dict(),
                "epsilon": epsilon,
                "best_single_reward": best_single_reward,
            }
            torch.save(best_single_checkpoint, BEST_SINGLE_CHECKPOINT_PATH)
            print(f"🌟 New best single-episode model saved at episode {episode} with reward {episode_reward:.2f}")

        episode_rewards.append(episode_reward)
        print(
            f"Episode {episode} => Reward: {episode_reward:.2f}, Highest Reward: {best_episode_reward:.2f} Steps: {steps}, Total Frames: {total_frames}, Replay_Buffer: {len(replay_buffer)}, Epsilon: {epsilon:.3f}"
        )

        if replay_buffer and len(replay_buffer) >= INITIAL_BUFFER_SIZE:
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode > 0 and episode % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10.0
            print(f"Average reward over last 10 episodes: {avg_reward:.2f} - best so far: {best_avg_reward:.2f}")
            latest_checkpoint = {"model_state": policy_net.state_dict(), "epsilon": epsilon}
            torch.save(latest_checkpoint, LATEST_CHECKPOINT_PATH)
            print(f"📌 Latest checkpoint saved at episode {episode}")
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                episodes_since_improvement = 0
                best_checkpoint = {
                    "model_state": policy_net.state_dict(),
                    "epsilon": epsilon,
                    "best_avg_reward": best_avg_reward,
                }
                torch.save(best_checkpoint, BEST_CHECKPOINT_PATH)
                print(f"🏆 New best average model saved at episode {episode} with average reward {avg_reward:.2f}")
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
    # Decide which checkpoint to load based on PLAY_CHECKPOINT setting.
    if PLAY_CHECKPOINT == "best_single":
        checkpoint_path = BEST_SINGLE_CHECKPOINT_PATH
    elif PLAY_CHECKPOINT == "best_avg":
        checkpoint_path = BEST_CHECKPOINT_PATH
    else:
        checkpoint_path = LATEST_CHECKPOINT_PATH

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint["model_state"])
        policy_net.eval()
        print(f"🏆 Loaded trained model from {checkpoint_path} for play mode.")
    else:
        print("🚨 No checkpoint found. Exiting play mode.")
        return

    play_epsilon = 0.0
    env = PacmanEnv(fixed_maze=True)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = select_action(state, play_epsilon)
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
