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
from train.priotized_replay_buffer import PrioritizedReplayBuffer
from train.replay_buffer import ReplayBuffer
from train.settings import (
    ACTION_DIM,
    BATCH_SIZE,
    BEST_AVG_CHECKPOINT_PATH,  # New path for best average model
    BEST_SINGLE_CHECKPOINT_PATH,  # Checkpoint for best single-episode reward
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
    LOAD_CHECKPOINT,  # New flag to choose which checkpoint to load ("best_avg", "best_single", or "latest")
    LR,
    MAX_STEPS_PER_EPISODE,
    MODE,
    NOISY_DECAY,
    NUM_EPISODES,
    PER_ALPHA,
    PER_BETA_FRAMES,
    PER_BETA_START,
    TARGET_UPDATE_FREQ,
    USE_8BIT,
    USE_PRIORITY_BUFFER,
)

# Remove average_rewards.csv if it exists
if os.path.exists("average_rewards.csv"):
    os.remove("average_rewards.csv")

csvfile = open("training_log.csv", "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["episode", "step", "total_frames", "loss", "avg_q", "epsilon", "episode_reward"])

# For reproducibility:
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

policy_net = DuelingDQN(INPUT_CHANNELS, ACTION_DIM).to(device)
target_net = DuelingDQN(INPUT_CHANNELS, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)

if USE_PRIORITY_BUFFER:
    replay_buffer = PrioritizedReplayBuffer(BUFFER_CAPACITY, PER_ALPHA)
else:
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

epsilon = EPSILON_START  # default starting epsilon

# Global variables for tracking performance
best_single_reward = float("-inf")
best_avg_reward = float("-inf")

# --- Load checkpoint based on the LOAD_CHECKPOINT flag ---
if LOAD_CHECKPOINT == "best_single" and os.path.exists(BEST_SINGLE_CHECKPOINT_PATH):
    checkpoint = torch.load(BEST_SINGLE_CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    best_single_reward = float(checkpoint.get("best_single_reward", float("-inf")))
    if EPSILON_LOAD_OVERWRITE:
        epsilon = EPSILON_START
    print(
        f"🏆 Resumed from BEST SINGLE checkpoint with epsilon {epsilon:.3f} and best_single_reward {best_single_reward:.2f}."
    )
elif LOAD_CHECKPOINT == "best_avg" and os.path.exists(BEST_AVG_CHECKPOINT_PATH):
    checkpoint = torch.load(BEST_AVG_CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    best_avg_reward = float(checkpoint.get("best_avg_reward", float("-inf")))
    if EPSILON_LOAD_OVERWRITE:
        epsilon = EPSILON_START
    print(f"🏆 Resumed from BEST AVG checkpoint with epsilon {epsilon:.3f} and best_avg_reward {best_avg_reward:.2f}.")
elif os.path.exists(LATEST_CHECKPOINT_PATH):
    checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    if EPSILON_LOAD_OVERWRITE:
        epsilon = EPSILON_START
    print(f"📌 Resumed from LATEST checkpoint with epsilon {epsilon:.3f}.")
else:
    print("🚨 No checkpoint found. Starting from scratch.")


def select_action_with_inertia(state, epsilon, last_action=None, inertia=1.0):
    if random.random() < epsilon:
        return random.randrange(ACTION_DIM)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if USE_8BIT:
            state_tensor = state_tensor / 255.0
        q_values = policy_net(state_tensor).squeeze()
    if last_action is not None:
        biased_q = q_values.clone()
        biased_q[last_action] += inertia
        return biased_q.argmax().item()
    else:
        return q_values.argmax().item()


def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(ACTION_DIM)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            if USE_8BIT:
                state_tensor = state_tensor / 255.0
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()


def decay_noisy_params(model, decay_factor):
    for module in model.modules():
        if hasattr(module, "sigma"):
            module.sigma.data.mul_(decay_factor)


global_train_step = 0


def train_step_double_dqn(episode, total_frames, episode_reward, beta):
    global global_train_step
    if len(replay_buffer.buffer) < INITIAL_BUFFER_SIZE:  # Use .buffer if using PER
        return

    if USE_PRIORITY_BUFFER:
        # Sample with priorities
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(BATCH_SIZE, beta)
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        if USE_8BIT:
            states = torch.tensor(states, dtype=torch.float32).to(device) / 255.0
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device) / 255.0
        else:
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
    else:
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)  # Uniform sampling
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)

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

    # If using PER, weight the loss
    if USE_PRIORITY_BUFFER:
        loss_values = nn.MSELoss(reduction="none")(q_values, target_q_values)
        loss = (loss_values * weights_tensor).mean()
    else:
        loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # If using PER, update priorities using TD error (add a small constant for stability)
    if USE_PRIORITY_BUFFER:
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy().flatten() + 1e-6
        replay_buffer.update_priorities(indices, td_errors)

    global_train_step += 1
    if global_train_step % 100 == 0:
        avg_q = q_values.mean().item()
        writer.writerow(
            [episode, global_train_step, total_frames, f"{loss.item():.4f}", f"{avg_q:.4f}", epsilon, episode_reward]
        )
        csvfile.flush()


# In your main training loop, you need to compute beta for PER:
def train_dqn():
    global epsilon, best_single_reward, best_avg_reward
    env = PacmanEnv(fixed_maze=FIXED_MAZE)
    total_frames = 0
    best_episode_reward = float("-inf")
    episode_rewards = []
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        last_action = None

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = select_action_with_inertia(state, epsilon, last_action, inertia=1)
            last_action = action
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

            # Compute PER beta linearly annealed from PER_BETA_START to 1.0 over PER_BETA_FRAMES.
            if USE_PRIORITY_BUFFER:
                beta = min(1.0, PER_BETA_START + total_frames * (1.0 - PER_BETA_START) / PER_BETA_FRAMES)
            else:
                beta = 1.0

            train_step_double_dqn(episode, total_frames, episode_reward, beta)
            total_frames += 1
            if total_frames % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward

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
            f"Episode {episode} => Reward: {episode_reward:.2f}, Highest Reward: {best_episode_reward:.2f}, "
            f"Steps: {steps}, Total Frames: {total_frames}, Replay_Buffer: {len(replay_buffer.buffer) if USE_PRIORITY_BUFFER else len(replay_buffer)}, Epsilon: {epsilon:.3f}"
        )

        if (
            replay_buffer
            and (len(replay_buffer.buffer) if USE_PRIORITY_BUFFER else len(replay_buffer)) >= INITIAL_BUFFER_SIZE
        ):
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode > 0 and episode % 25 == 0:
            window_size = 25
            avg_reward = sum(episode_rewards[-window_size:]) / window_size
            print(f"Average reward over last {window_size} episodes: {avg_reward:.2f}")
            with open("average_rewards.csv", "a", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([episode, avg_reward])
            latest_checkpoint = {"model_state": policy_net.state_dict(), "epsilon": epsilon}
            torch.save(latest_checkpoint, LATEST_CHECKPOINT_PATH)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_avg_checkpoint = {
                    "model_state": policy_net.state_dict(),
                    "epsilon": epsilon,
                    "best_avg_reward": best_avg_reward,
                }
                torch.save(best_avg_checkpoint, BEST_AVG_CHECKPOINT_PATH)
                print(f"🏆 New best average model saved at episode {episode} with average reward {avg_reward:.2f}")

        decay_noisy_params(policy_net, NOISY_DECAY)

    env.close()


def evaluate_dqn(policy_net, env, num_episodes=5, max_steps=150):
    """
    Runs num_episodes with epsilon=0 (pure exploitation),
    prints the reward (and steps) for each episode, and then
    prints the average reward over those episodes.

    - policy_net.eval() temporarily disables noise for NoisyNets.
    - You can also pass a custom max_steps if your environment allows it.
    """
    policy_net.eval()  # Disables noise sampling in NoisyLinear layers if you're using NoisyNets

    total_rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            # Pure greedy action (epsilon=0)
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                if USE_8BIT:
                    state_tensor = state_tensor / 255.0
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

            state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        print(f"[EVAL] Episode {ep} => Reward: {episode_reward:.2f}, Steps: {steps}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"[EVAL] Over {num_episodes} episodes, average reward = {avg_reward:.2f}")

    # Switch back to training mode so we can keep training
    policy_net.train()
    return avg_reward


if __name__ == "__main__":
    pygame.init()
    if MODE == "train":
        train_dqn()
    elif MODE == "play":
        eval_env = PacmanEnv(fixed_maze=True)
        evaluate_dqn(policy_net, eval_env, num_episodes=5, max_steps=MAX_STEPS_PER_EPISODE)
        eval_env.close()
    pygame.time.delay(3000)
    pygame.quit()
