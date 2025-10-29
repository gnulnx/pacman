import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_channels=3, n_actions=4):
        super().__init__()
        # conv layers: small receptive field, handles any spatial input
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # global pooling → fixed-length regardless of input size
        self.head = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        # x shape: [B, 3, H, W]
        feats = self.conv(x)
        pooled = feats.mean(dim=[2, 3])  # global average pooling
        q = self.head(pooled)
        return q


class Agent:
    def __init__(self, obs_shape, n_actions, lr=1e-3, gamma=0.99, device=None, memory_size=10000):
        # ✅ auto-detect best device
        if not device:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        input_channels = obs_shape[0]  # first dimension = number of channels (3)
        self.model = DQN(input_channels, n_actions).to(self.device)
        self.target = DQN(input_channels, n_actions).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.n_actions = n_actions
        self.lr = lr
        self.obj_shape = obs_shape

    def select_action(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.model(state)
            return int(torch.argmax(qvals).item())

    def remember(self, transition):
        # dequeu setup with maxlen.  So we pop old ones automatically if self.memory > maxlen
        self.memory.append(transition)

    # Padded replay buffer to avoid errors when buffer is smaller than batch size
    def replay(self, batch_size=64, gamma=0.99):
        """Sample random experiences and perform one SGD step, with automatic shape padding."""
        if len(self.memory) < batch_size:
            return

        # --- Sample minibatch ---
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # --- Detect or define target shape ---
        # If agent has fixed obs_shape (from init), use that; else infer from largest sample
        target_shape = getattr(self, "obs_shape", None)
        if target_shape is None:
            max_h = max(s.shape[1] for s in states)
            max_w = max(s.shape[2] for s in states)
            target_shape = (states[0].shape[0], max_h, max_w)

        # --- Pad helper ---
        def pad_state(state, target_shape):
            c, h, w = state.shape
            _, th, tw = target_shape
            if (h, w) == (th, tw):
                return state
            padded = np.zeros((c, th, tw), dtype=np.float32)
            padded[:, :h, :w] = state
            return padded

        # --- Pad all states to uniform size ---
        states = [pad_state(s, target_shape) for s in states]
        next_states = [pad_state(ns, target_shape) for ns in next_states]

        # --- Convert to tensors ---
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # --- Compute targets ---
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target(next_states).max(1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

        # --- Optimize ---
        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # def replay(self, batch_size=64):
    #     if len(self.memory) < batch_size:
    #         return
    #     batch = random.sample(self.memory, batch_size)
    #     states, actions, rewards, next_states, dones = zip(*batch)
    #     states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
    #     actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
    #     rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
    #     next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
    #     dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

    #     q_values = self.model(states).gather(1, actions)
    #     next_q = self.target(next_states).max(1)[0].unsqueeze(1)

    #     # This is the Bellman target you’ve seen conceptually:
    #     # yi​=ri​+γ(1−donei​)a′max​Qtarget​(si′​,a′)
    #     target = rewards + (1 - dones) * self.gamma * next_q

    #     loss = nn.functional.mse_loss(q_values, target)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
