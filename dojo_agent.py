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
    def __init__(
        self,
        obs_shape,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=10000,
        memory_size=10000,
        memory=None,
    ):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ✅ auto-detect best device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print("Using", self.device)

        input_channels = obs_shape[0]  # first dimension = number of channels (3)
        self.model = DQN(input_channels, n_actions).to(self.device)
        self.target = DQN(input_channels, n_actions).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

        if memory is not None:
            self.memory = memory
        else:
            self.memory = deque(maxlen=memory_size)
        self.steps = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.epsilon = float(eps_start)
        self.n_actions = n_actions
        self.lr = lr

    def set_epsilon(self, value: float) -> None:
        """Clamp exploration to a fixed epsilon; keep decay history external."""
        self.epsilon = float(value)
        self.eps_start = float(value)

    def select_action(self, state, epsilon: float | None = None, steps: int | None = None):
        if epsilon is not None:
            current_eps = float(epsilon)
        elif steps is not None:
            current_eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * steps / self.eps_decay)
        else:
            current_eps = float(self.epsilon)

        self.epsilon = current_eps

        if random.random() < current_eps:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.model(state_t)
            return int(torch.argmax(qvals).item())

    def remember(self, transition):
        self.memory.append(transition)

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q = self.target(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
