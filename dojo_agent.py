import random
import time  # noqa
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================
# DQN Model
# =============================
class DQN(nn.Module):
    def __init__(self, input_channels=3, n_actions=4):
        super().__init__()
        # convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # global pooling → fixed-size latent
        self.head = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        feats = self.conv(x)
        pooled = feats.mean(dim=[2, 3])  # global average pooling
        return self.head(pooled)


# =============================
# Replay Buffer
# =============================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.maxlen = capacity

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def push(self, transition, origin="self"):
        """
        Add transition to replay buffer.
        transition: (s, a, r, s', done) or (s, a, r, s', done, origin)
        origin: str tag (default: "self")
        """
        if len(transition) == 6:
            self.buffer.append(transition)
        else:
            self.buffer.append((*transition, origin))

    def extend(self, transitions, origin="self"):
        """Batch add multiple transitions efficiently using deque.extend()."""
        if not transitions:
            return

        if isinstance(transitions, (list, tuple)):
            # Normalize to 6-tuple with origin where needed
            formatted = (exp if len(exp) == 6 else (*exp, origin) for exp in transitions)
            self.buffer.extend(formatted)
        else:
            # Fallback single push
            self.push(transitions, origin=origin)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def counts(self):
        """Legacy: returns total self/teacher counts."""
        origins = [o for *_, o in self.buffer]
        n_self = sum(o == "self" for o in origins)
        n_teacher = len(origins) - n_self
        return {"self": n_self, "teacher": n_teacher, "total": len(self.buffer)}

    def counts_by_origin(self):
        """Full breakdown by all origin tags."""
        counts = {}
        for *_, o in self.buffer:
            counts[o] = counts.get(o, 0) + 1
        counts["total"] = len(self.buffer)
        return counts

    def report(self):
        """Human-readable summary string."""
        counts = self.counts_by_origin()
        total = counts.pop("total", 0)
        parts = [f"total={total}"]
        for k in sorted(counts.keys()):
            parts.append(f"{k}={counts[k]}")
        return ", ".join(parts)


# =============================
# Agent
# =============================
class Agent:
    def __init__(
        self,
        obs_shape,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        device=None,
        memory_size=10000,
        tb_writer=None,
    ):
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

        input_channels = obs_shape[0]
        self.model = DQN(input_channels, n_actions).to(self.device)
        self.target = DQN(input_channels, n_actions).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.memory = ReplayBuffer(memory_size)
        self.n_actions = n_actions
        self.lr = lr
        self.obs_shape = obs_shape  # fixed small typo (was obj_shape)
        self.writer = tb_writer
        self.global_step = 0

    # -----------------------------
    # Action selection (epsilon-greedy)
    # -----------------------------
    def select_action(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.model(state)
            return int(torch.argmax(qvals).item())

    # -----------------------------
    # Memory interface
    # -----------------------------
    def remember(self, transition, origin="self"):
        self.memory.push(transition, origin=origin)

    # -----------------------------
    # Training (single replay step)
    # -----------------------------
    def replay(self, batch_size=64, gamma=None, buffer=None):
        """
        Sample from buffer (if provided) or internal memory.
        buffer may be:
          - list/deque of 5- or 6-tuples
          - a ReplayBuffer instance
        """
        t0 = time.time()
        gamma = gamma or self.gamma

        # Resolve buffer reference
        if buffer is None:
            pool = self.memory.buffer
        else:
            pool = buffer.buffer if hasattr(buffer, "buffer") else buffer

        if len(pool) < batch_size:
            return

        minibatch = random.sample(pool, batch_size)
        five = [t[:5] for t in minibatch]  # handle both 5/6-tuples
        states, actions, rewards, next_states, dones = zip(*five)

        # Determine shape
        target_shape = getattr(self, "obs_shape", None)
        if target_shape is None:
            max_h = max(s.shape[1] for s in states)
            max_w = max(s.shape[2] for s in states)
            target_shape = (states[0].shape[0], max_h, max_w)

        # Pad helper
        def pad_state(state, target_shape):
            c, h, w = state.shape
            _, th, tw = target_shape
            if (h, w) == (th, tw):
                return state
            padded = np.zeros((c, th, tw), dtype=np.float32)
            padded[:, :h, :w] = state
            return padded

        same_shape = all(s.shape == states[0].shape for s in states)
        if not same_shape:  # Don't waste any time if all same shape
            states = [pad_state(s, target_shape) for s in states]
            next_states = [pad_state(ns, target_shape) for ns in next_states]

        # May be best approach on local hardware to not use np arrays
        states = np.stack(states, axis=0).astype(np.float32, copy=False)
        next_states = np.stack(next_states, axis=0).astype(np.float32, copy=False)
        states = torch.from_numpy(states).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q-learning update
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target(next_states).max(1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets.detach())

        if self.writer and loss is not None:
            self.writer.add_scalar("loss/td_error", loss.item(), self.global_step)
            self.global_step += 1

        # self.optimizer.zero_grad()

        # Optimize the model and use set_to_none=True to avoid redundant tensor writes.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # torch.mps.synchronize()
        # print(f"replay step time: {time.time() - t0:.4f}s")
        return time.time() - t0

    # -----------------------------
    # Target sync
    # -----------------------------
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
