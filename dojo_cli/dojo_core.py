# dojo_core.py
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from dojo_agent import Agent
from dojo_train_impl import preprocess_state, save_state, train_stage
from pacman_env import ACTIONS, Config, MazeSpec, PacmanEnv


@dataclass
class DojoConfig:
    base_dir: str = "runs"
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 10000
    device: str = "mps"
    render_every: int = 100


class DojoTrainer:
    """
    Orchestrates multi-stage training curricula.
    """

    def __init__(self, config: Optional[DojoConfig] = None):
        self.cfg = config or DojoConfig()

    def train_stage(self, stage_name: str, maze_spec: MazeSpec, **kwargs):
        print(f"🏋️ Training stage {stage_name} ({maze_spec.width}x{maze_spec.height})")
        train_stage(stage_name, maze_spec, **kwargs)

    def run(self, curriculum_fn: Callable):
        """
        Executes a curriculum generator that yields dicts
        containing stage_name, maze_spec, and other kwargs.
        """
        stage_counter = 1
        prev_model = None
        for params in curriculum_fn():
            params.setdefault("pretrained_path", prev_model)
            self.train_stage(**params)
            prev_model = f"runs/{params['stage_name']}/final_model.pt"
            stage_counter += 1
        print("✅ Curriculum finished. Final model:", prev_model)
        return prev_model

    def fine_tune_from_failures(
        self,
        model_path: str,
        failure_file: str = "failed_runs/final_model.pt.jsonl",
        mixed_sizes: bool = True,
        max_failures: int = 500,
        episodes_per_failure: int = 50,
    ):
        """
        Replays failed episodes to fine-tune an existing model.

        Parameters
        ----------
        model_path : str
            Path to the pretrained model (.pt) to fine-tune.
        failure_file : str
            Path to the .jsonl file recorded by FailedRunRecorder.
        mixed_sizes : bool
            If False, only replays failures that match the model's maze size.
        max_failures : int
            Limit to avoid thousands of replays at once.
        episodes_per_failure : int
            Number of episodes to train per failure replay.
        """
        if not os.path.exists(failure_file):
            print(f"⚠️ No failed run file found: {failure_file}")
            return

        with open(failure_file) as f:
            records = [json.loads(line) for line in f if line.strip()]

        if not records:
            print("⚠️ No failure records found.")
            return

        print(f"🧠 Loaded {len(records)} failures from {failure_file}")
        if not mixed_sizes:
            # Determine model's native size from first MazeSpec
            base_spec = records[0]
            base_size = (base_spec["width"], base_spec["height"])
            records = [r for r in records if (r["width"], r["height"]) == base_size]
            print(f"🎯 Filtered to same-size failures: {base_size}")

        # Limit the total number of failure replays
        records = records[:max_failures]

        # Run a mini training loop for each failure
        for i, rec in enumerate(records, 1):
            w, h = rec["width"], rec["height"]
            pellet_mode = rec.get("pellet_mode", "unknown")
            stage_name = f"finetune_{i:05d}_{w}x{h}"

            # Rebuild MazeSpec from record
            spec = MazeSpec(
                width=w,
                height=h,
                pellet_mode=pellet_mode,
                pacman_start=tuple(rec["pacman_start"]),
                pellet_positions=[tuple(p) for p in rec.get("pellet_positions", [])],
                surround_walls=True,
                include_ghosts=False,
            )

            print(f"🔁 [{i}/{len(records)}] Fine-tuning on {w}x{h} ({pellet_mode}) ...")
            train_stage(
                stage_name,
                maze_spec=spec,
                pretrained_path=model_path,
                episodes=episodes_per_failure,
                eps_start=0.3,
                eps_end=0.05,
            )

        print(f"✅ Fine-tune complete ({len(records)} failures processed).")

    def train_with_failure_mix(
        self,
        model_path: str,
        failure_file: str = "failed_runs/final_model.pt.jsonl",
        total_episodes: int = 10000,
        failure_prob: float = 0.25,
        eps_start: float = 0.5,
        eps_end: float = 0.05,
        eps_decay: int = 10000,
        width: int = 8,
        height: int = 8,
        output_dir: str = "runs/failure_mix",
    ):
        """
        Continuous training session that mixes normal random episodes with recorded failures.

        This fine-tuning phase continuously samples new environments and past failed
        episodes to improve model robustness and generalization. Each episode is
        either:

        • a random maze (pellet_mode="full") — normal exposure training, or
        • a recorded failure maze — loaded from `failed_runs/*.jsonl`, representing
            prior failure scenarios.

        Parameters
        ----------
        model_path : str
            Path to pretrained model (.pt) to fine-tune.
        failure_file : str
            JSONL file containing recorded failed mazes.
        total_episodes : int
            Total number of training episodes to run.
        failure_prob : float
            Probability that a given episode uses a failure maze.
        eps_start : float
            Starting epsilon for epsilon-greedy exploration.
        eps_end : float
            Minimum epsilon after decay.
        eps_decay : int
            Controls exponential epsilon decay.
        width, height : int
            Maze dimensions for normal episodes.

        Live Output
        -----------
        Every 50 episodes:
            Ep  800 | avgR= 35.80 | normR= 42.20 | failR= 21.47 | eps=0.287 | failure%=30.0

            • avgR — rolling mean reward over last ~200 episodes
            • normR — mean reward on random mazes
            • failR — mean reward on replayed failures
            • eps — current epsilon
            • failure% — % of episodes sourced from failures

        Saving Behavior
        ---------------
            best_overall.pt   — best overall average reward (avgR)
            best_failure.pt   — best average reward on failures (failR)
            checkpoint_XXXX.pt — snapshot every 1000 episodes
            final_model.pt    — final weights after completion
        """

        # --- Load failures once ---
        if not os.path.exists(failure_file):
            print(f"⚠️ No failed run file found at {failure_file}")
            return
        with open(failure_file) as f:
            failures = [json.loads(line) for line in f if line.strip()]
        if not failures:
            print("⚠️ No failures found.")
            return
        print(f"💾 Loaded {len(failures)} failure records for mixed replay.")

        # --- Base 8×8 environment (for random mode) ---
        base_spec = MazeSpec(width=width, height=height, pellet_mode="full", surround_walls=True)
        env = PacmanEnv(Config(maze_spec=base_spec, fps=2000), human_mode=False, headless=True)

        sample_state = preprocess_state(env.reset())
        agent = Agent(sample_state.shape, len(ACTIONS))

        # --- Load pretrained weights ---
        if model_path and os.path.exists(model_path):
            agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            agent.target.load_state_dict(agent.model.state_dict())
            print(f"✅ Loaded pretrained weights from {model_path}")

        total_steps = 0
        recent_rewards, normal_rewards, failure_rewards = [], [], []
        best_avgR, best_failR = -float("inf"), -float("inf")

        # os.makedirs("runs/failure_mix", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # --- Training loop ---
        for ep in range(total_episodes):
            # --- Choose environment source ---
            using_failure = random.random() < failure_prob
            if using_failure:
                # Pick random failure of exact matching size
                count = 0
                while True:
                    rec = random.choice(failures)
                    w, h = rec.get("width", width), rec.get("height", height)
                    if w == width and h == height:
                        break
                    count += 1
                    if count >= 1000:
                        print("⚠️ Could not find suitable failure record with matching size.")
                        break

                spec = MazeSpec(
                    width=w,
                    height=h,
                    pellet_mode=rec["pellet_mode"],
                    pacman_start=tuple(rec["pacman_start"]),
                    pellet_positions=[tuple(p) for p in rec.get("pellet_positions", [])],
                    surround_walls=True,
                    include_ghosts=False,
                )
            else:
                spec = base_spec

            env.close()
            env = PacmanEnv(Config(maze_spec=spec, fps=2000), human_mode=False, headless=True)

            # --- Run one episode ---
            state = preprocess_state(env.reset())
            done = False
            total_reward = 0
            while not done:
                epsilon = eps_end + (eps_start - eps_end) * np.exp(-1.0 * ep / eps_decay)
                action = agent.select_action(state, epsilon)
                raw_next, reward, done, info = env.step(action)
                next_state = preprocess_state(raw_next)
                agent.remember((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                total_steps += 1
                if total_steps % 10 == 0:
                    agent.replay(batch_size=32)

            # --- Reward tracking ---
            recent_rewards.append(total_reward)
            if len(recent_rewards) > 200:
                recent_rewards.pop(0)

            if using_failure:
                failure_rewards.append(total_reward)
                if len(failure_rewards) > 200:
                    failure_rewards.pop(0)
            else:
                normal_rewards.append(total_reward)
                if len(normal_rewards) > 200:
                    normal_rewards.pop(0)

            # --- Logging ---
            if ep % 50 == 0:
                avg = np.mean(recent_rewards)
                normR = np.mean(normal_rewards) if normal_rewards else 0
                failR = np.mean(failure_rewards) if failure_rewards else 0
                print(
                    f"Ep {ep:5d} | avgR={avg:6.2f} | normR={normR:6.2f} "
                    f"| failR={failR:6.2f} | eps={epsilon:.3f} | failure%={failure_prob*100:.1f}"
                )

            # --- Best model saving ---
            avg = np.mean(recent_rewards)
            failR = np.mean(failure_rewards) if failure_rewards else -float("inf")

            if avg > best_avgR:
                best_avgR = avg
                # torch.save(agent.model.state_dict(), "runs/failure_mix/best_overall.pt")
                save_state(
                    agent.model.state_dict(),
                    stage_name="failure_mix/best_overall",  # shouldn't really be used here...
                    current_spec=base_spec,
                    pretrained_path=model_path,
                    episodes=total_episodes,
                    max_possible=None,
                    success_rate=None,
                    avg=np.mean(recent_rewards),
                    std=np.std(recent_rewards),
                    output_dir=f"{output_dir}/best_overall",
                    model_name="final_model.pt",
                )

                print(f"💾 New best avgR={avg:.2f} at ep={ep}")

            if failR > best_failR:
                best_failR = failR
                # torch.save(agent.model.state_dict(), "runs/failure_mix/best_failure.pt")
                save_state(
                    agent.model.state_dict(),
                    stage_name="failure_mix/best_failure",
                    current_spec=base_spec,
                    pretrained_path=model_path,
                    episodes=total_episodes,
                    max_possible=None,
                    success_rate=None,
                    avg=np.mean(recent_rewards),
                    std=np.std(recent_rewards),
                    model_name="final_model.pt",
                )

                print(f"💾 New best failR={failR:.2f} at ep={ep}")

            # --- Periodic checkpoints ---
            if ep % 1000 == 0 and ep > 0:
                # torch.save(agent.model.state_dict(), f"runs/failure_mix/checkpoint_{ep}.pt")
                save_state(
                    agent.model.state_dict(),
                    stage_name=f"failure_mix/checkpoint_{ep}",
                    current_spec=base_spec,
                    pretrained_path=model_path,
                    episodes=total_episodes,
                    max_possible=None,
                    success_rate=None,
                    avg=np.mean(recent_rewards),
                    std=np.std(recent_rewards),
                    model_name="final_model.pt",
                )

        # --- Wrap up ---
        env.close()
        torch.save(agent.model.state_dict(), "runs/failure_mix/final_model.pt")
        print("✅ Mixed failure fine-tune complete → runs/failure_mix/final_model.pt")
