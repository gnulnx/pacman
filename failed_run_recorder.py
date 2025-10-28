import json
import os
from datetime import UTC, datetime
from typing import Any, Dict

import numpy as np
import pygame

from pacman_env import Config, MazeSpec, PacmanEnv


class FailedRunRecorder:
    """
    Collects and saves failed Pac-Man runs for later retraining.

    Each record includes:
      - maze_spec (width, height, pellet_mode, etc.)
      - final_state (pacman, ghosts, pellets)
      - episode metrics (reward, steps, pellets_remaining)
      - failure_reason ("collision", "max_steps", etc.)
    """

    def __init__(self, run_name: str, save_dir: str = "failed_runs"):
        self.run_name = run_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = os.path.join(save_dir, f"{run_name}.jsonl")
        self._buffer = []

    def record_failure(
        self,
        episode: int,
        maze_spec: Any,
        final_state: Dict[str, Any],
        reward: float,
        steps: int,
        pellets_remaining: int,
        failure_reason: str,
    ) -> Dict[str, Any]:
        """Append one failed episode to memory buffer.

        Returns the created record."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "episode": episode,
            "width": maze_spec.width,
            "height": maze_spec.height,
            "pellet_mode": maze_spec.pellet_mode,
            "pacman_start": maze_spec.pacman_start,
            "pellet_positions": maze_spec.pellet_positions,
            "final_state": {
                "pacman": final_state["pacman"].tolist(),
                "ghosts": final_state["ghosts"].tolist(),
                "pellets": final_state["pellets"].astype(int).tolist(),
            },
            "pellets_remaining": pellets_remaining,
            "steps": steps,
            "reward": reward,
            "failure_reason": failure_reason,
        }
        self._buffer.append(record)

        return record

    def save(self):
        """Write all buffered records to disk (append mode)."""
        if not self._buffer:
            return
        with open(self.file_path, "a") as f:
            for rec in self._buffer:
                json.dump(rec, f)
                f.write("\n")
        self._buffer.clear()
        print(f"💾 Saved {len(self._buffer)} failed runs to {self.file_path}")

    def view_failures(self, fps: int = 10):
        """Interactive flicker-free viewer for failed runs."""
        if not os.path.exists(self.file_path):
            print(f"⚠️ No failed run file found at {self.file_path}")
            return

        records = [json.loads(line) for line in open(self.file_path) if line.strip()]
        if not records:
            print("No failures found.")
            return

        pygame.init()
        font = pygame.font.SysFont("monospace", 18)
        clock = pygame.time.Clock()
        header_h = 80
        index = 0

        # --- environment factory (headless) ---
        def build_env(rec):
            pellets = np.array(rec["final_state"]["pellets"], dtype=np.uint8)
            h, w = pellets.shape
            spec = MazeSpec(width=w, height=h, include_ghosts=False, pellet_mode="custom", surround_walls=False)
            cfg = Config(maze_spec=spec, fps=fps)
            env = PacmanEnv(cfg, human_mode=False, headless=True)
            env.maze.pellets = pellets.copy()
            env.pacman.position = tuple(rec["final_state"]["pacman"])
            return env

        def draw_header(surface, rec):
            surface.fill((25, 25, 25))
            w = surface.get_width()

            # --- button areas ---
            btn_w, btn_h = 80, 24
            btn_y = 4
            prev_rect = pygame.Rect(10, btn_y, btn_w, btn_h)
            next_rect = pygame.Rect(w - btn_w - 10, btn_y, btn_w, btn_h)

            # --- draw buttons ---
            pygame.draw.rect(surface, (60, 60, 60), prev_rect, border_radius=5)
            pygame.draw.rect(surface, (60, 60, 60), next_rect, border_radius=5)

            prev_label = font.render("◀ Prev", True, (255, 255, 255))
            next_label = font.render("Next ▶", True, (255, 255, 255))
            surface.blit(prev_label, (prev_rect.x + 10, prev_rect.y + 3))
            surface.blit(next_label, (next_rect.x + 10, next_rect.y + 3))

            # --- meta text below buttons ---
            meta = (
                f"{index+1}/{len(records)} | "
                f"Episode {rec['episode']} | "
                f"Reason: {rec['failure_reason']} | "
                f"Pellets: {rec['pellets_remaining']} | "
                f"Reward: {rec['reward']:.2f} | Steps: {rec['steps']}"
            )
            meta_label = font.render(meta, True, (255, 255, 255))
            surface.blit(meta_label, (10, btn_y + btn_h + 6))

            # return rectangles so click handler knows hit zones
            return prev_rect, next_rect

        env = build_env(records[index])
        tile = env.config.tile_size
        win_w, win_h = tile * env.config.width, tile * env.config.height + header_h
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Failed Run Viewer")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_RIGHT:
                        index = (index + 1) % len(records)
                        env = build_env(records[index])
                    elif event.key == pygame.K_LEFT:
                        index = (index - 1) % len(records)
                        env = build_env(records[index])
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if y <= header_h:
                        # click on header arrows
                        if x < 40:
                            index = (index - 1) % len(records)
                            env = build_env(records[index])
                        elif x > win_w - 40:
                            index = (index + 1) % len(records)
                            env = build_env(records[index])
                    else:
                        # ignore clicks on playfield
                        pass

            # --- compose frame manually ---
            frame = pygame.Surface((win_w, win_h))
            header = pygame.Surface((win_w, header_h))
            draw_header(header, records[index])
            frame.blit(header, (0, 0))

            rgb = env.render("rgb_array")
            surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            frame.blit(surf, (0, header_h))

            screen.blit(frame, (0, 0))
            pygame.display.flip()
            clock.tick(fps)

        env.close()
        pygame.quit()

    # def view_failures(self, fps: int = 10):
    #     """
    #     Interactive pygame viewer to step through saved failed runs.
    #     Use arrow keys ← / → to navigate and ESC to exit.
    #     """
    #     if not os.path.exists(self.file_path):
    #         print(f"⚠️ No failed run file found at {self.file_path}")
    #         return

    #     # Load all saved records
    #     records = []
    #     with open(self.file_path) as f:
    #         for line in f:
    #             if line.strip():
    #                 records.append(json.loads(line))

    #     if not records:
    #         print("No failures found.")
    #         return

    #     pygame.init()
    #     index = 0
    #     env = None
    #     clock = pygame.time.Clock()

    #     def build_env_from_record(rec):
    #         spec = MazeSpec(
    #             width=rec["width"],
    #             height=rec["height"],
    #             include_ghosts=False,
    #             pellet_mode="custom",
    #             # pellet_positions=rec["pellet_positions"],
    #             pellet_positions=[tuple(p) for p in rec.get("pellet_positions", [])],
    #             pacman_start=tuple(rec["pacman_start"]),
    #             surround_walls=True,
    #         )
    #         cfg = Config(maze_spec=spec, fps=fps)
    #         e = PacmanEnv(cfg, human_mode=False, headless=False)
    #         # Restore state
    #         state = rec["final_state"]
    #         e.maze.pellets = np.array(state["pellets"], dtype=np.uint8)
    #         e.pacman.position = tuple(state["pacman"])
    #         for g, pos in zip(e.ghosts, state["ghosts"]):
    #             g.position = tuple(pos)
    #         return e

    #     def render_header(surface, rec, font):
    #         text = (
    #             f"Episode {rec['episode']} | Reason: {rec['failure_reason']} | "
    #             f"Pellets Left: {rec['pellets_remaining']} | "
    #             f"Reward: {rec['reward']:.2f} | Steps: {rec['steps']}"
    #         )
    #         header = font.render(text, True, (255, 255, 255))
    #         surface.blit(header, (10, 10))

    #     running = True
    #     font = pygame.font.SysFont("monospace", 16)

    #     env = build_env_from_record(records[index])

    #     while running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False
    #             elif event.type == pygame.KEYDOWN:
    #                 if event.key == pygame.K_ESCAPE:
    #                     running = False
    #                 elif event.key == pygame.K_RIGHT:
    #                     index = (index + 1) % len(records)
    #                     env.close()
    #                     env = build_env_from_record(records[index])
    #                 elif event.key == pygame.K_LEFT:
    #                     index = (index - 1) % len(records)
    #                     env.close()
    #                     env = build_env_from_record(records[index])

    #         env.render("human")
    #         render_header(env.screen, records[index], font)
    #         pygame.display.flip()
    #         clock.tick(fps)

    #     env.close()
    #     pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.save()


if __name__ == "__main__":
    recorder = FailedRunRecorder("final_model.pt")
    recorder.view_failures()
