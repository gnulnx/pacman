import json
import os
from datetime import UTC, datetime
from typing import Any, Dict

import numpy as np
import pygame

from pacman_env import Config, MazeSpec, PacmanEnv


class FailedRunRecorder:
    """Collects, saves, and replays failed Pac-Man runs for retraining/debugging."""

    def __init__(self, run_name: str, save_dir: str = "failed_runs"):
        self.run_name = run_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = os.path.join(save_dir, f"{run_name}.jsonl")
        self._buffer = []

    # ------------------------------------------------------------------
    # RECORD FAILURES
    # ------------------------------------------------------------------
    def record_failure(
        self,
        env: PacmanEnv,
        episode: int,
        maze_spec: Any,
        final_state: Dict[str, Any],
        reward: float,
        steps: int,
        pellets_remaining: int,
        failure_reason: str,
    ) -> Dict[str, Any]:
        """Append one failed episode to memory buffer."""
        layout_str = ["".join(row) for row in env.maze._layout]
        walls = [list(w) for w in env.maze.walls]

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "episode": episode,
            "width": maze_spec.width,
            "height": maze_spec.height,
            "pellet_mode": maze_spec.pellet_mode,
            "pacman_start": list(maze_spec.pacman_start),
            "pellet_positions": [list(p) for p in (maze_spec.pellet_positions or [])],
            "final_state": {
                "pacman": list(final_state["pacman"]),
                "ghosts": [list(g) for g in final_state.get("ghosts", [])],
                "pellets": np.asarray(final_state["pellets"], dtype=int).tolist(),
            },
            "layout": layout_str,
            "walls": walls,
            "pellets_remaining": pellets_remaining,
            "steps": steps,
            "reward": reward,
            "failure_reason": failure_reason,
        }

        self._buffer.append(record)
        return record

    # ------------------------------------------------------------------
    def save(self):
        """Write buffered records to disk."""
        if not self._buffer:
            return
        with open(self.file_path, "a") as f:
            for rec in self._buffer:
                json.dump(rec, f)
                f.write("\n")
        print(f"💾 Saved {len(self._buffer)} failed runs to {self.file_path}")
        self._buffer.clear()

    # ------------------------------------------------------------------
    # ENVIRONMENT RECONSTRUCTOR
    # ------------------------------------------------------------------
    def build_env_from_record(self, rec: Dict[str, Any], fps: int = 30) -> PacmanEnv:
        """Rebuild Pac-Man environment exactly as it was recorded."""
        # If full layout stored → prefer that
        if "layout" in rec:
            cfg = Config(maze_layout=tuple(rec["layout"]), fps=fps)
            env = PacmanEnv(cfg, human_mode=False, headless=True)
            env.reset()
        else:
            # Fallback for older logs
            pellets = np.array(rec["final_state"]["pellets"], dtype=np.uint8)
            h, w = pellets.shape
            spec = MazeSpec(width=w, height=h, include_ghosts=False, pellet_mode="custom", surround_walls=True)
            cfg = Config(maze_spec=spec, fps=fps)
            env = PacmanEnv(cfg, human_mode=False, headless=True)
            env.reset()

        # Restore pellets + pacman position
        pellets = np.array(rec["final_state"]["pellets"], dtype=np.uint8)
        env.maze.pellets[: pellets.shape[0], : pellets.shape[1]] = pellets
        env.pacman.position = tuple(rec["final_state"]["pacman"])
        return env

    # ------------------------------------------------------------------
    # VIEWER
    # ------------------------------------------------------------------
    def view_failures(self, fps: int = 10):
        """Interactive viewer for failed runs."""
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
        header_h = 60
        index = 0

        env = self.build_env_from_record(records[index], fps=fps)
        tile = env.config.tile_size
        win_w, win_h = tile * env.config.width, tile * env.config.height + header_h
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Failed Run Viewer")

        # --------------------------
        def draw_header(surface, rec):
            surface.fill((25, 25, 25))
            w = surface.get_width()
            btn_w, btn_h = 80, 24
            btn_y = 4
            prev_rect = pygame.Rect(10, btn_y, btn_w, btn_h)
            next_rect = pygame.Rect(w - btn_w - 10, btn_y, btn_w, btn_h)
            pygame.draw.rect(surface, (60, 60, 60), prev_rect, border_radius=5)
            pygame.draw.rect(surface, (60, 60, 60), next_rect, border_radius=5)
            surface.blit(font.render("◀ Prev", True, (255, 255, 255)), (prev_rect.x + 10, prev_rect.y + 3))
            surface.blit(font.render("Next ▶", True, (255, 255, 255)), (next_rect.x + 10, next_rect.y + 3))
            meta = (
                f"{index+1}/{len(records)} | "
                f"Episode {rec['episode']} | Reason: {rec['failure_reason']} | "
                f"Pellets: {rec['pellets_remaining']} | "
                f"Reward: {rec['reward']:.2f} | Steps: {rec['steps']}"
            )
            meta_label = font.render(meta, True, (255, 255, 255))
            surface.blit(meta_label, (10, btn_y + btn_h + 6))
            return prev_rect, next_rect

        # --------------------------
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if y <= header_h:
                        if x < 120:  # left button area
                            index = (index - 1) % len(records)
                            env = self.build_env_from_record(records[index], fps=fps)
                        elif x > win_w - 120:  # right button area
                            index = (index + 1) % len(records)
                            env = self.build_env_from_record(records[index], fps=fps)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RIGHT, pygame.K_LEFT):
                        index = (index + (1 if event.key == pygame.K_RIGHT else -1)) % len(records)
                        env = self.build_env_from_record(records[index], fps=fps)

            # Draw frame
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

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.save()


if __name__ == "__main__":
    recorder = FailedRunRecorder("final_model.pt")
    recorder.view_failures()
