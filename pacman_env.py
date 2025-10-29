"""Minimal modular Pac-Man environment built on pygame for RL experimentation."""

from __future__ import annotations

import os
import random
from collections import deque
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

with open(os.devnull, "w") as f, redirect_stdout(f):
    import pygame

    pygame.init()

GridPos = Direction = Tuple[int, int]

ACTIONS: Dict[int, Direction] = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


@dataclass
class MazeSpec:
    """Small helper describing simple rectangular mazes for RL curricula."""

    width: int
    height: int
    surround_walls: bool = True
    include_pellets: bool = True
    pellet_mode: str = "full"  # full, single, stripe_h, stripe_v, l_shape, custom
    pellet_positions: Optional[List[GridPos]] = None
    pellet_density: float = 1.0
    include_power_pellets: bool = False
    power_pellet_positions: Optional[List[GridPos]] = None
    include_ghosts: bool = True
    ghost_positions: Optional[List[GridPos]] = None
    pacman_start: GridPos = (0, 0)  # interior coordinates (0,0) top-left
    random_seed: Optional[int] = None


def generate_rect_layout(spec: MazeSpec) -> List[str]:
    """Create a simple rectangular maze from the provided specification."""
    rng = random.Random(spec.random_seed)
    width, height = spec.width, spec.height
    grid_width = width + 2 if spec.surround_walls else width
    grid_height = height + 2 if spec.surround_walls else height

    def to_grid(pos: GridPos) -> GridPos:
        x, y = pos
        if spec.surround_walls:
            return (x + 1, y + 1)
        return pos

    # --- Initialize grid with walls or spaces ---
    rows = (
        [["#"] * grid_width for _ in range(grid_height)]
        if spec.surround_walls
        else [[" "] * grid_width for _ in range(grid_height)]
    )

    # Fill interior cells as empty spaces
    for y in range(height):
        for x in range(width):
            gx, gy = to_grid((x, y))
            rows[gy][gx] = " "

    # --- Pellet placement logic ---
    pellets: List[GridPos] = []
    if spec.include_pellets:
        interior_coords = [(x, y) for y in range(height) for x in range(width)]
        start = spec.pacman_start
        interior_coords = [pos for pos in interior_coords if pos != start]

        if spec.pellet_mode == "full":
            for pos in interior_coords:
                if spec.pellet_density >= 1.0 or rng.random() <= spec.pellet_density:
                    pellets.append(pos)
        elif spec.pellet_mode == "single":
            if spec.pellet_positions:
                pellets.extend(spec.pellet_positions)
            else:
                pellets.append((width - 1, height - 1))
        elif spec.pellet_mode == "stripe_h":
            for y in range(height):
                if y % 2 == 0:
                    for x in range(width):
                        pellets.append((x, y))
        elif spec.pellet_mode == "stripe_v":
            for x in range(width):
                if x % 2 == 0:
                    for y in range(height):
                        pellets.append((x, y))
        elif spec.pellet_mode == "l_shape":
            for x in range(width):
                pellets.append((x, 0))
            for y in range(height):
                pellets.append((0, y))
        elif spec.pellet_mode == "custom" and spec.pellet_positions:
            pellets.extend(spec.pellet_positions)

    # Ensure pellet positions are valid
    pellets = [pos for pos in pellets if 0 <= pos[0] < width and 0 <= pos[1] < height]

    # --- Power pellets (optional) ---
    power_positions: List[GridPos] = []
    if spec.include_power_pellets:
        if spec.power_pellet_positions:
            power_positions = spec.power_pellet_positions
        else:
            power_positions = [
                (0, 0),
                (width - 1, 0),
                (0, height - 1),
                (width - 1, height - 1),
            ]
    power_positions = [pos for pos in power_positions if 0 <= pos[0] < width and 0 <= pos[1] < height]

    # --- Ghosts (optional) ---
    ghost_positions = spec.ghost_positions or []
    if spec.include_ghosts and not ghost_positions:
        default = (width - 1, height - 1)
        if default != spec.pacman_start:
            ghost_positions = [default]
        else:
            ghost_positions = [(width // 2, height // 2)]

    # --- Draw pellets first ---
    unique_pellets = set(pellets)
    for pos in unique_pellets:
        if pos == spec.pacman_start:
            continue  # never overwrite Pac-Man tile
        gx, gy = to_grid(pos)
        rows[gy][gx] = "."

    # --- Draw power pellets ---
    for pos in power_positions:
        gx, gy = to_grid(pos)
        rows[gy][gx] = "o"

    # --- Draw ghosts ---
    for pos in ghost_positions:
        if 0 <= pos[0] < width and 0 <= pos[1] < height:
            gx, gy = to_grid(pos)
            # do not overwrite Pac-Man
            if pos != spec.pacman_start:
                rows[gy][gx] = "G"

    # --- Finally, place Pac-Man last so he is never overwritten ---
    pacman_grid = to_grid(spec.pacman_start)
    rows[pacman_grid[1]][pacman_grid[0]] = "P"

    return ["".join(row) for row in rows]


def generate_connected_layout(size: int, seed: Optional[int] = None) -> List[str]:
    """Generate a loop-rich maze by braiding a DFS-carved grid to remove dead-ends."""
    size = max(7, size | 1)  # ensure odd dimension and reasonable minimum
    rng = random.Random(seed)
    grid = [["#"] * size for _ in range(size)]

    def carve(x: int, y: int) -> None:
        grid[y][x] = "."
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        rng.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and grid[ny][nx] == "#":
                grid[y + dy // 2][x + dx // 2] = "."
                carve(nx, ny)

    carve(1, 1)

    def open_neighbors(x: int, y: int) -> int:
        return sum(
            1
            for dx, dy in ACTIONS.values()
            if 0 <= x + dx < size and 0 <= y + dy < size and grid[y + dy][x + dx] == "."
        )

    changed = True
    while changed:
        changed = False
        for y in range(1, size - 1):
            for x in range(1, size - 1):
                if grid[y][x] == "." and open_neighbors(x, y) == 1:
                    candidates: List[GridPos] = []
                    for dx, dy in ACTIONS.values():
                        nx, ny = x + dx, y + dy
                        if 1 <= nx < size - 1 and 1 <= ny < size - 1 and grid[ny][nx] == "#":
                            candidates.append((nx, ny))
                    if candidates:
                        nx, ny = rng.choice(candidates)
                        grid[ny][nx] = "."
                        changed = True

    walkable = {(x, y) for y in range(size) for x in range(size) if grid[y][x] == "."}
    if not walkable:
        walkable = {(1, 1)}

    def farthest(start: GridPos) -> GridPos:
        queue: deque[Tuple[GridPos, int]] = deque([(start, 0)])
        visited = {start}
        best = (start, 0)
        while queue:
            (cx, cy), dist = queue.popleft()
            if dist > best[1]:
                best = ((cx, cy), dist)
            for dx, dy in ACTIONS.values():
                nxt = (cx + dx, cy + dy)
                if nxt in walkable and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return best[0]

    center = (size // 2, size // 2)
    pacman_pos = min(walkable, key=lambda pos: abs(pos[0] - center[0]) + abs(pos[1] - center[1]))
    ghost_pos = farthest(pacman_pos)
    grid[pacman_pos[1]][pacman_pos[0]] = "P"
    if ghost_pos != pacman_pos:
        grid[ghost_pos[1]][ghost_pos[0]] = "G"

    return ["".join(row) for row in grid]


@dataclass
class Config:
    """Shared configuration for the Pac-Man environment and game."""

    tile_size: int = 48
    maze_size: int = 17
    maze_layout: Optional[Sequence[str]] = None
    maze_spec: Optional[MazeSpec] = None
    fps: int = 10
    background_color: Tuple[int, int, int] = (0, 0, 0)
    wall_color: Tuple[int, int, int] = (0, 51, 153)
    pellet_color: Tuple[int, int, int] = (255, 204, 0)
    pacman_color: Tuple[int, int, int] = (255, 255, 0)
    ghost_color: Tuple[int, int, int] = (255, 0, 0)
    random_seed: Optional[int] = None
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:

        if self.maze_layout is None:
            if self.maze_spec is not None:
                self.maze_layout = tuple(generate_rect_layout(self.maze_spec))
            else:
                self.maze_layout = tuple(generate_connected_layout(self.maze_size, self.random_seed))

    @property
    def width(self) -> int:
        if self.maze_layout is None:
            raise ValueError("Maze layout is not initialized.")
        return len(self.maze_layout[0])

    @property
    def height(self) -> int:
        if self.maze_layout is None:
            raise ValueError("Maze layout is not initialized.")
        return len(self.maze_layout)


class Maze:
    """Grid-based maze holding walls, pellets, and spawn locations."""

    def __init__(self, layout: Sequence[str]) -> None:
        """Prepare wall and pellet structures from the provided layout."""
        self._layout = [list(row) for row in layout]
        self.width, self.height = len(self._layout[0]), len(self._layout)
        self.walls = {(x, y) for y, row in enumerate(self._layout) for x, ch in enumerate(row) if ch == "#"}
        self.pellets: np.ndarray = np.array(
            [[1 if ch in ".PGo" else 0 for ch in row] for row in self._layout], dtype=np.uint8
        )
        pacman_spawn: Optional[GridPos] = None
        ghost_spawns: List[GridPos] = []
        for y, row in enumerate(self._layout):
            for x, ch in enumerate(row):
                if ch == "P":
                    pacman_spawn = (x, y)
                elif ch == "G":
                    ghost_spawns.append((x, y))
        if pacman_spawn is None:
            for y, row in enumerate(self._layout):
                for x, ch in enumerate(row):
                    if ch != "#":
                        pacman_spawn = (x, y)
                        break
                if pacman_spawn is not None:
                    break
            if pacman_spawn is None:
                raise ValueError("Maze missing Pac-Man spawn and contains no walkable tiles.")
        self.pacman_spawn = pacman_spawn
        self.ghost_spawns = ghost_spawns
        self._validate_layout()
        # Preserve the post-validation pellet mask so reset() restores the exact layout
        self._initial_pellets = self.pellets.copy()

    def in_bounds(self, pos: GridPos) -> bool:
        """Check whether the position lies inside the maze."""
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def is_wall(self, pos: GridPos) -> bool:
        """Return True when the given tile is a wall."""
        return pos in self.walls

    def consume_pellet(self, pos: GridPos) -> bool:
        """Remove a pellet from the given tile and report whether one existed."""
        x, y = pos
        if self.pellets[y, x]:
            self.pellets[y, x] = 0
            return True
        return False

    def reset(self) -> None:
        """Restore the pellet field to its initial state."""
        np.copyto(self.pellets, self._initial_pellets)

    def _validate_layout(self) -> None:
        """Ensure every pellet is reachable; repair layout when needed."""
        walkable = {(x, y) for y in range(self.height) for x in range(self.width) if not self.is_wall((x, y))}
        if not walkable:
            raise ValueError("Maze contains no walkable tiles.")

        def nearest_walkable(target: GridPos) -> GridPos:
            return min(walkable, key=lambda pos: abs(pos[0] - target[0]) + abs(pos[1] - target[1]))

        if self.pacman_spawn not in walkable:
            original = self.pacman_spawn
            self.pacman_spawn = nearest_walkable(original)

        queue: deque[GridPos] = deque([self.pacman_spawn])
        visited = {self.pacman_spawn}
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nxt = (nx, ny)
                if nxt in walkable and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        for y in range(self.height):
            for x in range(self.width):
                if self.pellets[y, x] and (x, y) not in visited:
                    self.pellets[y, x] = 0
                    if self._layout[y][x] == ".":
                        self._layout[y][x] = " "

        repaired_ghosts: List[GridPos] = []
        for pos in self.ghost_spawns:
            if pos in walkable and pos not in repaired_ghosts:
                repaired_ghosts.append(pos)
        self.ghost_spawns = repaired_ghosts

        for y, row in enumerate(self._layout):
            for x, ch in enumerate(row):
                if ch in "PG":
                    self._layout[y][x] = "."
        px, py = self.pacman_spawn
        self._layout[py][px] = "P"
        self.pellets[py, px] = 0
        for gx, gy in self.ghost_spawns:
            if (gx, gy) != self.pacman_spawn:
                self._layout[gy][gx] = "G"
            self.pellets[gy, gx] = 1


class Pacman:
    """Player-controlled agent with discrete grid movement."""

    def __init__(self, start: GridPos) -> None:
        """Create Pac-Man at the supplied starting tile."""
        self.position: GridPos = start
        self.direction: Direction = (0, 0)
        self.desired_direction: Direction = (0, 0)

    def set_direction(self, direction: Direction, maze: Maze) -> None:
        """Remember desired direction and apply it when possible."""
        self.desired_direction = direction
        self._apply_desired_direction(maze)

    def step(self, maze: Maze) -> None:
        """Advance one tile in the current direction when possible."""
        self._apply_desired_direction(maze)
        next_pos = (self.position[0] + self.direction[0], self.position[1] + self.direction[1])
        if maze.in_bounds(next_pos) and not maze.is_wall(next_pos):
            self.position = next_pos
        elif self.desired_direction != self.direction:
            self._apply_desired_direction(maze)

    def _apply_desired_direction(self, maze: Maze) -> None:
        next_pos = (self.position[0] + self.desired_direction[0], self.position[1] + self.desired_direction[1])
        if maze.in_bounds(next_pos) and not maze.is_wall(next_pos):
            self.direction = self.desired_direction


class Ghost:
    """Randomly moving enemy following simple grid navigation."""

    def __init__(self, start: GridPos) -> None:
        """Spawn a ghost that initially travels in a random direction."""
        self.position: GridPos = start
        self.direction: Direction = random.choice(list(ACTIONS.values()))

    def step(self, maze: Maze) -> None:
        """Choose a valid direction and move one tile."""
        options = self._valid_directions(maze)
        if not options:
            return
        if self.direction not in options or random.random() < 0.2:
            self.direction = random.choice(options)
        next_pos = (self.position[0] + self.direction[0], self.position[1] + self.direction[1])
        if maze.in_bounds(next_pos) and not maze.is_wall(next_pos):
            self.position = next_pos

    def _valid_directions(self, maze: Maze) -> List[Direction]:
        """Enumerate all non-wall directions from the current tile."""
        return [
            direction
            for direction in ACTIONS.values()
            if maze.in_bounds((self.position[0] + direction[0], self.position[1] + direction[1]))
            and not maze.is_wall((self.position[0] + direction[0], self.position[1] + direction[1]))
        ]


class PacmanEnv:
    """Gym-style environment exposing reset, step, and render interfaces."""

    def __init__(self, config: Config, human_mode: bool = False, headless: bool = False) -> None:
        """Instantiate the environment with configurable control and rendering modes."""
        self.config, self.human_mode, self.headless = config, human_mode, headless
        self.maze = Maze(config.maze_layout)
        self.pacman = Pacman(self.maze.pacman_spawn)
        self.ghost_spawn_points = list(self.maze.ghost_spawns)
        self.ghosts = [Ghost(pos) for pos in self.ghost_spawn_points]
        self.max_steps = config.max_steps
        self._step_counter = 0
        self.screen: Optional[pygame.Surface]
        self.surface: Optional[pygame.Surface]
        self.screen = self.surface = None
        self.clock = pygame.time.Clock()
        self._trajectory: List[Tuple[Dict[str, np.ndarray], Optional[int], float]] = []
        self._recording = False
        self._quit_requested = False

    def _pump_events(self) -> None:
        # Always keep OS/window events flowing even in non-human mode
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment to the starting state."""
        self.maze.reset()
        self.pacman = Pacman(self.maze.pacman_spawn)
        self.ghosts = [Ghost(pos) for pos in self.ghost_spawn_points]
        self._step_counter = 0
        state = self._get_state()
        if self._recording:
            self._trajectory.clear()
        return state

    def step(self, action: Optional[int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, int]]:
        """Advance the environment by one tick."""
        if not self.human_mode:
            if action is None or action not in ACTIONS:
                raise ValueError("Non-human mode requires an action from {0,1,2,3}.")
            self.pacman.set_direction(ACTIONS[action], self.maze)

        self.pacman.step(self.maze)
        reward = -0.01
        if self.maze.consume_pellet(self.pacman.position):
            reward += 1.0

        collision = self._check_collision()
        if not collision:
            for ghost in self.ghosts:
                ghost.step(self.maze)
            collision = self._check_collision()

        self._step_counter += 1
        done = collision or not self.maze.pellets.any()
        max_steps_reached = False
        if not done and self.max_steps is not None and self._step_counter >= self.max_steps:
            done = True
            max_steps_reached = True

        if collision:
            reward -= 1.0

        state = self._get_state()
        info = {"pellets_remaining": int(self.maze.pellets.sum())}
        if max_steps_reached:
            info["max_steps_reached"] = 1
        if self._recording:
            self._trajectory.append((state, action, reward))
        return state, reward, done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the current state to the screen or return an RGB array."""
        if not pygame.get_init():
            pygame.init()

        # Pump events every frame to keep the window alive
        self._pump_events()
        if self._quit_requested:
            return None

        size = (self.config.tile_size * self.config.width, self.config.tile_size * self.config.height)

        if mode == "human" and not self.headless:
            if self.screen is None:
                self.screen = pygame.display.set_mode(size)
                pygame.display.set_caption("Minimal Pac-Man")
            surface = self.screen
        else:
            if self.surface is None:
                self.surface = pygame.Surface(size)
            surface = self.surface

        surface.fill(self.config.background_color)
        self._draw_maze(surface)
        self._draw_pacman(surface)
        self._draw_ghosts(surface)

        if mode == "human" and self.screen:
            pygame.display.flip()
            self.clock.tick(self.config.fps)
            return None

        array = pygame.surfarray.array3d(surface)
        return np.transpose(array, (1, 0, 2))

    def close(self) -> None:
        """Release pygame resources."""
        if pygame.get_init():
            pygame.quit()

    def record_trajectory(self, enable: bool = True) -> List[Tuple[Dict[str, np.ndarray], Optional[int], float]]:
        """Enable or disable trajectory capture; returns the data when disabling."""
        if enable:
            self._recording = True
            self._trajectory.clear()
            return []
        self._recording = False
        return list(self._trajectory)

    def set_human_direction(self, direction: Direction) -> None:
        """Update Pac-Man direction when in human mode."""
        if not self.human_mode:
            return
        self.pacman.set_direction(direction, self.maze)

    def _check_collision(self) -> bool:
        """Report whether a ghost occupies Pac-Man's tile."""
        return any(ghost.position == self.pacman.position for ghost in self.ghosts)

    def _get_state(self) -> Dict[str, np.ndarray]:
        """Build a compact representation of the current world state."""
        pacman_pos = np.array(self.pacman.position, dtype=np.int16)
        ghost_positions = np.array([ghost.position for ghost in self.ghosts], dtype=np.int16)
        pellets = self.maze.pellets.copy()
        return {"pacman": pacman_pos, "ghosts": ghost_positions, "pellets": pellets}

    def _draw_maze(self, surface: pygame.Surface) -> None:
        """Render walls and pellets."""
        tile = self.config.tile_size
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                rect = pygame.Rect(x * tile, y * tile, tile, tile)
                if self.maze.is_wall((x, y)):
                    pygame.draw.rect(surface, self.config.wall_color, rect)
                elif self.maze.pellets[y, x]:
                    center = (x * tile + tile // 2, y * tile + tile // 2)
                    pygame.draw.circle(surface, self.config.pellet_color, center, tile // 6)

    def _draw_pacman(self, surface: pygame.Surface) -> None:
        """Render Pac-Man."""
        tile = self.config.tile_size
        center = (self.pacman.position[0] * tile + tile // 2, self.pacman.position[1] * tile + tile // 2)
        pygame.draw.circle(surface, self.config.pacman_color, center, tile // 2 - 2)

    def _draw_ghosts(self, surface: pygame.Surface) -> None:
        """Render all ghosts."""
        tile = self.config.tile_size
        for ghost in self.ghosts:
            center = (ghost.position[0] * tile + tile // 2, ghost.position[1] * tile + tile // 2)
            pygame.draw.circle(surface, self.config.ghost_color, center, tile // 2 - 2)


class Game:
    """Thin wrapper to run the Pac-Man environment in human-controlled mode."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Create a playable game instance with the provided configuration."""

        # Example of a custom maze spec that you would use for simple RL training
        spec_stage1 = MazeSpec(
            width=2,
            height=2,
            include_ghosts=False,
            pellet_mode="single",
            pellet_positions=[(1, 1)],
            include_power_pellets=False,
            surround_walls=True,
        )
        self.env = PacmanEnv(Config(maze_spec=spec_stage1), human_mode=True, headless=False)

        # Example show how we can setup an exact matrix
        # maze = (
        #     "################",
        #     "#P..#......#..G#",
        #     "#.#.#.####.#.###",  # +#
        #     "#.#.#....#.#..##",  # +#
        #     "#.#.####.#.##.##",  # +#
        #     "#.#......#....##",  # +#
        #     "################",
        # )
        # self.env = PacmanEnv(Config(maze_layout=maze), human_mode=True)

        # This runs in default game mode
        self.config = config or Config(maze_size=2)
        self.env = PacmanEnv(self.config, human_mode=True)

    def run(self) -> None:
        """Start the interactive game loop."""
        if not pygame.get_init():
            pygame.init()
        running = True
        self.env.reset()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)
            state, reward, done, _ = self.env.step(action=None)
            self.env.render(mode="human")
            if done:
                pygame.time.wait(1000)
                self.env.reset()
        self.env.close()

    def _handle_key(self, key: int) -> None:
        """Translate keyboard input into Pac-Man movement commands."""
        mapping = {
            pygame.K_UP: ACTIONS[0],
            pygame.K_DOWN: ACTIONS[1],
            pygame.K_LEFT: ACTIONS[2],
            pygame.K_RIGHT: ACTIONS[3],
        }
        if key in mapping:
            self.env.set_human_direction(mapping[key])


if __name__ == "__main__":
    Game().run()
