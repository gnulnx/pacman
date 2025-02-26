# env.py
#
import math
import os
import random
from collections import deque

import cv2  # OpenCV for image processing
import numpy as np
import pygame

from maze import Maze, generate_maze, get_open_cells, safe_spawn_pacman
from pacman import PacMan
from train.settings import (  # noqa
    ACTION_DIM,
    BLACK,
    CLEAR_BOARD,
    COLS,
    DEBUG,
    EAT_GHOST_SCORE,
    FRAME_STACK_SIZE,
    FRUIT_SCORE,
    GHOST_CATCH_SCORE,
    HEADLESS,
    MODE,
    MOVE_TOWARD_PELLOT,
    NOVELTY_BONUS,
    PELLET_SCORE,
    ROWS,
    STEP_PENALTY,
    TILE_SIZE,
    USE_8BIT,
    USE_ABSOLUTE_ACTIONS,
    WALL_COLLISION_PENALTY,
)

# For reproducibility:
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

# For headless mode, set the SDL video driver if desired.
if MODE == "train" and HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# -----------------------------
# Pre-generate a fixed maze layout if required.
# -----------------------------
FIXED_MAZE_LAYOUT = generate_maze(ROWS, COLS)


# -----------------------------
# Pacman Environment
# -----------------------------
class PacmanEnv:
    def __init__(self, fixed_maze=False):
        self.fixed_maze = fixed_maze
        self.screen_width = COLS * TILE_SIZE
        self.screen_height = ROWS * TILE_SIZE
        print(f"Screen size: {self.screen_width}x{self.screen_height}")
        pygame.display.set_caption("Pac-Man RL")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.frame_stack = deque(maxlen=FRAME_STACK_SIZE)  # For frame stacking

        self.reset()

    def reset(self):
        if self.fixed_maze:
            maze_layout = FIXED_MAZE_LAYOUT
        else:
            maze_layout = generate_maze(ROWS, COLS)

        self.maze_obj = Maze(maze_layout)
        open_cells = get_open_cells(maze_layout)
        if not open_cells:
            raise Exception("No open cells in maze!")

        ghosts = []
        # ghost_cells = open_cells[:]  # copy list
        # random.shuffle(ghost_cells)
        # for _ in range(3):
        #     if ghost_cells:
        #         cell = ghost_cells.pop()
        #         ghost_x = cell[1] * TILE_SIZE + TILE_SIZE // 2
        #         ghost_y = cell[0] * TILE_SIZE + TILE_SIZE // 2
        #         ghosts.append(Ghost(ghost_x, ghost_y))
        self.ghosts = ghosts

        pac_cell = random.choice(open_cells)
        pac_x = pac_cell[1] * TILE_SIZE + TILE_SIZE // 2
        pac_y = pac_cell[0] * TILE_SIZE + TILE_SIZE // 2
        pac_x, pac_y = safe_spawn_pacman(maze_layout, ghosts)
        self.pacman = PacMan(pac_x, pac_y)
        self.done = False
        directions = [
            pygame.math.Vector2(0, -1),
            pygame.math.Vector2(0, 1),
            pygame.math.Vector2(-1, 0),
            pygame.math.Vector2(1, 0),
        ]
        self.last_direction = random.choice(directions)
        self.old_tile = (int(pac_y // TILE_SIZE), int(pac_x // TILE_SIZE))

        # Get the initial frame and fill the frame stack
        initial_frame = self._get_frame()
        self.frame_stack.clear()
        for _ in range(FRAME_STACK_SIZE):
            self.frame_stack.append(initial_frame)

        return self._get_stacked_state()

    def distance_to_nearest_pellet(self):
        if not self.maze_obj.pellets:
            return 0.0
        px, py = self.pacman.x, self.pacman.y
        min_dist = float("inf")
        for r, c in self.maze_obj.pellets:
            cx = c * TILE_SIZE + TILE_SIZE // 2
            ry = r * TILE_SIZE + TILE_SIZE // 2
            dist = math.hypot(px - cx, py - ry)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _action_to_direction(self, action, base_direction):
        """
        Convert an action index into a pygame Vector2 direction.
        - If USE_ABSOLUTE_ACTIONS=True, we interpret 0=Up,1=Down,2=Left,3=Right.
        - If USE_ABSOLUTE_ACTIONS=False, we interpret 0=forward,1=left,2=right,3=reverse relative to base_direction.
        """
        if USE_ABSOLUTE_ACTIONS:
            # Absolute directions
            if action == 0:
                return pygame.math.Vector2(0, -1)  # Up
            elif action == 1:
                return pygame.math.Vector2(0, 1)  # Down
            elif action == 2:
                return pygame.math.Vector2(-1, 0)  # Left
            elif action == 3:
                return pygame.math.Vector2(1, 0)  # Right
        else:
            # Relative directions (forward, left, right, reverse)
            if action == 0:
                return base_direction
            elif action == 1:
                return pygame.math.Vector2(-base_direction.y, base_direction.x)
            elif action == 2:
                return pygame.math.Vector2(base_direction.y, -base_direction.x)
            elif action == 3:
                return -base_direction

    def legal_actions(self):
        legal = []
        for action in range(ACTION_DIM):
            candidate = self._action_to_direction(action, self.last_direction)
            new_x = self.pacman.x + candidate.x * self.pacman.speed
            new_y = self.pacman.y + candidate.y * self.pacman.speed
            if not self.pacman.collides_with_wall(new_x, new_y, self.maze_obj):
                legal.append(action)
        return legal

    def step(self, action):
        reward = 0.0
        pygame.event.pump()
        dist_before = self.distance_to_nearest_pellet()
        # legal = self.legal_actions()

        # Convert the chosen action to a candidate direction.
        candidate = self._action_to_direction(action, self.last_direction)
        new_x = self.pacman.x + candidate.x * self.pacman.speed
        new_y = self.pacman.y + candidate.y * self.pacman.speed

        # Wall collision penalty
        if self.pacman.collides_with_wall(new_x, new_y, self.maze_obj):
            reward -= WALL_COLLISION_PENALTY

        # Update last_direction and Pac-Man’s intended_direction with the final candidate.
        self.last_direction = candidate
        self.pacman.intended_direction = candidate

        pre_pellet = len(self.maze_obj.pellets)
        pre_fruits = len(self.maze_obj.fruits)

        self.pacman.update(self.maze_obj)
        for ghost in self.ghosts:
            ghost.vulnerable = pygame.time.get_ticks() < self.pacman.powerpellet_end
            ghost.update(self.maze_obj, self.pacman)

        for ghost in self.ghosts:
            distance = math.hypot(self.pacman.x - ghost.x, self.pacman.y - ghost.y)
            if distance < self.pacman.radius + ghost.radius:
                if pygame.time.get_ticks() < self.pacman.powerpellet_end:
                    # self.pacman.score += 10
                    reward += EAT_GHOST_SCORE
                    ghost_cell = random.choice(get_open_cells(self.maze_obj.layout))
                    ghost.x = ghost_cell[1] * TILE_SIZE + TILE_SIZE // 2
                    ghost.y = ghost_cell[0] * TILE_SIZE + TILE_SIZE // 2
                    ghost.vulnerable = False
                else:
                    self.done = True
                    reward -= GHOST_CATCH_SCORE
                    break

        post_pellet = len(self.maze_obj.pellets)
        fruit_post = len(self.maze_obj.fruits)
        pellets_collected = pre_pellet - post_pellet
        reward += pellets_collected * PELLET_SCORE
        fruits_collected = pre_fruits - fruit_post
        reward += fruits_collected * FRUIT_SCORE

        # new_tile = (int(self.pacman.y // TILE_SIZE), int(self.pacman.x // TILE_SIZE))
        # if new_tile == self.old_tile:
        #     reward -= 0.1
        # self.old_tile = new_tile

        if (not self.maze_obj.pellets) and (not self.maze_obj.fruits) and (not self.maze_obj.power_pellets):
            reward += CLEAR_BOARD  # Bonus for clearing the board.
            self.done = True

        # Distance based learning. off whiel we try frame stacking...
        dist_after = self.distance_to_nearest_pellet()

        if dist_after < dist_before:
            reward += MOVE_TOWARD_PELLOT

        # Novelty reward: bonus for visiting a new tile
        tile = (int(self.pacman.y // TILE_SIZE), int(self.pacman.x // TILE_SIZE))
        if not hasattr(self, "visited_tiles"):
            self.visited_tiles = set()
        if tile not in self.visited_tiles:
            reward += NOVELTY_BONUS  # small bonus for novelty
            self.visited_tiles.add(tile)

        # # Step penalty: small negative reward for each step taken
        reward -= STEP_PENALTY

        if HEADLESS:
            self.draw_offscreen()
        else:
            self.render()

        new_frame = self._get_frame()
        self.frame_stack.append(new_frame)
        next_state = self._get_stacked_state()
        return next_state, reward, self.done, {}

    def _get_frame(self):
        """Capture the current screen as a grayscale image."""
        image = pygame.surfarray.array3d(self.screen)
        image = np.transpose(image, (1, 0, 2))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (84, 84))

        if USE_8BIT:
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32) / 255.0

        return image  # shape: (84, 84)

    def _get_stacked_state(self):
        """Stack the frames along the channel dimension."""
        # Convert the deque to a numpy array with shape (FRAME_STACK_SIZE, 84, 84)
        stacked_state = np.array(self.frame_stack)
        # Optionally add a batch dimension if needed later: (1, FRAME_STACK_SIZE, 84, 84)
        return stacked_state

    def draw_offscreen(self):
        self.screen.fill(BLACK)
        self.maze_obj.draw(self.screen)
        self.pacman.draw(self.screen)

    def render(self):
        self.screen.fill(BLACK)
        self.maze_obj.draw(self.screen)
        self.pacman.draw(self.screen)
        for ghost in self.ghosts:
            ghost.draw(self.screen)
        pygame.display.flip()

    def get_state(self):
        image = pygame.surfarray.array3d(self.screen)
        image = np.transpose(image, (1, 0, 2))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (84, 84))
        if USE_8BIT:
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32) / 255.0

        if DEBUG:
            cv2.imshow("State", image)
            input("Check image state. Press Enter to continue...")

        return np.expand_dims(image, axis=0)

    def close(self):
        pygame.quit()
