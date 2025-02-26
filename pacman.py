import math
import random
from collections import deque

import pygame

from settings import (
    BLACK,
    FRUIT_SCORE,
    PACMAN_YELLOW,
    PELLET_SCORE,
    POWER_PELLET_SCORE,
    TILE_SIZE,
)


class PacMan:
    def __init__(self, x, y, auto_play=False, use_dqn=False):
        self.x = x
        self.y = y
        self.speed = 2
        self.direction = pygame.math.Vector2(0, 0)
        self.intended_direction = pygame.math.Vector2(0, 0)
        self.radius = TILE_SIZE // 2
        self.score = 0
        self.lives = 3
        self.mouth_open = True
        self.mouth_timer = pygame.time.get_ticks()
        self.mouth_interval = 200  # milliseconds between toggles
        self.powerup_end = 0  # For fruit power-up (double speed)
        self.powerpellet_end = 0  # For power pellet effect (ghost vulnerability)
        self.auto_play = auto_play
        self.use_dqn = use_dqn  # If True, a neural network (DQN) controls movement externally.

    def handle_keys(self):
        if not self.auto_play:
            keys = pygame.key.get_pressed()
            new_dir = pygame.math.Vector2(0, 0)
            if keys[pygame.K_LEFT]:
                new_dir = pygame.math.Vector2(-1, 0)
            elif keys[pygame.K_RIGHT]:
                new_dir = pygame.math.Vector2(1, 0)
            elif keys[pygame.K_UP]:
                new_dir = pygame.math.Vector2(0, -1)
            elif keys[pygame.K_DOWN]:
                new_dir = pygame.math.Vector2(0, 1)
            if new_dir.length_squared() != 0:
                self.intended_direction = new_dir

    def update(self, maze):
        current_time = pygame.time.get_ticks()
        # If auto_play is enabled and we're NOT using the DQN controller,
        # then use the built-in (BFS) navigation.
        if self.auto_play and not self.use_dqn:
            self.auto_navigate(maze)
        tolerance = 5  # allowable misalignment for turning
        if self.intended_direction != self.direction:
            if self.intended_direction.x != 0:
                center_y = int(self.y // TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2
                if abs(self.y - center_y) <= tolerance:
                    self.y = center_y
                    if not self.collides_with_wall(self.x + self.intended_direction.x * self.speed, self.y, maze):
                        self.direction = self.intended_direction
            elif self.intended_direction.y != 0:
                center_x = int(self.x // TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2
                if abs(self.x - center_x) <= tolerance:
                    self.x = center_x
                    if not self.collides_with_wall(self.x, self.y + self.intended_direction.y * self.speed, maze):
                        self.direction = self.intended_direction
            if self.direction == pygame.math.Vector2(0, 0):
                if not self.collides_with_wall(
                    self.x + self.intended_direction.x * self.speed,
                    self.y + self.intended_direction.y * self.speed,
                    maze,
                ):
                    self.direction = self.intended_direction

        effective_speed = self.speed * 2 if current_time < self.powerup_end else self.speed
        new_x = self.x + self.direction.x * effective_speed
        new_y = self.y + self.direction.y * effective_speed
        if not self.collides_with_wall(new_x, self.y, maze):
            self.x = new_x
        if not self.collides_with_wall(self.x, new_y, maze):
            self.y = new_y

        self.check_for_collection(maze)

        if current_time - self.mouth_timer > self.mouth_interval:
            self.mouth_open = not self.mouth_open
            self.mouth_timer = current_time

    def collides_with_wall(self, x, y, maze):
        pac_rect = pygame.Rect(x - self.radius, y - self.radius, self.radius * 2, self.radius * 2)
        left_tile = pac_rect.left // TILE_SIZE
        right_tile = pac_rect.right // TILE_SIZE
        top_tile = pac_rect.top // TILE_SIZE
        bottom_tile = pac_rect.bottom // TILE_SIZE
        for r in range(top_tile, bottom_tile + 1):
            for c in range(left_tile, right_tile + 1):
                if 0 <= r < maze.rows and 0 <= c < maze.cols:
                    if maze.layout[r][c] == "1":
                        wall_rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                        if pac_rect.colliderect(wall_rect):
                            return True
        return False

    def check_for_collection(self, maze):
        row = int(self.y // TILE_SIZE)
        col = int(self.x // TILE_SIZE)
        if (row, col) in maze.pellets:
            maze.pellets.remove((row, col))
            self.score += PELLET_SCORE
        if (row, col) in maze.fruits:
            maze.fruits.remove((row, col))
            self.score += FRUIT_SCORE
            current_time = pygame.time.get_ticks()
            self.powerup_end = max(self.powerup_end, current_time) + 5000
        if (row, col) in maze.power_pellets:
            maze.power_pellets.remove((row, col))
            self.score += POWER_PELLET_SCORE
            current_time = pygame.time.get_ticks()
            self.powerpellet_end = current_time + 4000

    def draw(self, surface):
        if self.mouth_open and (self.direction.x != 0 or self.direction.y != 0):
            angle = math.atan2(self.direction.y, self.direction.x)
            pygame.draw.circle(surface, PACMAN_YELLOW, (int(self.x), int(self.y)), self.radius)
            mouth_angle = 35
            open_angle = math.radians(mouth_angle)
            left_angle = angle + open_angle
            right_angle = angle - open_angle
            wedge_radius = self.radius + 3
            left_point = (
                int(self.x + wedge_radius * math.cos(left_angle)),
                int(self.y + wedge_radius * math.sin(left_angle)),
            )
            right_point = (
                int(self.x + wedge_radius * math.cos(right_angle)),
                int(self.y + wedge_radius * math.sin(right_angle)),
            )
            pygame.draw.polygon(surface, BLACK, [(int(self.x), int(self.y)), left_point, right_point])
        else:
            pygame.draw.circle(surface, PACMAN_YELLOW, (int(self.x), int(self.y)), self.radius)

    # def auto_navigate(self, maze):
    #     # Original BFS-based auto-navigation (used only when not using DQN control).
    #     row = int(self.y // TILE_SIZE)
    #     col = int(self.x // TILE_SIZE)
    #     center_x = col * TILE_SIZE + TILE_SIZE / 2
    #     center_y = row * TILE_SIZE + TILE_SIZE / 2
    #     if abs(self.x - center_x) < 1 and abs(self.y - center_y) < 1:
    #         path = find_path_to_nearest_item(maze, row, col)
    #         if path and len(path) > 1:
    #             next_tile = path[1]
    #             dr = next_tile[0] - row
    #             dc = next_tile[1] - col
    #             self.intended_direction = pygame.math.Vector2(dc, dr)

    def auto_navigate(self, maze):
        """
        Use BFS to determine the next move.
        This version computes a BFS path from the current grid cell to the nearest target.
        If a path exists, it immediately sets the intended direction toward the next cell.
        If no path exists, it chooses a random legal direction.
        """
        row = int(self.y // TILE_SIZE)
        col = int(self.x // TILE_SIZE)
        path = find_path_to_nearest_item(maze, row, col)
        if path and len(path) > 1:
            next_tile = path[1]
            # Compute the direction based on grid differences.
            dr = next_tile[0] - row
            dc = next_tile[1] - col
            desired_direction = pygame.math.Vector2(dc, dr)
            if desired_direction.length() > 0:
                self.intended_direction = desired_direction.normalize()
            else:
                # Fallback in case desired_direction is zero.
                self.intended_direction = pygame.math.Vector2(0, 0)
        else:
            # If BFS doesn't yield a path, choose a random legal direction.
            legal = []
            candidate_dirs = [
                pygame.math.Vector2(0, -1),  # Up
                pygame.math.Vector2(0, 1),  # Down
                pygame.math.Vector2(-1, 0),  # Left
                pygame.math.Vector2(1, 0),  # Right
            ]
            for vec in candidate_dirs:
                new_x = self.x + vec.x * self.speed
                new_y = self.y + vec.y * self.speed
                if not self.collides_with_wall(new_x, new_y, maze):
                    legal.append(vec)
            if legal:
                self.intended_direction = random.choice(legal)
            else:
                self.intended_direction = pygame.math.Vector2(0, 0)


def find_path_to_nearest_item(maze, start_r, start_c):
    targets = set(maze.pellets).union(set(maze.fruits)).union(set(maze.power_pellets))
    if not targets:
        return None
    rows = maze.rows
    cols = maze.cols
    queue = deque()
    queue.append((start_r, start_c))
    visited = {(start_r, start_c)}
    came_from = {}
    while queue:
        r, c = queue.popleft()
        if (r, c) in targets:
            path = []
            current = (r, c)
            while current != (start_r, start_c):
                path.append(current)
                current = came_from[current]
            path.append((start_r, start_c))
            path.reverse()
            return path
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if maze.layout[nr][nc] == "0":
                    visited.add((nr, nc))
                    came_from[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
    return None
