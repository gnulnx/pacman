# pacman.py
from settings import TILE_SIZE, PACMAN_YELLOW, BLACK, PELLET_SCORE, FRUIT_SCORE
import pygame
import math

class PacMan:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 2
        self.direction = pygame.math.Vector2(0, 0)
        self.intended_direction = pygame.math.Vector2(0, 0)
        self.radius = TILE_SIZE // 2
        self.score = 0
        self.lives = 3

        # For mouth animation:
        self.mouth_open = True
        self.mouth_timer = pygame.time.get_ticks()
        self.mouth_interval = 200  # milliseconds between state toggles

    def handle_keys(self):
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
        tolerance = 5  # Allowable misalignment (in pixels) for turning
        # Looser turning logic as before:
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
                if not self.collides_with_wall(self.x + self.intended_direction.x * self.speed,
                                               self.y + self.intended_direction.y * self.speed, maze):
                    self.direction = self.intended_direction

        new_x = self.x + self.direction.x * self.speed
        new_y = self.y + self.direction.y * self.speed
        if not self.collides_with_wall(new_x, self.y, maze):
            self.x = new_x
        if not self.collides_with_wall(self.x, new_y, maze):
            self.y = new_y

        self.check_for_collection(maze)

        # Update mouth animation (toggle state every mouth_interval ms)
        current_time = pygame.time.get_ticks()
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
                    if maze.layout[r][c] == '1':
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

    def draw(self, surface):
        # If moving and mouth is open, draw Pac-Man with an open mouth
        if self.mouth_open and (self.direction.x != 0 or self.direction.y != 0):
            angle = math.atan2(self.direction.y, self.direction.x)
            # Draw full circle first
            pygame.draw.circle(surface, PACMAN_YELLOW, (int(self.x), int(self.y)), self.radius)
            # Set mouth opening (half-angle of the wedge)
            open_angle = math.radians(30)
            # Compute left and right boundary points for the mouth wedge
            left_angle = angle + open_angle
            right_angle = angle - open_angle
            left_point = (int(self.x + self.radius * math.cos(left_angle)),
                          int(self.y + self.radius * math.sin(left_angle)))
            right_point = (int(self.x + self.radius * math.cos(right_angle)),
                           int(self.y + self.radius * math.sin(right_angle)))
            # Draw a black triangle to simulate an open mouth
            pygame.draw.polygon(surface, BLACK, [(int(self.x), int(self.y)), left_point, right_point])
        else:
            # Closed mouth: just a full circle
            pygame.draw.circle(surface, PACMAN_YELLOW, (int(self.x), int(self.y)), self.radius)
