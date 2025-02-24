from settings import TILE_SIZE, WHITE, GHOST_COLOR
import pygame
import random

class Ghost:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 2
        self.direction = pygame.math.Vector2(0, 0)
        self.radius = TILE_SIZE // 2
        self.vulnerable = False  # Flag to indicate if ghost can be eaten

    def update(self, maze, pacman):
        possible_directions = []
        for d in [pygame.math.Vector2(1, 0), pygame.math.Vector2(-1, 0),
                  pygame.math.Vector2(0, 1), pygame.math.Vector2(0, -1)]:
            new_x = self.x + d.x * self.speed
            new_y = self.y + d.y * self.speed
            if not self.collides_with_wall(new_x, new_y, maze):
                possible_directions.append(d)
        
        dx = pacman.x - self.x
        dy = pacman.y - self.y
        if abs(dx) > abs(dy):
            target_dir = pygame.math.Vector2(1, 0) if dx > 0 else pygame.math.Vector2(-1, 0)
        else:
            target_dir = pygame.math.Vector2(0, 1) if dy > 0 else pygame.math.Vector2(0, -1)
        # If vulnerable, reverse the target direction to run away.
        if self.vulnerable:
            target_dir = -target_dir
        
        if (target_dir in possible_directions and 
            (self.direction == pygame.math.Vector2(0, 0) or target_dir != -self.direction)):
            self.direction = target_dir
        else:
            if self.direction in possible_directions:
                if random.random() < 0.15:
                    valid = [d for d in possible_directions if d != -self.direction]
                    if valid:
                        self.direction = random.choice(valid)
            else:
                valid = [d for d in possible_directions if d != -self.direction]
                if valid:
                    self.direction = random.choice(valid)
                elif possible_directions:
                    self.direction = random.choice(possible_directions)

        new_x = self.x + self.direction.x * self.speed
        new_y = self.y + self.direction.y * self.speed
        if not self.collides_with_wall(new_x, self.y, maze):
            self.x = new_x
        if not self.collides_with_wall(self.x, new_y, maze):
            self.y = new_y

    def collides_with_wall(self, x, y, maze):
        ghost_rect = pygame.Rect(x - self.radius, y - self.radius, self.radius * 2, self.radius * 2)
        left_tile = ghost_rect.left // TILE_SIZE
        right_tile = ghost_rect.right // TILE_SIZE
        top_tile = ghost_rect.top // TILE_SIZE
        bottom_tile = ghost_rect.bottom // TILE_SIZE
        for r in range(top_tile, bottom_tile + 1):
            for c in range(left_tile, right_tile + 1):
                if 0 <= r < maze.rows and 0 <= c < maze.cols:
                    if maze.layout[r][c] == '1':
                        wall_rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                        if ghost_rect.colliderect(wall_rect):
                            return True
        return False

    def draw(self, surface):
        color = WHITE if self.vulnerable else GHOST_COLOR
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)
