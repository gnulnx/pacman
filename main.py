import pygame
import random
import sys
import math

# ---------------------------------------
# 1) Game & Maze Parameters
# ---------------------------------------
TILE_SIZE = 40           # Wider corridors
ROWS = 21                # Prefer odd numbers
COLS = 31                # Prefer odd numbers
FPS = 60

# ---------------------------------------
# 2) Color Definitions (Classic Pac-Man)
# ---------------------------------------
BLACK = (0, 0, 0)
WALL_BLUE = (33, 33, 222)
PACMAN_YELLOW = (255, 255, 0)
PELLET_ORANGE = (255, 153, 0)
FRUIT_RED = (255, 50, 50)
WHITE = (255, 255, 255)

# Scoring
PELLET_SCORE = 10
FRUIT_SCORE = 50
NUM_FRUITS = 5

# Ghost settings
NUM_GHOSTS = 3

# ---------------------------------------
# 3) Maze Generation & Utility Functions
# ---------------------------------------
def generate_maze(rows, cols):
    """
    Generate a maze using DFS carving, then braid it to eliminate dead ends.
    The maze is a 2D list of '1' (wall) and '0' (open).
    """
    # Initialize grid full of walls.
    maze = [['1' for _ in range(cols)] for _ in range(rows)]
    
    # Start at an odd cell (1,1)
    start_r, start_c = 1, 1
    maze[start_r][start_c] = '0'
    stack = [(start_r, start_c)]
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    while stack:
        r, c = stack[-1]
        random.shuffle(directions)
        carved = False
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows - 1 and 1 <= nc < cols - 1:
                if maze[nr][nc] == '1':
                    # Carve the wall between (r,c) and (nr,nc)
                    wall_r = r + dr // 2
                    wall_c = c + dc // 2
                    maze[wall_r][wall_c] = '0'
                    maze[nr][nc] = '0'
                    stack.append((nr, nc))
                    carved = True
                    break
        if not carved:
            stack.pop()
    
    braid_maze(maze)
    return maze

def braid_maze(maze):
    """
    For every open cell that is a dead end (only 1 open neighbor),
    open an extra wall (if possible) to create a loop.
    """
    rows = len(maze)
    cols = len(maze[0])
    
    def open_neighbors(r, c):
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if maze[r + dr][c + dc] == '0':
                count += 1
        return count

    changed = True
    while changed:
        changed = False
        # Loop over interior cells only.
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if maze[r][c] == '0' and open_neighbors(r, c) == 1:
                    # Open an adjacent wall to remove the dead end.
                    candidates = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if maze[nr][nc] == '1' and 1 <= nr < rows - 1 and 1 <= nc < cols - 1:
                            candidates.append((nr, nc))
                    if candidates:
                        nr, nc = random.choice(candidates)
                        maze[nr][nc] = '0'
                        changed = True

def get_open_cells(maze_layout):
    """Return a list of (row, col) for each open cell in the maze layout."""
    open_cells = []
    for r in range(len(maze_layout)):
        for c in range(len(maze_layout[0])):
            if maze_layout[r][c] == '0':
                open_cells.append((r, c))
    return open_cells

def safe_spawn_pacman(maze_layout, ghosts, min_distance=80):
    """
    Choose an open cell (converted to pixel center) that is at least
    min_distance away from every ghost. If none qualify, return a random open cell.
    """
    open_cells = get_open_cells(maze_layout)
    safe_positions = []
    for cell in open_cells:
        x = cell[1] * TILE_SIZE + TILE_SIZE // 2
        y = cell[0] * TILE_SIZE + TILE_SIZE // 2
        if all(math.hypot(ghost.x - x, ghost.y - y) >= min_distance for ghost in ghosts):
            safe_positions.append((x, y))
    if safe_positions:
        return random.choice(safe_positions)
    else:
        cell = random.choice(open_cells)
        return cell[1] * TILE_SIZE + TILE_SIZE // 2, cell[0] * TILE_SIZE + TILE_SIZE // 2

# ---------------------------------------
# 4) Maze Class (Walls, Pellets, Fruit)
# ---------------------------------------
class Maze:
    def __init__(self, layout):
        self.layout = layout
        self.rows = len(layout)
        self.cols = len(layout[0])
        # Place a pellet in every open cell.
        self.pellets = {(r, c) for r in range(self.rows) for c in range(self.cols) if layout[r][c] == '0'}
        # Place fruit in a few random open cells.
        open_cells = get_open_cells(layout)
        random.shuffle(open_cells)
        self.fruits = set(open_cells[:min(NUM_FRUITS, len(open_cells))])
    
    def draw(self, surface):
        # Draw walls
        for r in range(self.rows):
            for c in range(self.cols):
                if self.layout[r][c] == '1':
                    pygame.draw.rect(surface, WALL_BLUE, (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        # Draw pellets (small circles)
        for (r, c) in self.pellets:
            cx = c * TILE_SIZE + TILE_SIZE // 2
            cy = r * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.circle(surface, PELLET_ORANGE, (cx, cy), 4)
        # Draw fruit (small squares)
        for (r, c) in self.fruits:
            x = c * TILE_SIZE + TILE_SIZE // 4
            y = r * TILE_SIZE + TILE_SIZE // 4
            size = TILE_SIZE // 2
            pygame.draw.rect(surface, FRUIT_RED, (x, y, size, size))

# ---------------------------------------
# 5) Pac-Man Class (with Looser Turning & Lives)
# ---------------------------------------
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
        tolerance = 5  # Allowable misalignment (in pixels) to permit turning
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
        pygame.draw.circle(surface, PACMAN_YELLOW, (int(self.x), int(self.y)), self.radius)

# ---------------------------------------
# 6) Ghost Class (Simple Chasing Behavior)
# ---------------------------------------
class Ghost:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 2  # Adjust if needed
        self.direction = pygame.math.Vector2(0, 0)
        self.radius = TILE_SIZE // 2

    def update(self, maze, pacman):
        # Determine possible movement directions from current position.
        possible_directions = []
        for d in [pygame.math.Vector2(1, 0), pygame.math.Vector2(-1, 0),
                  pygame.math.Vector2(0, 1), pygame.math.Vector2(0, -1)]:
            new_x = self.x + d.x * self.speed
            new_y = self.y + d.y * self.speed
            if not self.collides_with_wall(new_x, new_y, maze):
                possible_directions.append(d)
        
        # Compute target direction based on Pac-Man's relative position.
        dx = pacman.x - self.x
        dy = pacman.y - self.y
        if abs(dx) > abs(dy):
            target_dir = pygame.math.Vector2(1, 0) if dx > 0 else pygame.math.Vector2(-1, 0)
        else:
            target_dir = pygame.math.Vector2(0, 1) if dy > 0 else pygame.math.Vector2(0, -1)
        
        # Choose direction:
        # If the target direction is available and not the reverse of the current direction, choose it.
        if (target_dir in possible_directions and 
            (self.direction == pygame.math.Vector2(0, 0) or target_dir != -self.direction)):
            self.direction = target_dir
        else:
            # If current direction is still valid, sometimes continue; 
            # otherwise, pick a new random valid direction (avoiding reversal if possible).
            if self.direction in possible_directions:
                if random.random() < 0.15:  # 15% chance to change direction at an intersection
                    valid = [d for d in possible_directions if d != -self.direction]
                    if valid:
                        self.direction = random.choice(valid)
            else:
                valid = [d for d in possible_directions if d != -self.direction]
                if valid:
                    self.direction = random.choice(valid)
                elif possible_directions:
                    self.direction = random.choice(possible_directions)
                # If no movement is possible, the direction remains unchanged.

        # Move the ghost along its chosen direction.
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
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), self.radius)


# ---------------------------------------
# 7) Main Game Loop
# ---------------------------------------
def main():
    pygame.init()
    maze_layout = generate_maze(ROWS, COLS)
    screen_width = COLS * TILE_SIZE
    screen_height = ROWS * TILE_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man with Ghosts")
    clock = pygame.time.Clock()

    maze_obj = Maze(maze_layout)
    open_cells = get_open_cells(maze_layout)
    if not open_cells:
        print("Error: No open cells found.")
        pygame.quit()
        sys.exit()

    # Spawn ghosts at random open cells.
    ghosts = []
    ghost_cells = open_cells[:]  # copy list
    random.shuffle(ghost_cells)
    for _ in range(NUM_GHOSTS):
        if ghost_cells:
            cell = ghost_cells.pop()
            ghost_x = cell[1] * TILE_SIZE + TILE_SIZE // 2
            ghost_y = cell[0] * TILE_SIZE + TILE_SIZE // 2
            ghosts.append(Ghost(ghost_x, ghost_y))

    # Spawn Pac-Man at a safe location (away from ghosts).
    pac_x, pac_y = safe_spawn_pacman(maze_layout, ghosts)
    pacman = PacMan(pac_x, pac_y)

    font = pygame.font.SysFont(None, 36)
    running = True

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pacman.handle_keys()
        pacman.update(maze_obj)
        for ghost in ghosts:
            ghost.update(maze_obj, pacman)

        # Check for collisions between Pac-Man and ghosts.
        for ghost in ghosts:
            distance = math.hypot(pacman.x - ghost.x, pacman.y - ghost.y)
            if distance < pacman.radius + ghost.radius:
                pacman.lives -= 1
                if pacman.lives <= 0:
                    print("Game Over!")
                    pygame.quit()
                    sys.exit()
                else:
                    # Respawn Pac-Man safely.
                    new_x, new_y = safe_spawn_pacman(maze_layout, ghosts)
                    pacman.x, pacman.y = new_x, new_y
                    pacman.direction = pygame.math.Vector2(0, 0)
                    pacman.intended_direction = pygame.math.Vector2(0, 0)
                    break

        screen.fill(BLACK)
        maze_obj.draw(screen)
        pacman.draw(screen)
        for ghost in ghosts:
            ghost.draw(screen)
        # Display score and lives.
        score_text = font.render(f"Score: {pacman.score}", True, WHITE)
        lives_text = font.render(f"Lives: {pacman.lives}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (10, 40))
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
