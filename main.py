import pygame
import random
import sys

# ---------------------------------------
# 1) Maze / World Generation Parameters
# ---------------------------------------
TILE_SIZE = 40   # Wider corridors
ROWS = 21        # Prefer odd numbers
COLS = 31        # Prefer odd numbers

# ---------------------------------------
# 2) Color Definitions (Classic Pac-Man Style)
# ---------------------------------------
BLACK = (0, 0, 0)
WALL_BLUE = (33, 33, 222)   # Dark-ish wall color
PACMAN_YELLOW = (255, 255, 0)
PELLET_ORANGE = (255, 153, 0)
FRUIT_RED = (255, 50, 50)
WHITE = (255, 255, 255)

FPS = 60
PELLET_SCORE = 10
FRUIT_SCORE = 50
NUM_FRUITS = 5

# ---------------------------------------
# 3) Maze Generation Functions
# ---------------------------------------

def generate_maze(rows, cols):
    """
    Generate a maze using DFS carving, then braid it to eliminate dead ends.
    The maze is represented as a 2D list of '1' (wall) and '0' (open).
    """
    # Create grid full of walls.
    maze = [['1' for _ in range(cols)] for _ in range(rows)]
    
    # Start at (1,1)
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
                    # Carve passage: open the wall between (r,c) and (nr,nc)
                    wall_r = r + dr // 2
                    wall_c = c + dc // 2
                    maze[wall_r][wall_c] = '0'
                    maze[nr][nc] = '0'
                    stack.append((nr, nc))
                    carved = True
                    break
        if not carved:
            stack.pop()
    
    # Braid the maze: eliminate dead ends by opening extra connections.
    braid_maze(maze)
    return maze

def braid_maze(maze):
    """
    For every open cell that is a dead end (only 1 open neighbor),
    open an extra wall (if possible) to create a loop.
    This process is repeated until no dead ends remain.
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
                if maze[r][c] == '0':
                    if open_neighbors(r, c) == 1:  # Dead end detected
                        # Look for wall neighbors we can open
                        candidates = []
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if maze[nr][nc] == '1':
                                # Avoid carving into the boundary
                                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1:
                                    candidates.append((nr, nc))
                        if candidates:
                            nr, nc = random.choice(candidates)
                            maze[nr][nc] = '0'
                            changed = True

def get_open_cells(maze):
    """Return list of (row, col) for each open cell in the maze."""
    open_cells = []
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == '0':
                open_cells.append((r, c))
    return open_cells

# ---------------------------------------
# 4) Maze & Pellets Class
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
# 5) Pac-Man Class
# ---------------------------------------
class PacMan:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 2
        # Current moving direction (used for movement)
        self.direction = pygame.math.Vector2(0, 0)
        # Intended direction from key input (for turning)
        self.intended_direction = pygame.math.Vector2(0, 0)
        self.radius = TILE_SIZE // 2
        self.score = 0

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
        # If a key is pressed, update the intended direction.
        if new_dir.length_squared() != 0:
            self.intended_direction = new_dir

    def update(self, maze):
        # Tolerance in pixels for snapping to corridor center
        tolerance = 5

        # Try to turn if intended direction differs from current movement.
        if self.intended_direction != self.direction:
            # Turning horizontally: intended_direction.x != 0.
            if self.intended_direction.x != 0:
                # Find the vertical center of the current cell.
                center_y = int(self.y // TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2
                if abs(self.y - center_y) <= tolerance:
                    # Snap to center and attempt turn.
                    self.y = center_y
                    if not self.collides_with_wall(self.x + self.intended_direction.x * self.speed, self.y, maze):
                        self.direction = self.intended_direction
            # Turning vertically: intended_direction.y != 0.
            elif self.intended_direction.y != 0:
                # Find the horizontal center of the current cell.
                center_x = int(self.x // TILE_SIZE) * TILE_SIZE + TILE_SIZE / 2
                if abs(self.x - center_x) <= tolerance:
                    self.x = center_x
                    if not self.collides_with_wall(self.x, self.y + self.intended_direction.y * self.speed, maze):
                        self.direction = self.intended_direction
            # If currently stopped, try to adopt the intended direction if possible.
            if self.direction == pygame.math.Vector2(0, 0):
                if not self.collides_with_wall(self.x + self.intended_direction.x * self.speed, 
                                               self.y + self.intended_direction.y * self.speed, maze):
                    self.direction = self.intended_direction

        # Move along x and y separately.
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
# 6) Main Game Loop
# ---------------------------------------
def main():
    pygame.init()

    # Generate a maze with no dead ends.
    maze_layout = generate_maze(ROWS, COLS)
    screen_width = COLS * TILE_SIZE
    screen_height = ROWS * TILE_SIZE

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man: Random Braided Maze")
    clock = pygame.time.Clock()

    maze_obj = Maze(maze_layout)
    
    open_cells = get_open_cells(maze_layout)
    if not open_cells:
        print("Error: No open cells found. Maze generation failed.")
        pygame.quit()
        sys.exit()

    # Start Pac-Man at a random open cell.
    start_r, start_c = random.choice(open_cells)
    start_x = start_c * TILE_SIZE + TILE_SIZE // 2
    start_y = start_r * TILE_SIZE + TILE_SIZE // 2
    pacman = PacMan(start_x, start_y)

    font = pygame.font.SysFont(None, 36)
    running = True

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pacman.handle_keys()
        pacman.update(maze_obj)

        screen.fill(BLACK)
        maze_obj.draw(screen)
        pacman.draw(screen)

        # Draw score
        score_text = font.render(f"Score: {pacman.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
