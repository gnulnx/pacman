from settings import TILE_SIZE, WALL_BLUE, PELLET_ORANGE, FRUIT_RED, NUM_FRUITS, NUM_POWER_PELLETS, POWER_PELLET_COLOR
import random
import pygame
import math

def generate_maze(rows, cols):
    maze = [['1' for _ in range(cols)] for _ in range(rows)]
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
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if maze[r][c] == '0' and open_neighbors(r, c) == 1:
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
    open_cells = []
    for r in range(len(maze_layout)):
        for c in range(len(maze_layout[0])):
            if maze_layout[r][c] == '0':
                open_cells.append((r, c))
    return open_cells

def safe_spawn_pacman(maze_layout, ghosts, min_distance=80):
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
#  Maze Class (Walls, Pellets, Fruit, Power Pellets)
# ---------------------------------------
class Maze:
    def __init__(self, layout):
        self.layout = layout
        self.rows = len(layout)
        self.cols = len(layout[0])
        open_cells = get_open_cells(layout)
        random.shuffle(open_cells)
        # Choose power pellets and fruits first.
        self.power_pellets = set(open_cells[:min(NUM_POWER_PELLETS, len(open_cells))])
        self.fruits = set(open_cells[min(NUM_POWER_PELLETS, len(open_cells)):
                                    min(NUM_POWER_PELLETS + NUM_FRUITS, len(open_cells))])
        self.pellets = {(r, c) for r in range(self.rows) for c in range(self.cols) if layout[r][c] == '0'}
        # Remove power pellet and fruit cells from pellets.
        self.pellets = self.pellets - self.power_pellets - self.fruits
    
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
        # Draw power pellets (now slightly larger)
        for (r, c) in self.power_pellets:
            cx = c * TILE_SIZE + TILE_SIZE // 2
            cy = r * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.circle(surface, POWER_PELLET_COLOR, (cx, cy), 10)
