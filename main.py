import pygame
import random
import sys
import math
from pacman import  PacMan
from ghost import Ghost
from maze import Maze, generate_maze, get_open_cells, safe_spawn_pacman
from settings import TILE_SIZE, FPS, BLACK, NUM_GHOSTS, ROWS, COLS, WHITE


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
