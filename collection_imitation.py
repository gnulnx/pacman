import os
import pickle
import random
import sys

import pygame

from maze import Maze, generate_maze, get_open_cells, safe_spawn_pacman
from pacman import PacMan
from train.settings import (
    BLACK,
    COLS,
    DEMO_DATA_PATH,
    FIXED_MAZE,
    FPS,
    IMITATION_MODE,
    MODE,
    RECORD_DEMOS,
    ROWS,
    TILE_SIZE,
)

# Set these flags to collect demonstrations.
MODE = "imitate"  # Should be "imitate" for demonstration collection.
IMITATION_MODE = True  # Enable imitation mode.
RECORD_DEMOS = True  # Enable recording demonstrations.


def load_existing_demos():
    """Load existing demonstration data if available."""
    if os.path.exists(DEMO_DATA_PATH):
        with open(DEMO_DATA_PATH, "rb") as f:
            demos = pickle.load(f)
        print(f"Loaded {len(demos)} existing demonstration episodes.")
    else:
        demos = []
    return demos


def save_episode_demo(episode_demo, episode):
    """Save the demonstration for a single episode to its own file."""
    episode_file = f"demonstration_episode_{episode}.pkl"
    with open(episode_file, "wb") as f:
        pickle.dump(episode_demo, f)
    print(f"Saved demonstration data for episode {episode} to {episode_file}.")


def save_combined_demos(demos):
    """Save the combined demonstration data to DEMO_DATA_PATH."""
    with open(DEMO_DATA_PATH, "wb") as f:
        pickle.dump(demos, f)
    print(f"Saved combined demonstration data to {DEMO_DATA_PATH}.")


def collect_demonstrations(num_episodes=25):
    demos = load_existing_demos()

    pygame.init()
    screen_width = COLS * TILE_SIZE
    screen_height = ROWS * TILE_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man Imitation Learning")
    clock = pygame.time.Clock()

    for episode in range(num_episodes):
        # Generate a new maze layout (or fixed maze if FIXED_MAZE is True)
        maze_layout = generate_maze(ROWS, COLS) if not FIXED_MAZE else None
        maze = Maze(maze_layout) if maze_layout else Maze(generate_maze(ROWS, COLS))
        open_cells = get_open_cells(maze.layout)
        if not open_cells:
            print("No open cells found in maze!")
            continue

        pac_cell = random.choice(open_cells)
        pac_x = pac_cell[1] * TILE_SIZE + TILE_SIZE // 2
        pac_y = pac_cell[0] * TILE_SIZE + TILE_SIZE // 2
        pac_x, pac_y = safe_spawn_pacman(maze.layout, ghosts=[])
        pacman = PacMan(pac_x, pac_y, auto_play=False, use_dqn=False)

        episode_demo = []  # List to store (state, action) pairs for this episode.
        done = False
        steps = 0

        while not done and steps < 1000:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Let the human control Pac-Man.
            pacman.handle_keys()

            # Capture the current state.
            state = pygame.surfarray.array3d(screen)

            # Update Pac-Man and maze.
            pacman.update(maze)

            # Record demonstration data.
            # Here we record the current screen state and the intended direction.
            demo_entry = (state, pacman.intended_direction.copy())
            episode_demo.append(demo_entry)

            # Render the game.
            screen.fill(BLACK)
            maze.draw(screen)
            pacman.draw(screen)
            pygame.display.flip()

            # End condition: if all pellets/fruit are collected.
            if not maze.pellets and not maze.fruits:
                done = True

            steps += 1

        demos.append(episode_demo)
        print(f"Collected demo for episode {episode}, length {len(episode_demo)} steps.")
        # Immediately save this episode's data.
        save_episode_demo(episode_demo, episode)
        # Save combined demonstrations.
        save_combined_demos(demos)

    pygame.quit()


if __name__ == "__main__":
    print("MODE:", MODE)
    print("IMITATION_MODE:", IMITATION_MODE)
    # Optional: press Enter to begin.
    input("Press Enter to start collecting demonstrations...")
    if MODE == "imitate" and IMITATION_MODE and RECORD_DEMOS:
        collect_demonstrations(num_episodes=5)
    else:
        print("Set MODE to 'imitate', IMITATION_MODE=True, and RECORD_DEMOS=True in settings.py")
