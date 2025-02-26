import os
import pickle
import random
import sys
from collections import deque

import cv2
import numpy as np
import pygame

from maze import Maze, generate_maze, get_open_cells, safe_spawn_pacman
from pacman import PacMan
from train.settings import (
    BLACK,
    COLS,
    DEMO_DATA_PATH,
    FIXED_MAZE,
    FPS,
    FRAME_STACK_SIZE,  # Ensure FRAME_STACK_SIZE is defined in settings, e.g., 4
    IMITATION_MODE,
    MODE,
    RECORD_DEMOS,
    ROWS,
    TILE_SIZE,
)


def load_existing_demos():
    if os.path.exists(DEMO_DATA_PATH):
        with open(DEMO_DATA_PATH, "rb") as f:
            demos = pickle.load(f)
        print(f"Loaded {len(demos)} existing demonstration episodes.")
    else:
        demos = []
    return demos


def save_episode_demo(episode_demo, episode):
    episode_file = f"demonstration_episode_{episode}.pkl"
    with open(episode_file, "wb") as f:
        pickle.dump(episode_demo, f)
    print(f"Saved demonstration data for episode {episode} to {episode_file}.")


def save_combined_demos(demos):
    with open(DEMO_DATA_PATH, "wb") as f:
        pickle.dump(demos, f)
    print(f"Saved combined demonstration data to {DEMO_DATA_PATH}.")


def preprocess_frame(frame):
    # Convert captured RGB frame (from pygame) to grayscale, resize to 84x84, and normalize.
    frame = np.transpose(frame, (1, 0, 2))  # (height, width, channels)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized.astype(np.float32) / 255.0
    return normalized  # shape (84,84)


def collect_demonstrations(num_episodes=25):
    demos = load_existing_demos()

    pygame.init()
    screen_width = COLS * TILE_SIZE
    screen_height = ROWS * TILE_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man Imitation Learning")
    clock = pygame.time.Clock()

    for episode in range(num_episodes):
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
        pacman = PacMan(pac_x, pac_y, auto_play=True, use_dqn=False)

        # Initialize a frame stack for this episode
        frame_stack = deque(maxlen=FRAME_STACK_SIZE)

        # Collect an initial frame and fill the stack if necessary.
        initial_frame = preprocess_frame(pygame.surfarray.array3d(screen))
        for _ in range(FRAME_STACK_SIZE):
            frame_stack.append(initial_frame)

        episode_demo = []  # This will store (stacked_state, intended_direction) pairs.
        done = False
        steps = 0

        while not done and steps < 1000:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pacman.handle_keys()
            # Update Pac-Man and maze.
            pacman.update(maze)

            # Capture the current frame, preprocess it, and add to the stack.
            current_frame = preprocess_frame(pygame.surfarray.array3d(screen))
            frame_stack.append(current_frame)
            # Create a stacked state as a numpy array with shape (FRAME_STACK_SIZE, 84, 84)
            stacked_state = np.array(frame_stack)

            # Record demonstration data (stacked state + intended direction)
            demo_entry = (stacked_state, pacman.intended_direction.copy())
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
        save_episode_demo(episode_demo, episode)
        save_combined_demos(demos)

    pygame.quit()


if __name__ == "__main__":
    MODE = "imitate"  # Set to "imitate" for imitation learning mode.

    print("MODE:", MODE)
    print("IMITATION_MODE:", IMITATION_MODE)
    input("Press Enter to start collecting demonstrations...")
    if MODE == "imitate" and IMITATION_MODE and RECORD_DEMOS:
        collect_demonstrations(num_episodes=100)
    else:
        print("Set MODE to 'imitate', IMITATION_MODE=True, and RECORD_DEMOS=True in settings.py")
