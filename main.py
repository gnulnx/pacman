import pygame
import random
import sys
import math
import torch
import torch.nn as nn
import cv2
import numpy as np

# --- Configuration for DQN auto-play ---
AUTO_PLAY = True
USE_DQN = True                              # Set to True to use the trained DQN for auto-play.
MODEL_PATH = "pacman_dqn_best.pth"            # Path to the trained model checkpoint.
INPUT_CHANNELS = 4                          # Number of input channels (frame stack size).
ACTION_DIM = 4                              # 0 = up, 1 = down, 2 = left, 3 = right.

from pacman import PacMan
from ghost import Ghost
from maze import Maze, generate_maze, get_open_cells, safe_spawn_pacman
from settings import TILE_SIZE, FPS, BLACK, NUM_GHOSTS, ROWS, COLS, WHITE, GHOST_SCORE

# --- DQN Model Definition ---
class DQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- If using DQN, load the model ---
if USE_DQN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(INPUT_CHANNELS, ACTION_DIM).to(device)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"Loaded DQN model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        USE_DQN = False  # Fall back to built-in auto-play if loading fails.

# --- Helper Functions for DQN state capture and action selection ---
def get_dqn_state(surface):
    """
    Capture the current screen and process it to produce a state for DQN.
    Returns a numpy array with shape (INPUT_CHANNELS, 84, 84).
    Here we simply capture one frame and replicate it.
    """
    image = pygame.surfarray.array3d(surface)
    image = np.transpose(image, (1, 0, 2))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized.astype(np.float32) / 255.0
    # Replicate the single frame to create a stack of frames.
    state = np.stack([normalized] * INPUT_CHANNELS, axis=0)
    return state

def select_action_dqn(state):
    """
    Given a state (numpy array with shape (INPUT_CHANNELS,84,84)), select an action.
    """
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    action = q_values.argmax(dim=1).item()
    return action

def action_to_direction(action):
    """
    Map DQN action index to a pygame.Vector2 direction.
    0: up, 1: down, 2: left, 3: right.
    """
    if action == 0:
        return pygame.math.Vector2(0, -1)
    elif action == 1:
        return pygame.math.Vector2(0, 1)
    elif action == 2:
        return pygame.math.Vector2(-1, 0)
    elif action == 3:
        return pygame.math.Vector2(1, 0)
    else:
        return pygame.math.Vector2(0, 0)

# --- Main Game Loop ---
def run_game():
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
        return "Error"

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

    # Spawn Pac-Man at a safe location.
    pac_x, pac_y = safe_spawn_pacman(maze_layout, ghosts)
    # If using DQN, set use_dqn=True so that internal auto-navigation is skipped.
    pacman = PacMan(pac_x, pac_y, auto_play=AUTO_PLAY, use_dqn=USE_DQN)

    font = pygame.font.SysFont(None, 36)
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # --- DQN control ---
        if USE_DQN:
            # Use the previous frame (or current screen) to decide an action.
            state = get_dqn_state(screen)
            action = select_action_dqn(state)
            pacman.intended_direction = action_to_direction(action)
        else:
            pacman.handle_keys()

        pacman.update(maze_obj)
        for ghost in ghosts:
            ghost.vulnerable = (pygame.time.get_ticks() < pacman.powerpellet_end)
            ghost.update(maze_obj, pacman)

        # Collision check between Pac-Man and ghosts.
        for ghost in ghosts:
            distance = math.hypot(pacman.x - ghost.x, pacman.y - ghost.y)
            if distance < pacman.radius + ghost.radius:
                if pygame.time.get_ticks() < pacman.powerpellet_end:
                    pacman.score += GHOST_SCORE
                    ghost_cell = random.choice(open_cells)
                    ghost.x = ghost_cell[1] * TILE_SIZE + TILE_SIZE // 2
                    ghost.y = ghost_cell[0] * TILE_SIZE + TILE_SIZE // 2
                    ghost.vulnerable = False
                else:
                    pacman.lives -= 1
                    if pacman.lives <= 0:
                        return "Game Over"
                    else:
                        new_x, new_y = safe_spawn_pacman(maze_layout, ghosts)
                        pacman.x, pacman.y = new_x, new_y
                        pacman.direction = pygame.math.Vector2(0, 0)
                        pacman.intended_direction = pygame.math.Vector2(0, 0)
                        break

        # Win condition: all pellets, fruits, and power pellets collected.
        if not maze_obj.pellets and not maze_obj.fruits and not maze_obj.power_pellets:
            return "You Win"

        # Drawing:
        screen.fill(BLACK)
        maze_obj.draw(screen)
        pacman.draw(screen)
        for ghost in ghosts:
            ghost.draw(screen)

        score_text = font.render(f"Score: {pacman.score}", True, WHITE)
        lives_text = font.render(f"Lives: {pacman.lives}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (10, 40))
        pygame.display.flip()

def show_end_screen(message):
    screen = pygame.display.get_surface()
    font = pygame.font.SysFont(None, 48)
    text = font.render(f"{message}! Play again? (Y/N)", True, WHITE)
    text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.fill(BLACK)
    screen.blit(text, text_rect)
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return True
                elif event.key == pygame.K_n:
                    return False

def main():
    pygame.init()
    play_again = True
    while play_again:
        outcome = run_game()
        if outcome == "Error":
            break
        play_again = show_end_screen(outcome)
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
