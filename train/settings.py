MODE = "train"  # "train" or "play"
# Checkpoint file paths:
BEST_CHECKPOINT_PATH = "pacman_dqn_best.pth"  # Always stores the best model (with best_avg_reward and epsilon)
LATEST_CHECKPOINT_PATH = "pacman_dqn_latest.pth"  # Stores the latest model (can be overwritten)


HEADLESS = False  # For evaluation, you might want rendering.
DEBUG = False  # Extra per-step debug rendering.
RENDER_EVERY = 10  # Render final frame every N episodes during training.
FIXED_MAZE = True  # Use a fixed maze layout for initial episodes.
NUM_EPISODES = 10000  # Total training episodes.
MAX_STEPS_PER_EPISODE = 2500  # Maximum steps per episode.
TARGET_UPDATE_FREQ = 1000  # Frequency (in steps) to update target network.

# DQN and training hyperparameters:
FRAME_STACK_SIZE = 4  # Number of consecutive frames to stack
INPUT_CHANNELS = FRAME_STACK_SIZE or 1
ACTION_DIM = 4  # 0 = up, 1 = down, 2 = left, 3 = right.
LR = 1e-3
# LR = 0.00025  # original value from Atari paper
GAMMA = 0.99
BATCH_SIZE = 32
INITIAL_BUFFER_SIZE = 25000  # Start training after this many steps.
BUFFER_CAPACITY = 100000  # changed from 1000000
EPSILON_START = 1.0
EPSILON_LOAD_OVERWRITE = True  # If True, will overwrite epsilon from checkpoint.
EPSILON_END = 0.001
EPSILON_DECAY = 0.995

FRAME_STACK_SIZE = 4  # Number of consecutive frames to stack
TILE_SIZE = 50
ROWS = 12
COLS = 12
FPS = 60
PACMAN_YELLOW = (255, 255, 0)
WALL_BLUE = (33, 33, 222)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PELLET_ORANGE = (255, 153, 0)
FRUIT_RED = (255, 50, 50)
PELLET_SCORE = 10
FRUIT_SCORE = 50
NUM_FRUITS = 5
NUM_GHOSTS = 3

NUM_POWER_PELLETS = 10
POWER_PELLET_SCORE = 100
GHOST_SCORE = 200
GHOST_COLOR = (0, 255, 0)  # Default ghost color (green)
POWER_PELLET_COLOR = (255, 255, 255)  # Color for power pellets (purple)
