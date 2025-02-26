# settings.py
# =====================================================
# General Settings & Modes
# =====================================================
MODE = "train"  # "train" or "play"
HEADLESS = False  # For evaluation, you might want rendering.
DEBUG = False  # Extra per-step debug rendering.
USE_8BIT = True
USE_NOISY = True  # Set to True to use NoisyLinear layers for exploration
USE_ABSOLUTE_ACTIONS = True  # If True, 0=Up,1=Down,2=Left,3=Right; if False, 0=forward,1=left,2=right,3=reverse

# For imitation learning, set to True to collect demonstration data.
NUM_EPOCHS = 1000

# =====================================================
# Checkpoint & Logging Settings
# =====================================================
# Which checkpoint to load when restarting training:
# Options: "best_avg", "best_single", or "latest"
LOAD_CHECKPOINT = "best_avg"

BEST_AVG_CHECKPOINT_PATH = "pacman_dqn_best_avg.pth"  # Will store the best model by average reward
BEST_SINGLE_CHECKPOINT_PATH = "pacman_dqn_best_single.pth"  # Already stores best single-episode score
LATEST_CHECKPOINT_PATH = "pacman_dqn_latest.pth"  # Always stores the latest model

MODE = "train"  # "train", "play", or "imitate"
IMITATION_MODE = True  # Set to True when collecting demonstration data

# Imitation learning settings:
RECORD_DEMOS = True  # If True, record demonstration data during imitation mode.
DEMO_DATA_PATH = "demonstrations.pkl"  # Where


RENDER_EVERY = 10  # Render final frame every N episodes during training.
FIXED_MAZE = True  # Use a fixed maze layout for initial episodes.

# =====================================================
# Episode & Target Network Settings
# =====================================================
NUM_EPISODES = 10000  # Total training episodes.
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps per episode.
TARGET_UPDATE_FREQ = 1000  # Frequency (in steps) to update target network.

# =====================================================
# DQN & Training Hyperparameters
# =====================================================
FRAME_STACK_SIZE = 4  # Number of consecutive frames to stack
INPUT_CHANNELS = FRAME_STACK_SIZE or 1  # Set to the number of frames stacked (or 1 if not stacking)
ACTION_DIM = 4  # 0 = up, 1 = down, 2 = left, 3 = right.
# LR = 1e-3  # Learning rate
LR = 0.00025  # Original value from Atari paper
GAMMA = 0.99  # Discount factor for future rewards
BATCH_SIZE = 32  # Number of transitions per training batch
INITIAL_BUFFER_SIZE = 100  # Start training after this many steps.
BUFFER_CAPACITY = 100000  # Maximum size of the replay buffer (changed from 1,000,000)
EPSILON_START = 0.05  # Initial epsilon for exploration
EPSILON_LOAD_OVERWRITE = True  # If True, will overwrite epsilon from checkpoint.
EPSILON_END = 0.001  # Minimum epsilon value
EPSILON_DECAY = 0.995  # Epsilon decay rate per step
NOISY_DECAY = 1.0  # Decay rate for NoisyLinear layers (if used) (1.0 for no decay)

# Priority buffer settings
USE_PRIORITY_BUFFER = True  # If True, use Prioritized Experience Replay; otherwise, use uniform sampling
PER_ALPHA = 0.6  # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
PER_BETA_START = 0.4  # Importance sampling exponent (start)
PER_BETA_FRAMES = 100000  # Over how many frames PER_BETA will anneal to 1.0

# =====================================================
# (Repeated) Frame Stacking Setting
# =====================================================
FRAME_STACK_SIZE = 4  # Number of consecutive frames to stack

# =====================================================
# Maze & Environment Dimensions
# =====================================================
TILE_SIZE = 50
ROWS = 10
COLS = 10
FPS = 60

# =====================================================
# Colors & Visuals
# =====================================================
PACMAN_YELLOW = (255, 255, 0)
WALL_BLUE = (33, 33, 222)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PELLET_ORANGE = (255, 153, 0)
FRUIT_RED = (255, 50, 50)

NUM_FRUITS = 5

# =====================================================
# Scoring & Gameplay Mechanics
# =====================================================
PELLET_SCORE = 5
FRUIT_SCORE = 5
EAT_GHOST_SCORE = 10
GHOST_CATCH_SCORE = 10
CLEAR_BOARD = 50
MOVE_TOWARD_PELLOT = 0.1
MOVE_AWAY_FROM_PELLOT = 0.2
WALL_COLLISION_PENALTY = -0.1
NOVELTY_BONUS = 1  # Bonus for exploring new areas of the maze
STEP_PENALTY = 0.1

# =====================================================
# Imitation Learning Integration
# =====================================================
LOAD_FROM_IMITATION = True  # Whether to start from an imitation-learned model
IMITATION_MODEL_PATH = "imitation_model.pth"  # Path to the imitation model
# IMITATION_EPSILON_START = 0.3  # Start with lower epsilon when using imitation model

# =====================================================
# Entity Appearance Settings
# =====================================================
GHOST_COLOR = (0, 255, 0)  # Default ghost color (green)
POWER_PELLET_COLOR = (255, 255, 255)  # Color for power pellets (purple)
NUM_POWER_PELLETS = 0
