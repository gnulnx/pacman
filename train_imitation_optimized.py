import os
import pickle

import cv2
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Import your DuelingDQN architecture for imitation training.
from train.dueling_dqn import DuelingDQN

# Import settings
from train.settings import (
    ACTION_DIM,
    BATCH_SIZE,
    DEMO_DATA_PATH,
    INPUT_CHANNELS,
    LR,
    NUM_EPOCHS,
)

# Set a number of workers for DataLoader to speed up data loading.
# Adjust NUM_WORKERS based on your system (e.g., 4 or 8).
NUM_WORKERS = 0
FRAME_STACK_SIZE = 4

# Device selection (using MPS, then CUDA, then CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


# Helper function: convert a pygame vector (intended_direction) to a discrete action.
def direction_to_action(direction):
    """
    Convert a pygame Vector2 intended_direction to a discrete action index
    using absolute actions:
        0: Up    (pygame.math.Vector2(0, -1))
        1: Down  (pygame.math.Vector2(0, 1))
        2: Left  (pygame.math.Vector2(-1, 0))
        3: Right (pygame.math.Vector2(1, 0))

    This implementation computes the dot product between the normalized input vector
    and each of the cardinal directions, returning the index of the direction with the highest similarity.
    """
    # Handle the case where the vector is zero-length; default to 'up' (action 0)
    if direction.length() == 0:
        return 0

    # Normalize the input vector
    norm_dir = direction.normalize()

    # Define the cardinal directions exactly as in your RL training function.
    cardinal_dirs = [
        pygame.math.Vector2(0, -1),  # Up (action 0)
        pygame.math.Vector2(0, 1),  # Down (action 1)
        pygame.math.Vector2(-1, 0),  # Left (action 2)
        pygame.math.Vector2(1, 0),  # Right (action 3)
    ]

    # Compute the dot product between the input vector and each cardinal direction.
    dot_products = [norm_dir.dot(card_dir) for card_dir in cardinal_dirs]

    # The action corresponding to the cardinal direction with the highest dot product is returned.
    return dot_products.index(max(dot_products))


# Dataset for imitation learning.
class ImitationDataset(Dataset):
    def __init__(self, demo_file):
        if not os.path.exists(demo_file):
            print(f"Demo file {demo_file} does not exist.")
            self.data = []
            return
        with open(demo_file, "rb") as f:
            demos = pickle.load(f)
        print(f"[DEBUG] Loaded demos from {demo_file}: type={type(demos)}, length={len(demos)}")
        self.data = []
        for episode in demos:
            for state, intended_direction in episode:
                action = direction_to_action(intended_direction)
                self.data.append((state, action))
        print(f"Loaded {len(self.data)} demonstration transitions from {demo_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action = self.data[idx]

        # Check the shape of the state.
        # If it's already a stacked state, its shape will be (FRAME_STACK_SIZE, 84, 84)
        if state.ndim == 3 and state.shape[0] == FRAME_STACK_SIZE:
            # Assume state is already preprocessed.
            state_processed = state
        else:
            # Otherwise, assume the state is a raw RGB image with shape (width, height, 3)
            # and process it as before.
            state = np.transpose(state, (1, 0, 2))  # now (height, width, 3)
            gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84))
            normalized = resized.astype(np.float32) / 255.0
            if INPUT_CHANNELS > 1:
                state_processed = np.stack([normalized] * INPUT_CHANNELS, axis=0)
            else:
                state_processed = np.expand_dims(normalized, axis=0)

        state_tensor = torch.tensor(state_processed, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        return state_tensor, action_tensor


def train_imitation():
    dataset = ImitationDataset(DEMO_DATA_PATH)
    if len(dataset) == 0:
        print("No demonstration data found. Exiting.")
        return

    # Use the improved NUM_WORKERS in DataLoader for faster data loading.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Ensure model is sent to the selected device.
    model = DuelingDQN(INPUT_CHANNELS, ACTION_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Set up GradScaler for mixed precision training (only effective on CUDA)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    print("Starting imitation training...")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for states, actions in dataloader:
            # Transfer data to device
            states = states.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            # Use autocast for mixed precision (if using CUDA)
            with torch.amp.autocast(enabled=(scaler is not None), device_type="mps"):
                logits = model(states)
                loss = criterion(logits, actions)

            # Backward pass using GradScaler if available
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

        # Early stopping: if loss hasn't improved by IMT_MIN_DELTA for IMT_EARLY_STOP_PATIENCE epochs, stop.
        if best_loss - epoch_loss > 0.001:  # IMT_MIN_DELTA
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No significant improvement for {patience_counter} epoch(s).")
            if patience_counter >= 10:  # IMT_EARLY_STOP_PATIENCE
                print("Early stopping triggered.")
                break

    torch.save({"model_state": model.state_dict()}, "imitation_model.pth")
    print("Imitation training complete. Model saved as imitation_model.pth.")


if __name__ == "__main__":
    print("MODE:", "imitate")
    print("IMITATION_MODE:", True)
    # Removed blocking input() call to avoid unnecessary waiting.
    train_imitation()
