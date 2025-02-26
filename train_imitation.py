import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Import settings
from train.settings import (
    ACTION_DIM,
    BATCH_SIZE,
    DEMO_DATA_PATH,
    INPUT_CHANNELS,
    LR,
    NUM_EPOCHS,
)

# If not in settings, you can define early stopping parameters here:
IMT_EARLY_STOP_PATIENCE = 10  # number of epochs with no improvement before stopping
IMT_MIN_DELTA = 0.001  # minimum improvement in loss to be considered as progress

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Helper function: convert a pygame vector (intended_direction) to a discrete action.
def direction_to_action(direction):
    # We assume direction is a pygame.math.Vector2.
    threshold = 0.5
    x, y = direction.x, direction.y
    if abs(x) < threshold and y < -threshold:
        return 0  # up
    elif abs(x) < threshold and y > threshold:
        return 1  # down
    elif x < -threshold and abs(y) < threshold:
        return 2  # left
    elif x > threshold and abs(y) < threshold:
        return 3  # right
    else:
        if abs(x) >= abs(y):
            return 2 if x < 0 else 3
        else:
            return 0 if y < 0 else 1


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
        # Process state: convert from (width, height, 3) to (INPUT_CHANNELS, 84, 84)
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


# Use your DuelingDQN architecture for imitation training.
from train.dueling_dqn import DuelingDQN


def train_imitation():
    dataset = ImitationDataset(DEMO_DATA_PATH)
    if len(dataset) == 0:
        print("No demonstration data found. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelingDQN(INPUT_CHANNELS, ACTION_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting imitation training...")
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for states, actions in dataloader:
            states = states.to(device)  # shape: (BATCH_SIZE, INPUT_CHANNELS, 84,84)
            actions = actions.to(device)  # shape: (BATCH_SIZE,)
            logits = model(states)  # shape: (BATCH_SIZE, ACTION_DIM)
            loss = criterion(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

        # Early stopping: if loss hasn't improved by IMT_MIN_DELTA for IMT_EARLY_STOP_PATIENCE epochs, stop.
        if best_loss - epoch_loss > IMT_MIN_DELTA:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No significant improvement for {patience_counter} epoch(s).")
            if patience_counter >= IMT_EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

    torch.save({"model_state": model.state_dict()}, "imitation_model.pth")
    print("Imitation training complete. Model saved as imitation_model.pth.")


if __name__ == "__main__":
    print("MODE:", "imitate")
    print("IMITATION_MODE:", True)
    input("Press Enter to start imitation training...")
    train_imitation()
