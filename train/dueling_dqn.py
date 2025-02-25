from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(DuelingDQN, self).__init__()
        # Shared convolutional feature extractor (same as before)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute the flattened size after conv layers
        # (Here we assume the output size is 7x7 based on input size 84x84.)
        self.fc_input_dim = 7 * 7 * 64

        # Value stream
        self.value_fc = nn.Sequential(nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        # Advantage stream
        self.advantage_fc = nn.Sequential(nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, output_dim))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        value = self.value_fc(x)  # shape: [batch, 1]
        advantage = self.advantage_fc(x)  # shape: [batch, output_dim]
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
