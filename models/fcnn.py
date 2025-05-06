import torch
import torch.nn as nn

class FCNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # outputs s(y) scalar
        )

    def forward(self, u, y):
        # Concatenate input function samples and query point
        x = torch.cat([u, y], dim=1)  # shape: [batch, m + 1]
        return self.net(x)
