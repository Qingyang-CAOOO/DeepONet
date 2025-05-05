import torch.nn as nn

class TrunkNet(nn.Module):
    def __init__(self, input_dim=1, latent_dim=30, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, y):
        return self.net(y)
