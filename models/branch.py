import torch.nn as nn

class BranchNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, u):
        return self.net(u)
