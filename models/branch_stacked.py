import torch
import torch.nn as nn

class StackedBranchNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(latent_dim)
        ])

    def forward(self, u):
        # stack outputs from all branch networks
        b_k = [branch(u).squeeze(-1) for branch in self.branches]  # list of shape [batch]
        return torch.stack(b_k, dim=1)  # shape: [batch, latent_dim]
