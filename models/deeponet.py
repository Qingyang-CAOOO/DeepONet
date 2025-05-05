import torch
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net

    def forward(self, u, y):
        b = self.branch(u)     # [batch_size, p]
        t = self.trunk(y)      # [batch_size, p]
        return torch.sum(b * t, dim=-1, keepdim=True)  # [batch_size, 1]

    def encode_branch(self, u):
        return self.branch(u)

    def encode_trunk(self, y):
        return self.trunk(y)

    def evaluate_with_gradients(self, u, y):
        u = u.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        s = self.forward(u, y)
        s.backward(torch.ones_like(s), retain_graph=True)
        return {
            "value": s.detach(),
            "∂s/∂u": u.grad.detach(),
            "∂s/∂y": y.grad.detach()
        }
