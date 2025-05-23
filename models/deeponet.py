import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net, use_output_bias=False):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net
        self.use_output_bias = use_output_bias
        self.output_bias = nn.Parameter(torch.zeros(1)) if use_output_bias else None

    def forward(self, u, y):
        b = self.branch(u)    # [batch, p]
        t = self.trunk(y)     # [batch, p]
        out = torch.sum(b * t, dim=-1, keepdim=True)  # scalar [batch_size, 1]
        if self.use_output_bias:
            out = out + self.output_bias
        return out

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
