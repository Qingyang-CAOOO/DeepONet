# evaluate_compare.py

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.branch import BranchNet
from models.trunk import TrunkNet
from models.deeponet import DeepONet
from models.fcnn import FCNNBaseline
import config

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generate a smooth test function u(x) = sum a_n sin(nπx)
def generate_test_function(x_sensor):
    coeffs = np.array([1.0, -0.5, 0.25])  # fixed for reproducibility
    u = (
        coeffs[0] * np.sin(np.pi * x_sensor)
        + coeffs[1] * np.sin(2 * np.pi * x_sensor)
        + coeffs[2] * np.sin(3 * np.pi * x_sensor)
    )
    return u.astype(np.float32)

# evaluation grid
y_grid = np.linspace(0, 1, 200).astype(np.float32)
x_sensor = np.linspace(0, 1, config.NUM_SENSORS).astype(np.float32)

# generate test u(x)
u_test = generate_test_function(x_sensor)
u_tensor = torch.tensor(u_test, dtype=torch.float32).unsqueeze(0).repeat(len(y_grid), 1).to(device)
y_tensor = torch.tensor(y_grid[:, None], dtype=torch.float32).to(device)

# compute true antiderivative
s_true = np.array([
    np.trapz(u_test[x_sensor <= y], x_sensor[x_sensor <= y]) for y in y_grid
])

# load DeepONet model
branch = BranchNet(config.NUM_SENSORS, config.LATENT_DIM, config.HIDDEN_DIM)
trunk = TrunkNet(1, config.LATENT_DIM, config.HIDDEN_DIM)
deeponet = DeepONet(branch, trunk).to(device)
deeponet.load_state_dict(torch.load("deeponet_antiderivative.pt", map_location=device))
deeponet.eval()

# load FCNN model
fcnn = FCNNBaseline(input_dim=config.NUM_SENSORS + 1, hidden_dim=128).to(device)
fcnn.load_state_dict(torch.load("fcnn_antiderivative.pt", map_location=device))
fcnn.eval()

# predict with DeepONet and FCNN
with torch.no_grad():
    s_pred_deep = deeponet(u_tensor, y_tensor).cpu().numpy().flatten()
    s_pred_fcnn = fcnn(u_tensor, y_tensor).cpu().numpy().flatten()

# plot results
plt.figure(figsize=(8, 5))
plt.plot(y_grid, s_true, label="True Integral", linewidth=2)
plt.plot(y_grid, s_pred_deep, label="DeepONet", linestyle="--")
plt.plot(y_grid, s_pred_fcnn, label="FCNN", linestyle=":")
plt.xlabel("y")
plt.ylabel("s(y)")
plt.title("Prediction Comparison on Test Function")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_comparison.png")
plt.show()
