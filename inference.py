import torch
import numpy as np
import matplotlib.pyplot as plt

from models.branch import BranchNet
from models.branch_stacked import StackedBranchNet
from models.trunk import TrunkNet
from models.deeponet import DeepONet
from models.fcnn import FCNNBaseline
import config

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# define test function u(x)
def generate_test_function(x_sensor):
    coeffs = np.array([1.0, -0.5, 0.25])  # fixed coefficients
    u = (
        coeffs[0] * np.sin(np.pi * x_sensor)
        + coeffs[1] * np.sin(2 * np.pi * x_sensor)
        + coeffs[2] * np.sin(3 * np.pi * x_sensor)
    )
    return u.astype(np.float32)

# prepare sensor and evaluation grid
x_sensor = np.linspace(0, 1, config.NUM_SENSORS).astype(np.float32)
y_grid = np.linspace(0, 1, 200).astype(np.float32)

u_test = generate_test_function(x_sensor)
u_tensor = torch.tensor(u_test, dtype=torch.float32).unsqueeze(0).repeat(len(y_grid), 1).to(device)
y_tensor = torch.tensor(y_grid[:, None], dtype=torch.float32).to(device)

# compute true antiderivative
s_true = np.array([
    np.trapz(u_test[x_sensor <= y], x_sensor[x_sensor <= y]) for y in y_grid
])

# define model variants to load
model_variants = {
    "deeponet_unstacked": (BranchNet, False),
    "deeponet_unstacked_bias": (BranchNet, True),
    "deeponet_stacked": (StackedBranchNet, False),
    "deeponet_stacked_bias": (StackedBranchNet, True),
    "fcnn": None
}

# run inference for all models
predictions = {}

for name, config_tuple in model_variants.items():
    print(f"Evaluating: {name}")
    if name == "fcnn":
        model = FCNNBaseline(config.NUM_SENSORS + 1, hidden_dim=128)
        model.load_state_dict(torch.load(f"{name}.pt", map_location=device))
        model.to(device).eval()
        with torch.no_grad():
            pred = model(u_tensor, y_tensor).cpu().numpy().flatten()
        predictions[name] = pred
    else:
        branch_cls, use_bias = config_tuple
        branch = branch_cls(config.NUM_SENSORS, config.LATENT_DIM, config.HIDDEN_DIM)
        trunk = TrunkNet(1, config.LATENT_DIM, config.HIDDEN_DIM)
        model = DeepONet(branch, trunk, use_output_bias=use_bias)
        model.load_state_dict(torch.load(f"{name}.pt", map_location=device))
        model.to(device).eval()
        with torch.no_grad():
            pred = model(u_tensor, y_tensor).cpu().numpy().flatten()
        predictions[name] = pred

# plot all predictions
plt.figure(figsize=(16, 9))
plt.plot(y_grid, s_true, label="True", linewidth=2, color='black')
for name, pred in predictions.items():
    plt.plot(y_grid, pred, label=name.replace("_", " "), linestyle='--')
plt.xlabel("y")
plt.ylabel("s(y)")
plt.title("Prediction Comparison on Test Function")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_comparison_all_models.png", dpi=300)
plt.show()
