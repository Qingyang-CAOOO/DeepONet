# train_compare.py

import torch
from torch.utils.data import DataLoader
from models.branch import BranchNet
from models.trunk import TrunkNet
from models.deeponet import DeepONet
from models.fcnn import FCNNBaseline
from data.antiderivative_dataset import AntiderivativeDataset
import config
from utils import set_seed
import matplotlib.pyplot as plt

# Set reproducibility and device
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = AntiderivativeDataset(
    num_samples=2000,
    num_sensors=config.NUM_SENSORS,
    num_targets=3
)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Instantiate models
branch = BranchNet(config.NUM_SENSORS, config.LATENT_DIM, config.HIDDEN_DIM)
trunk = TrunkNet(1, config.LATENT_DIM, config.HIDDEN_DIM)
deeponet = DeepONet(branch, trunk).to(device)

fcnn = FCNNBaseline(input_dim=config.NUM_SENSORS + 1, hidden_dim=128).to(device)

# Loss and optimizers
criterion = torch.nn.MSELoss()
optimizer_deeponet = torch.optim.Adam(deeponet.parameters(), lr=config.LEARNING_RATE)
optimizer_fcnn = torch.optim.Adam(fcnn.parameters(), lr=config.LEARNING_RATE)

# Store loss per epoch
deeponet_losses = []
fcnn_losses = []

# Training loop
for epoch in range(config.NUM_EPOCHS):
    deeponet_loss_epoch = 0
    fcnn_loss_epoch = 0

    for u, y, s_true in dataloader:
        u, y, s_true = u.to(device), y.to(device), s_true.to(device)

        # --- DeepONet forward + backward ---
        optimizer_deeponet.zero_grad()
        s_pred_deep = deeponet(u, y)
        loss_deep = criterion(s_pred_deep, s_true)
        loss_deep.backward()
        optimizer_deeponet.step()
        deeponet_loss_epoch += loss_deep.item() * u.size(0)

        # --- FCNN forward + backward ---
        optimizer_fcnn.zero_grad()
        s_pred_fcnn = fcnn(u, y)  # internally concatenates
        loss_fcnn = criterion(s_pred_fcnn, s_true)
        loss_fcnn.backward()
        optimizer_fcnn.step()
        fcnn_loss_epoch += loss_fcnn.item() * u.size(0)

    # Log losses
    avg_loss_deep = deeponet_loss_epoch / len(dataset)
    avg_loss_fcnn = fcnn_loss_epoch / len(dataset)
    deeponet_losses.append(avg_loss_deep)
    fcnn_losses.append(avg_loss_fcnn)

    # Print progress
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}: DeepONet Loss = {avg_loss_deep:.6f} | FCNN Loss = {avg_loss_fcnn:.6f}")

# Save both models
torch.save(deeponet.state_dict(), "deeponet_antiderivative.pt")
torch.save(fcnn.state_dict(), "fcnn_antiderivative.pt")

# Plot training curves
plt.figure(figsize=(8, 5))
plt.plot(deeponet_losses, label="DeepONet")
plt.plot(fcnn_losses, label="FCNN")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Training Loss (log scale)")
plt.title("Training Loss Comparison: DeepONet vs FCNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_comparison.png")
plt.show()
