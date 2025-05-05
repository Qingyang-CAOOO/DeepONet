# train.py

# Import core PyTorch modules
import torch
from torch.utils.data import DataLoader

# Import DeepONet components
from models.branch import BranchNet            # Branch network definition
from models.trunk import TrunkNet              # Trunk network definition
from models.deeponet import DeepONet           # Combines branch & trunk into DeepONet

# Import the dataset specific to Problem 1.A (antiderivative operator)
from data.antiderivative_dataset import AntiderivativeDataset

# Import hyperparameter configuration
import config

# Import utility function for reproducibility
from utils import set_seed

# Set random seed for reproducibility
set_seed()

# ---------------------------------------------------------------------
# 🧪 DATASET & DATALOADER SETUP
# ---------------------------------------------------------------------

# Create dataset: 2000 input functions, 100 sensor points, 3 target y-points per function
dataset = AntiderivativeDataset(
    num_samples=2000,
    num_sensors=config.NUM_SENSORS,
    num_targets=3
)

# Wrap dataset in a DataLoader for batching and shuffling during training
dataloader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

# ---------------------------------------------------------------------
# 🧠 MODEL INSTANTIATION
# ---------------------------------------------------------------------

# Create branch and trunk networks using config-specified dimensions
branch = BranchNet(
    input_dim=config.NUM_SENSORS,
    latent_dim=config.LATENT_DIM,
    hidden_dim=config.HIDDEN_DIM
)

trunk = TrunkNet(
    input_dim=1,  # query point y is scalar
    latent_dim=config.LATENT_DIM,
    hidden_dim=config.HIDDEN_DIM
)

# Combine into full DeepONet model
model = DeepONet(branch, trunk)

# ---------------------------------------------------------------------
# ⚙️ OPTIMIZER AND LOSS
# ---------------------------------------------------------------------

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Use mean squared error loss
criterion = torch.nn.MSELoss()

# ---------------------------------------------------------------------
# 🔁 TRAINING LOOP
# ---------------------------------------------------------------------

# Loop over epochs
for epoch in range(config.NUM_EPOCHS):
    total_loss = 0  # track total loss for the epoch

    # Iterate over mini-batches
    for u, y, s_true in dataloader:
        optimizer.zero_grad()        # Reset gradients to zero
        s_pred = model(u, y)         # Forward pass through DeepONet
        loss = criterion(s_pred, s_true)  # Compute MSE loss
        loss.backward()              # Backpropagate the error
        optimizer.step()             # Update model parameters
        total_loss += loss.item() * u.size(0)  # Accumulate weighted loss

    # Log average epoch loss every 100 epochs (or first)
    if (epoch + 1) % 100 == 0 or epoch == 0:
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

# ---------------------------------------------------------------------
# 💾 SAVE TRAINED MODEL
# ---------------------------------------------------------------------

# Save model weights to a .pt file for future inference
torch.save(model.state_dict(), "deeponet_antiderivative.pt")
