import torch
from torch.utils.data import DataLoader
from models.branch import BranchNet
from models.trunk import TrunkNet
from models.branch_stacked import StackedBranchNet
from models.deeponet import DeepONet
from models.fcnn import FCNNBaseline
from data.antiderivative_dataset import AntiderivativeDataset
import config
from utils import set_seed
import matplotlib.pyplot as plt

# set seed and device
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# create dataset and dataloader
dataset = AntiderivativeDataset(
    num_samples=2000,
    num_sensors=config.NUM_SENSORS,
    num_targets=3
)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# define model variants
model_variants = {
    "deeponet_unstacked": {
        "branch": BranchNet(config.NUM_SENSORS, config.LATENT_DIM, config.HIDDEN_DIM),
        "trunk": TrunkNet(1, config.LATENT_DIM, config.HIDDEN_DIM),
        "bias": False
    },
    "deeponet_unstacked_bias": {
        "branch": BranchNet(config.NUM_SENSORS, config.LATENT_DIM, config.HIDDEN_DIM),
        "trunk": TrunkNet(1, config.LATENT_DIM, config.HIDDEN_DIM),
        "bias": True
    },
    "deeponet_stacked": {
        "branch": StackedBranchNet(config.NUM_SENSORS, config.LATENT_DIM, config.HIDDEN_DIM),
        "trunk": TrunkNet(1, config.LATENT_DIM, config.HIDDEN_DIM),
        "bias": False
    },
    "deeponet_stacked_bias": {
        "branch": StackedBranchNet(config.NUM_SENSORS, config.LATENT_DIM, config.HIDDEN_DIM),
        "trunk": TrunkNet(1, config.LATENT_DIM, config.HIDDEN_DIM),
        "bias": True
    },
    "fcnn": {
        "model": FCNNBaseline(config.NUM_SENSORS + 1, hidden_dim=128)
    }
}

# store training losses for plotting
training_losses = {}

# train all models
for name, cfg in model_variants.items():
    print(f"\nTraining model: {name}")
    if "model" in cfg:
        model = cfg["model"]
    else:
        model = DeepONet(cfg["branch"], cfg["trunk"], use_output_bias=cfg["bias"])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    losses = []

    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0
        for u, y, s_true in dataloader:
            u, y, s_true = u.to(device), y.to(device), s_true.to(device)
            optimizer.zero_grad()

            if name == "fcnn":
                s_pred = model(u, y)
            else:
                s_pred = model(u, y)

            loss = criterion(s_pred, s_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * u.size(0)

        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    training_losses[name] = losses
    torch.save(model.state_dict(), f"{name}.pt")

# plot all training curves
plt.figure(figsize=(10, 6))
for name, loss_list in training_losses.items():
    plt.plot(loss_list, label=name)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Training Loss (log scale)")
plt.title("Training Losses Across DeepONet Variants and FCNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("all_training_loss_comparison.png")
plt.show()
