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

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = AntiderivativeDataset(
    num_samples=2000,
    num_sensors=config.NUM_SENSORS,
    num_targets=3
)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

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

training_losses = {}
models = {}
optimizers = {}

# initialize all models and optimizers
for name, cfg in model_variants.items():
    if "model" in cfg:
        model = cfg["model"]
    else:
        model = DeepONet(cfg["branch"], cfg["trunk"], use_output_bias=cfg["bias"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    models[name] = model
    optimizers[name] = optimizer
    training_losses[name] = []

criterion = torch.nn.MSELoss()

# training loop
for epoch in range(config.NUM_EPOCHS):
    losses_epoch = {name: 0.0 for name in models}
    
    for u, y, s_true in dataloader:
        u, y, s_true = u.to(device), y.to(device), s_true.to(device)
        
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad()
            
            if name == "fcnn":
                s_pred = model(u, y)
            else:
                s_pred = model(u, y)

            loss = criterion(s_pred, s_true)
            loss.backward()
            optimizer.step()
            losses_epoch[name] += loss.item() * u.size(0)

    for name in losses_epoch:
        avg_loss = losses_epoch[name] / len(dataset)
        training_losses[name].append(avg_loss)

    if (epoch + 1) % 100 == 0 or epoch == 0:
        log = f"Epoch {epoch + 1}:"
        for name in models:
            log += f" {name.replace('_', ' ')} Loss = {training_losses[name][-1]:.6f} |"
        print(log.rstrip("|"))

# save models and plot
for name, model in models.items():
    torch.save(model.state_dict(), f"{name}.pt")

plt.figure(figsize=(16, 9))
for name, losses in training_losses.items():
    plt.plot(losses, label=name)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Training Loss (log scale)")
plt.title("Training Losses Across DeepONet Variants and FCNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("all_training_loss_comparison.png", dpi=300)
plt.show()
