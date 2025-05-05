import torch
from torch.utils.data import Dataset
import numpy as np

class AntiderivativeDataset(Dataset):
    def __init__(self, num_samples=1000, num_sensors=100, num_targets=1):
        super().__init__()
        self.num_samples = num_samples
        self.num_sensors = num_sensors
        self.num_targets = num_targets
        self.x_sensor = np.linspace(0, 1, num_sensors)

        self.data = []
        for _ in range(num_samples):
            coeffs = np.random.randn(3)
            u = (
                coeffs[0] * np.sin(np.pi * self.x_sensor)
                + coeffs[1] * np.sin(2 * np.pi * self.x_sensor)
                + coeffs[2] * np.sin(3 * np.pi * self.x_sensor)
            )

            y_targets = np.random.rand(num_targets)
            s_targets = np.array([
                np.trapz(u[self.x_sensor <= y], self.x_sensor[self.x_sensor <= y])
                for y in y_targets
            ])

            self.data.append((u.astype(np.float32), y_targets.astype(np.float32), s_targets.astype(np.float32)))

    def __len__(self):
        return self.num_samples * self.num_targets

    def __getitem__(self, idx):
        sample_idx = idx // self.num_targets
        target_idx = idx % self.num_targets
        u, y_array, s_array = self.data[sample_idx]
        return (
            torch.tensor(u), 
            torch.tensor([y_array[target_idx]]), 
            torch.tensor([s_array[target_idx]])
        )
