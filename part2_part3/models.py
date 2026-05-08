"""Batch-normalized MLP and SmallCNN architectures used throughout Part II/III."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Linear(784,256)-BN-ReLU - Linear(256,128)-BN-ReLU - Linear(128,10)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class SmallCNN(nn.Module):
    """Two conv blocks (Conv-BN-ReLU-Pool) with 32 channels, then Linear-ReLU-Linear."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1568, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_flat_params(model):
    """Concatenate all parameters into a single 1-D tensor."""
    return torch.cat([p.detach().flatten() for p in model.parameters()])
