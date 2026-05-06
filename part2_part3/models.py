"""
Model definitions: Batch-normalized MLP and SmallCNN for Fashion-MNIST.
Architectures match the paper specification exactly.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Batch-normalized MLP:
      Linear(784, 256) -> BN -> ReLU
      Linear(256, 128) -> BN -> ReLU
      Linear(128, 10)
    """
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
    """
    Two conv blocks (Conv -> BN -> ReLU -> MaxPool), then:
      Linear(1568, 128) -> ReLU -> Linear(128, 10)
    Input: 28x28 grayscale.
    After block1: 32 x 14 x 14  (stride-1 conv, then 2x2 pool)
    After block2: 64 x  7 x  7  (stride-1 conv, then 2x2 pool)
    Flattened: 64 * 7 * 7 = 3136...

    Actually to get 1568 as in the paper: use 32 channels each block.
    After block1: 32 x 14 x 14
    After block2: 32 x  7 x  7 -> 32*7*7 = 1568. Correct.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32x14x14 -> 32x7x7
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
    """Return all parameters as a single 1-D tensor."""
    return torch.cat([p.detach().flatten() for p in model.parameters()])
