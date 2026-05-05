"""Fashion-MNIST data loading with deterministic shuffling."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _worker_init_fn(worker_id: int, seed: int) -> None:
    np.random.seed(seed * 100 + worker_id)
    torch.manual_seed(seed * 100 + worker_id)


def get_fashion_mnist_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple:
    """Return (train_loader, test_loader) for Fashion-MNIST."""
    mean, std = (0.2860,), (0.3530,)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=lambda wid: _worker_init_fn(wid, seed),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
