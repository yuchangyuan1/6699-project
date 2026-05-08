"""
Fashion-MNIST loaders. The seed argument controls the train shuffle generator,
so SGD and Adam runs sharing a seed see an identical minibatch order.
"""
import torch
from torchvision import datasets, transforms


FMNIST_MEAN = 0.2860
FMNIST_STD  = 0.3530

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((FMNIST_MEAN,), (FMNIST_STD,)),
])


def get_loaders(seed: int, batch_size: int = 128, data_root: str = "./data"):
    train_set = datasets.FashionMNIST(data_root, train=True,  download=True, transform=_transform)
    test_set  = datasets.FashionMNIST(data_root, train=False, download=True, transform=_transform)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        generator=g, num_workers=0, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=0,
    )
    return train_loader, test_loader
