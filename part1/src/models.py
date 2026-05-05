"""MLP and SmallCNN model definitions + optimizer factory."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims=None, num_classes: int = 10, use_bn: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, use_bn: bool = True):
        super().__init__()
        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def build_model(model_name: str, **kwargs) -> nn.Module:
    registry = {'mlp': MLP, 'small_cnn': SmallCNN}
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(registry)}")
    return registry[model_name](**kwargs)


def build_optimizer(optimizer_name: str, model: nn.Module, opt_cfg: dict) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg['lr'],
            momentum=opt_cfg.get('momentum', 0.9),
            weight_decay=opt_cfg.get('weight_decay', 0.0),
        )
    elif name == 'adam':
        betas = opt_cfg.get('betas', [0.9, 0.999])
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg['lr'],
            betas=tuple(betas),
            eps=opt_cfg.get('eps', 1e-8),
            weight_decay=opt_cfg.get('weight_decay', 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
