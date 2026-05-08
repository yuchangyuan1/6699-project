"""
Loaders for trained model checkpoints.

State_dict (.pt) files include BatchNorm running stats, so all eval-time scripts
(perturbation, mode connectivity, function space, CKA, loss-matched geometry)
must load through these helpers rather than from the flat-params .npy files.
"""
import os

import torch
import torch.nn as nn

from models import MLP, SmallCNN

ARCHITECTURES = {"MLP": MLP, "SmallCNN": SmallCNN}

RESULTS_DIR = "./results_part2"
CKPT_SUBDIR = "checkpoints"


def build_model(arch_name: str) -> nn.Module:
    return ARCHITECTURES[arch_name]()


def load_final_model(arch_name, opt_name, seed, device) -> nn.Module:
    state_path = os.path.join(
        RESULTS_DIR, f"{arch_name}_{opt_name}_seed{seed}_state.pt"
    )
    return _load_from_path(arch_name, state_path, device)


def load_epoch_model(arch_name, opt_name, seed, epoch, device) -> nn.Module:
    state_path = os.path.join(
        RESULTS_DIR, CKPT_SUBDIR,
        f"{arch_name}_{opt_name}_seed{seed}",
        f"epoch_{epoch:02d}.pt",
    )
    return _load_from_path(arch_name, state_path, device)


def _load_from_path(arch_name, state_path, device) -> nn.Module:
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing state_dict: {state_path}")
    model = build_model(arch_name).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    return model


def get_flat_params_from_state(state_dict, model: nn.Module) -> torch.Tensor:
    """Reconstruct the flat parameter vector in model.parameters() order, excluding BN buffers."""
    pieces = []
    for name, _ in model.named_parameters():
        if name not in state_dict:
            raise KeyError(f"state_dict missing parameter: {name}")
        pieces.append(state_dict[name].flatten())
    return torch.cat(pieces)
