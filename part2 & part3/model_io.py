"""
Helpers for loading trained models from per-epoch state_dict checkpoints.

State_dict files (.pt) include BatchNorm running_mean / running_var buffers,
which the flat-params .npy files do not. All downstream evaluation scripts
(perturbation, mode connectivity, function-space, representation CKA,
loss-matched geometry) must use these state_dict files for correct eval-mode
forward passes.
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


def load_final_model(arch_name: str, opt_name: str, seed: int,
                     device: torch.device) -> nn.Module:
    """Load the trained model for (arch, opt, seed) at the final epoch."""
    state_path = os.path.join(
        RESULTS_DIR, f"{arch_name}_{opt_name}_seed{seed}_state.pt"
    )
    return _load_from_path(arch_name, state_path, device)


def load_epoch_model(arch_name: str, opt_name: str, seed: int, epoch: int,
                     device: torch.device) -> nn.Module:
    """Load the model at end of `epoch` for (arch, opt, seed). epoch=0 is init."""
    state_path = os.path.join(
        RESULTS_DIR, CKPT_SUBDIR,
        f"{arch_name}_{opt_name}_seed{seed}",
        f"epoch_{epoch:02d}.pt",
    )
    return _load_from_path(arch_name, state_path, device)


def _load_from_path(arch_name: str, state_path: str,
                    device: torch.device) -> nn.Module:
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing state_dict: {state_path}")
    model = build_model(arch_name).to(device)
    sd = torch.load(state_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def get_flat_params_from_state(state_dict, model: nn.Module) -> torch.Tensor:
    """
    Reconstruct the flat parameter vector from a state_dict in the same order
    as `model.parameters()`. Buffers (BN running stats) are excluded — same
    convention as `models.get_flat_params(model)`.
    """
    pieces = []
    for name, p in model.named_parameters():
        if name in state_dict:
            pieces.append(state_dict[name].flatten())
        else:
            raise KeyError(f"state_dict missing parameter: {name}")
    return torch.cat(pieces)
