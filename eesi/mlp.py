"""Time-conditioned MLP field on R^d.

A minimal velocity/score backbone for the `EESI` model when each sample is a
single d-dimensional vector (no particle structure). The scalar time `t` is
concatenated per sample with the coordinates and passed through a plain MLP.

Forward:
    t [B] or scalar, x [B, d] -> [B, d]
"""
from __future__ import annotations

from typing import Callable, Union

import torch
from torch import nn


# Named activations, resolved to a fresh module instance per layer.
_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}

# An activation spec is either a registered name or a factory returning a module.
Activation = Union[str, Callable[[], nn.Module]]


def _resolve_activation(activation: Activation) -> Callable[[], nn.Module]:
    """Return a factory producing a fresh activation module for each layer."""
    if isinstance(activation, str):
        key = activation.lower()
        if key not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)} or a callable, "
                f"got {activation!r}"
            )
        return _ACTIVATIONS[key]
    if callable(activation):
        return activation
    raise TypeError(f"activation must be a str or callable, got {type(activation).__name__}")


class TimeMLP(nn.Module):
    """Time-conditioned MLP mapping (t, x) -> a d-dimensional field value.

    Args:
        d: input/output dimension.
        hidden: hidden width (default 128).
        n_layers: number of hidden layers (default 3).
        activation: hidden-layer nonlinearity — a name from `_ACTIVATIONS`
            (e.g. "silu" (default), "relu", "gelu", "tanh") or a callable
            returning a fresh `nn.Module` (e.g. `nn.SiLU` or
            `lambda: nn.LeakyReLU(0.1)`).
    """

    def __init__(
        self,
        d: int,
        hidden: int = 128,
        n_layers: int = 3,
        activation: Activation = "silu",
    ):
        super().__init__()
        self.d = d
        act = _resolve_activation(activation)
        layers: list[nn.Module] = [nn.Linear(d + 1, hidden), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, d))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be [B, d]; got {tuple(x.shape)}")
        B, d = x.shape
        if d != self.d:
            raise ValueError(f"d mismatch: model d={self.d}, input d={d}")

        t_f = t if t.is_floating_point() else t.float()
        if t_f.dim() == 0:
            t_col = t_f.expand(B, 1)
        else:
            t_col = t_f.view(B, 1)
        t_col = t_col.to(x.dtype)

        return self.net(torch.cat([x, t_col], dim=-1))
