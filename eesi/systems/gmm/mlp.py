"""Time-conditioned MLP field on R^d.

A minimal velocity/score backbone for the `EESI` model when each sample is a
single d-dimensional vector (no particle structure). The scalar time `t` is
lifted to a sinusoidal positional encoding, refined by a small MLP, and
concatenated per sample with the coordinates before the plain MLP trunk.

Forward:
    t [B] or scalar, x [B, d] -> [B, d]
"""
from __future__ import annotations

import math
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


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        x = (x + 1e-1).log() / 4
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class TimeMLP(nn.Module):
    """Time-conditioned MLP mapping (t, x) -> a d-dimensional field value.

    `t` is encoded by `timestep_embedding` into `hidden` channels, refined by a
    2-layer MLP of the same width, then concatenated with `x` as
    [time_embedding, x] ([B, hidden + d]) and fed to the trunk. Only the trunk's
    first layer is widened by the embedding; every later layer stays `hidden`
    wide and the output is [B, d].

    Args:
        d: input/output dimension.
        hidden: hidden width, also the time-embedding width (default 128).
        n_layers: number of hidden layers in the trunk (default 3).
        activation: hidden-layer nonlinearity — a name from `_ACTIVATIONS`
            (e.g. "silu" (default), "relu", "gelu", "tanh") or a callable
            returning a fresh `nn.Module` (e.g. `nn.SiLU` or
            `lambda: nn.LeakyReLU(0.1)`).
        max_period: frequency span of the time encoding, see
            `timestep_embedding`.
        time_scale: multiplier applied to `t` before the frequency ladder, see
            `timestep_embedding`.
    """

    def __init__(
        self,
        d: int,
        hidden: int = 128,
        n_layers: int = 3,
        activation: Activation = "silu",
        max_period: float = 10_000.0,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        self.d = d
        self.hidden = hidden
        act = _resolve_activation(activation)

        self.time_embed = PositionalEmbedding(self.hidden)

        # [B, hidden] sinusoidal encoding -> 2-layer MLP at the same width. No
        # trailing activation: the trunk applies one immediately.
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), act(), nn.Linear(hidden, hidden)
        )

        # Only the first trunk layer sees the widened [time_embedding, x] input.
        layers: list[nn.Module] = [nn.Linear(hidden + d, hidden), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, d))
        self.net = nn.Sequential(*layers)

    def _time_vector(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Normalize a scalar / [B] / [B, 1] time tensor to the [B] `time_embed` wants,
        in x's dtype -- `PositionalEmbedding` builds an outer product, so it accepts
        nothing else, and it would otherwise take its output dtype from `t`."""
        B = x.shape[0]
        t_f = t if t.is_floating_point() else t.float()
        t_v = t_f.expand(B) if t_f.dim() == 0 else t_f.reshape(B)
        return t_v.to(x.dtype)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be [B, d]; got {tuple(x.shape)}")
        B, d = x.shape
        if d != self.d:
            raise ValueError(f"d mismatch: model d={self.d}, input d={d}")

        t_emb = self.time_embed(self._time_vector(t, x))     # [B, hidden]
        t_emb = self.time_mlp(t_emb)                         # [B, hidden]
        return self.net(torch.cat([t_emb, x], dim=-1))       # [B, d]
