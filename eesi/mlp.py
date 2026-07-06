"""Time-conditioned MLP field on R^d.

A minimal velocity/score backbone for the `EESI` model when each sample is a
single d-dimensional vector (no particle structure). The scalar time `t` is
concatenated per sample with the coordinates and passed through a plain MLP.

Forward:
    t [B] or scalar, x [B, d] -> [B, d]
"""
from __future__ import annotations

import torch
from torch import nn


class TimeMLP(nn.Module):
    """SiLU MLP that maps (t, x) -> a d-dimensional field value.

    Args:
        d: input/output dimension.
        hidden: hidden width (default 128).
        n_layers: number of hidden layers (default 3).
    """

    def __init__(self, d: int, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.d = d
        layers: list[nn.Module] = [nn.Linear(d + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
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
