"""Time-conditioned MLP field on R^d.

A minimal velocity/score backbone for the `EESI` model when each sample is a
single d-dimensional vector (no particle structure). The scalar time `t` is
concatenated per sample with the coordinates and passed through a plain MLP.

Forward:
    t [B] or scalar, x [B, d] -> [B, d]

`ConstrainedMLP` wraps this backbone with a learned cubic restoring asymptote
and a t·(1-t) endpoint gate, so the drift satisfies b(0,·)=b(1,·)=0 exactly.
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

    def _time_column(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Normalize a scalar / [B] time tensor to a [B, 1] column matching x."""
        B = x.shape[0]
        t_f = t if t.is_floating_point() else t.float()
        if t_f.dim() == 0:
            t_col = t_f.expand(B, 1)
        else:
            t_col = t_f.view(B, 1)
        return t_col.to(x.dtype)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be [B, d]; got {tuple(x.shape)}")
        B, d = x.shape
        if d != self.d:
            raise ValueError(f"d mismatch: model d={self.d}, input d={d}")

        t_col = self._time_column(t, x)
        return self.net(torch.cat([x, t_col], dim=-1))


class ConstrainedMLP(TimeMLP):
    """Drift field with a built-in restoring asymptote and endpoint gate.

    Wraps a raw `TimeMLP` field ``N(t, x)`` and returns

        b(t, x) = t·(1 - t) · ( -s · (x - c)·‖x - c‖² + N(t, x) )

    Two structural priors are baked in:

    - **Cubic restoring asymptote** ``-s·(x - c)·‖x - c‖²`` (with ‖·‖² the
      squared L2 norm). For large ‖x - c‖ this dominates and points back toward
      the learned center ``c`` with magnitude ~‖x - c‖³, so the network only has
      to learn the *correction* inside the data region.
    - **Endpoint gate** ``t·(1 - t)``, which forces b(0,·) = b(1,·) = 0 exactly.
      This matches interpolant schedules whose total velocity vanishes at both
      endpoints (e.g. path="trig2" with gamma="sin2"); pairing it with a schedule
      that has nonzero endpoint velocity (e.g. gamma="quad", γ̇(0)=1) would make
      the gate fight the training target. Intended for the drift field `net_b`.

    The center ``c`` (a d-vector) and drift scale ``s`` are learned. ``s`` is
    parameterized as ``softplus(raw)`` so it stays positive and the asymptote
    always restores rather than repels.

    Args:
        d, hidden, n_layers, activation: forwarded to `TimeMLP` for the raw net N.
        center_init: initial value for the center c (float, broadcast to all d
            dims, or a [d] tensor). Default 0.0.
        drift_scale_init: initial value for the (positive) drift scale s.
            Default 1.0.
    """

    def __init__(
        self,
        d: int,
        hidden: int = 128,
        n_layers: int = 3,
        activation: Activation = "silu",
        center_init: Union[float, torch.Tensor] = 0.0,
        drift_scale_init: float = 1.0,
    ):
        super().__init__(d, hidden=hidden, n_layers=n_layers, activation=activation)

        center = torch.as_tensor(center_init, dtype=torch.get_default_dtype())
        center = center.expand(d).clone() if center.dim() == 0 else center.reshape(d).clone()
        self.center = nn.Parameter(center)

        if drift_scale_init <= 0.0:
            raise ValueError(f"drift_scale_init must be positive, got {drift_scale_init}")
        # softplus(raw) = drift_scale_init  ->  raw = log(exp(scale) - 1)
        raw = math.log(math.expm1(drift_scale_init))
        self._drift_scale_raw = nn.Parameter(torch.tensor(raw))

    @property
    def drift_scale(self) -> torch.Tensor:
        """Effective (positive) drift scale s = softplus(raw)."""
        return nn.functional.softplus(self._drift_scale_raw)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raw = super().forward(t, x)  # N(t, x); also validates x shape / dim
        t_col = self._time_column(t, x)

        xc = x - self.center
        r2 = xc.square().sum(dim=-1, keepdim=True)
        asymptote = -self.drift_scale * xc * r2

        gate = t_col * (1.0 - t_col)
        #return gate * (asymptote + raw)
        return gate * raw
