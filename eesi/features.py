"""Feature expansions for network inputs, shared by more than one system.

    fourier_expand      [cos(kx), sin(kx)] for k = 1..order
    cos_expand          the even half, [cos(kx)] alone
    half_period_expand  the same on [0, 1] with period 2, not 1
    time_features       non-periodic features of the interpolant time

All pure functions of a scalar field: no parameters, no module state, no learned
weights. They exist to widen a raw scalar into something a single `nn.Linear`
embedding can actually condition on -- a network handed one number has to
manufacture its own nonlinear basis out of a ramp before it can resolve nearby
values, and these hand it that basis instead.

They sit in the core rather than in a system subpackage for the same reason
`eesi.egnn` does: more than one system uses them -- the 1D XY chain's edge and time
features, and the tangentially active polymer's node features -- and `eesi.systems`
forbids one system importing from another. `eesi.systems.xy.gnn` re-exports all
three of the names it used to define, so importing them from there still works.

Conventions:
    Input `x` is [...] with NO trailing feature axis; output appends one.
    `order = 0` returns a [..., 0] tensor rather than raising, so a caller can
    `cat` the expansion on unconditionally and let the order control whether it
    contributes anything.
"""
from __future__ import annotations

import math

import torch


def fourier_expand(x: torch.Tensor, order: int) -> torch.Tensor:
    """Fourier features of a scalar field.

    Args:
        x: [...] scalar values (no trailing feature axis).
        order: number of harmonics n >= 1.

    Returns:
        [..., 2*order] = [cos(x), ..., cos(order*x), sin(x), ..., sin(order*x)].
    """
    k = torch.arange(1, order + 1, device=x.device, dtype=x.dtype)
    ang = x.unsqueeze(-1) * k          # [..., order]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


def cos_expand(x: torch.Tensor, order: int) -> torch.Tensor:
    """Cosine-only Fourier features -- EVEN under x -> -x.

    The even half of `fourier_expand`. Used for the edge angle differences so the
    per-edge scalar `phi` is even in the configuration, which is exactly the
    condition for `trans = d_theta * phi` to be ODD, i.e. for the network to be
    equivariant under the spin reflection theta -> -theta (review Sec. 6.3).

    No expressivity is lost within the correct hypothesis class: `d_theta *
    phi(cos d_theta, cos 2 d_theta, ...)` already spans the odd 2*pi-periodic
    functions of d_theta (e.g. sin(d) is phi = sin(d)/d, which is even, smooth,
    and continuous at the +-pi seam).

    Args:
        x: [...] scalar values (no trailing feature axis).
        order: number of harmonics n >= 1.

    Returns:
        [..., order] = [cos(x), ..., cos(order*x)].
    """
    k = torch.arange(1, order + 1, device=x.device, dtype=x.dtype)
    return (x.unsqueeze(-1) * k).cos()


def half_period_expand(s: torch.Tensor, order: int) -> torch.Tensor:
    """Half-period Fourier features of a scalar on [0, 1].

    HALF-period, not full, and that is the whole point: `fourier_expand(2*pi*s, K)`
    has period exactly 1 and therefore maps s = 0 and s = 1 to the SAME vector. For
    an interpolant time that collapses the two endpoints, where the true drift and
    score differ completely (`plans/XY_MODEL_REVIEW.md`, F1, and see
    `time_features`); for a normalized chain index it collapses the tail with the
    head, which for a DIRECTED polymer is the one distinction the feature exists to
    make. Period 2 has neither problem: cos(k*pi*s) alternates between +1 and -1 at
    the two ends, so the cosine channels alone already separate them.

    The sine channels vanish at both endpoints (sin(k*pi*0) = sin(k*pi*1) = 0), so
    they add no endpoint discrimination -- they add interior resolution, halving the
    spacing at which two nearby s are told apart for a given number of harmonics.

    Args:
        s: [...] scalar values on [0, 1] (no trailing feature axis).
        order: number of harmonics n >= 0.

    Returns:
        [..., 2*order] = [cos(pi*s), ..., cos(order*pi*s),
                          sin(pi*s), ..., sin(order*pi*s)].
        order = 0 gives a [..., 0] tensor, so callers can `cat` unconditionally.
    """
    return fourier_expand(math.pi * s, order)


def time_features(t: torch.Tensor, order: int = 0, w_max: float = 30.0) -> torch.Tensor:
    """Non-periodic features of the interpolant time.

    `t` itself is always the first channel, and on its own it is already
    injective. That is the whole point: the previous embedding was
    `fourier_expand(2*pi*t, K)`, which is exactly 1-periodic and therefore maps
    t=0 and t=1 to the SAME vector, leaving the network structurally incapable of
    separating the two endpoints -- where the true drift and score differ
    completely (plans/XY_MODEL_REVIEW.md, F1).

    Args:
        t: [...] times in [0, 1] (no trailing feature axis).
        order: number of log-spaced frequency pairs appended to raw `t`. 0 (the
            default) is raw `t` alone, matching the LJ13 EGNN's `h = ones * t`.
            The frequencies are deliberately NOT harmonics of 2*pi, so distinct
            t in [0, 1] keep distinct embeddings for any order.
        w_max: largest angular frequency, when order >= 1.

    Returns:
        [..., 1 + 2*order].
    """
    if order < 1:
        return t.unsqueeze(-1)
    w = torch.logspace(0.0, math.log10(w_max), order, device=t.device, dtype=t.dtype)
    a = t.unsqueeze(-1) * w            # [..., order]
    return torch.cat([t.unsqueeze(-1), a.cos(), a.sin()], dim=-1)
