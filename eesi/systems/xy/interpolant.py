"""Stochastic interpolant for the 1D XY chain: the periodic-angle specialisation.

Holds `xyEESI`, which differs from the general `eesi.EESI` only in the geometry
of the interpolant path -- see the class docstring.
"""
from __future__ import annotations

import math

import torch

from ...interpolant import EESI


def _min_image(d: torch.Tensor) -> torch.Tensor:
    """Minimum-image angle difference, wrapped into (-pi, pi].

    Matches the convention in `eesi.systems.xy.data`:
    d - 2*pi*rint(d / 2*pi). Used by `xyEESI` to interpolate along the shortest
    geodesic on the periodic angle manifold (S^1)^N.
    """
    two_pi = 2.0 * math.pi
    return d - two_pi * torch.round(d / two_pi)



class xyEESI(EESI):
    """Stochastic interpolant on the periodic angle manifold (S^1)^N.

    Identical to `EESI` in every objective (drift/score losses, entropy
    estimators, samplers) but with a periodicity-aware interpolant, achieved by
    overriding the single geometry-aware hook `_interpolant_sample`. Rather than
    the Euclidean straight line `alpha·x0 + beta·x1`, which can cross the 0/2*pi
    seam and produce spuriously large velocity targets, `xyEESI` interpolates
    along the minimum-image geodesic. With `d = min_image(x1 - x0)` (each
    component wrapped into (-pi, pi]):

        x_t      = wrap(x0 + beta(t)·d + gamma(t)·z)   # position, wrapped onto (S^1)^N
        b_target = beta'(t)·d + gamma'(t)·z            # tangent-space velocity dx/dt
        s_target = -z / gamma(t)                       # tangent-space conditional score

    The latent noise `z` is added in the tangent space and the sampled point is
    wrapped back onto the manifold before it reaches the network. The path's
    `alpha` is unused: the geodesic is parameterised by `beta`, which runs 0 -> 1
    for the `linear`/`trig`/`trig2` paths (`linear` gives exactly `x0 + t·d`).

    The predicted velocity/score live in the tangent space R^N, so pair `xyEESI`
    with a periodicity-aware network such as `XYChainGNN`, whose output is a
    per-node tangent scalar. Note the endpoints are only recovered modulo 2*pi
    (x_t at t=1 equals x1 up to wrapping), which is exactly the manifold identity.

    The ODE/SDE samplers are inherited unchanged; because a periodicity-aware
    network is invariant to wrapping its input, integrated states may drift off
    (-pi, pi] but represent the same manifold points — wrap the final samples if a
    canonical representative is wanted.
    """

    def _interpolant_sample(
        self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor
    ):
        """Geodesic interpolant position and tangent-space targets (see class doc)."""
        self._assert_matched(x0, x1)
        _, beta, _, beta_dot = self._path(t)
        g, g_dot = self._scaled_gamma(t)
        d = _min_image(x1 - x0)
        x_t = _min_image(x0 + beta * d + g * z)
        b_target = beta_dot * d + g_dot * z
        s_target = -z / g.clamp_min(1e-12)
        return x_t, b_target, s_target

