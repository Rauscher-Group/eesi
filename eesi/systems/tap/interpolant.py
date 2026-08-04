"""Stochastic interpolant for TAP chains on the tail-anchored subspace.

Holds `TAPEESI`, which differs from the general `eesi.EESI` only in keeping every
Gaussian it draws on the subspace {v : v_0 = 0} -- see the class docstring.
"""
from __future__ import annotations

import torch

from ...interpolant import EESI
from .dynamics import divergence as _subspace_divergence


class TAPEESI(EESI):
    """Stochastic interpolant for TAP chains on the tail-anchored subspace.

    The TAP analogue of `LJ13EESI`, for states of shape (B, N, 3) -- N-agnostic, so
    one class serves any chain length. The interpolant geometry is the plain Euclidean
    straight line (the subspace is flat and affine), so `_interpolant_sample` is
    inherited unchanged. The ONE specialisation is that every Gaussian this class draws
    must live on that subspace, which is achieved by overriding `_noise_like`.

    TAP lives on V = {x : x_0 = 0}, where both the prior p0 (the semiflexible chain)
    and the target p1 (the active steady state) are supported. All the interpolant needs
    of them is that they are supported on the same flat subspace, so no tangent-space /
    exp-map machinery is needed. Zeroing the latent's tail row is what keeps the whole
    path x_t = alpha x0 + beta x1 + gamma z on V (x0, x1, z all anchored, the map
    linear); an off-subspace z would drag the pinned tail away from the origin at
    intermediate t even with anchored endpoints.

    The latent is NOT a prior draw, and this is load-bearing rather than incidental.
    `_noise_like` returns an ISOTROPIC normal on the subspace: z is the interpolant's
    latent variable, whose job is to define the smoothing schedule gamma(t), and the
    denoising score target -z/gamma assumes exactly that isotropic law. The prior is a
    semiflexible chain and is not Gaussian at all, so routing `_noise_like` through
    `data.sample_prior` would not merely differ from LJ13 (where the prior happens to BE
    a subspace isotropic normal, and the two coincide) -- it would make the score target
    flatly wrong. Do not "fix" it.

    The latent is kept INDEPENDENT of the (OT-coupled) base on purpose: the one-sided
    equivalence that would let one fold alpha x0 + gamma z into a single Gaussian only
    holds for an uncoupled base, and `tap_ot_couple` aligns x0 to x1. With z
    independent, the antithetic denoising score stays exact.

    Divergence/entropy: `entropy_estimate(method="div")` (and `sample(entropy="div")`)
    trace net_b by Hutchinson with the same anchored draw, so the probes are
    v ~ N(0, P) and E[v^T J v] = tr(P J) is the divergence on the DOF = 3(N-1)
    subspace -- the estimator analogue of `eesi.systems.tap.dynamics.divergence`. Pair
    with two `TAPDynamics` fields, whose output already has a zero tail row.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force ISM off, regardless of what the caller passed -- same reasoning as
        # LJ13EESI. `learn_score` only matters for gamma="none", and ISM needs a
        # differentiable divergence, but this class traces net_b with
        # `_subspace_divergence`, which runs under no_grad (see `_divergence`).
        # Learning the score here would silently backprop through a detached graph.
        self.learn_score = False

    def _divergence(self, s: torch.Tensor, x_t: torch.Tensor, create_graph: bool = True,
                    t: torch.Tensor | None = None) -> torch.Tensor:
        """Exact subspace divergence of net_b, via `tap.dynamics.divergence`.

        Traces the velocity field over the DOF = 3(N-1) anchored directions with
        forward-mode jvp. It re-evaluates net_b at `t`, so `s` (the precomputed output)
        is unused, as is `create_graph`: the estimator is `@torch.no_grad()` and never
        differentiable (which is why `learn_score` is forced off; see `__init__`).
        """
        return _subspace_divergence(self.net_b, t, x_t)

    def _noise_like(self, ref: torch.Tensor) -> torch.Tensor:
        """An anchored standard-normal draw shaped like `ref` (B, N, 3).

        Zeroes the tail row, projecting the draw onto the tangent subspace. Serves
        every Gaussian the base class samples: the latent z, the SDE diffusion term,
        and the Hutchinson probes.
        """
        g = torch.randn_like(ref)
        g[..., 0, :] = 0.0
        return g
