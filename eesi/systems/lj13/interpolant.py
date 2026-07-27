"""Stochastic interpolant for LJ13-type point clouds on the COM-free subspace.

Holds `LJ13EESI`, which differs from the general `eesi.EESI` only in keeping
every Gaussian it draws on the mean-zero subspace -- see the class docstring.
"""
from __future__ import annotations

import torch

from ...interpolant import EESI
from .dynamics import divergence as _subspace_divergence


class LJ13EESI(EESI):
    """Stochastic interpolant for LJ13-type point clouds on the COM-free subspace.

    The LJ13 analogue of `xyEESI`, for states of shape (B, N, 3) -- N-agnostic, so
    the same class serves 13 particles today and larger clusters later. The
    interpolant geometry is the plain Euclidean straight line (the mean-zero
    subspace is flat), so `_interpolant_sample` is inherited unchanged. The ONE
    specialisation is that every Gaussian this class draws must live on that
    subspace, which is achieved by overriding the single `_noise_like` hook.

    Why that is the whole story (see plans/LJ13_SI_PLAN.md and the `eesi.systems.lj13.ot`
    docstring): LJ13 lives on V = {x : sum_i x_i = 0}, where both the prior p0 (a
    COM-free Gaussian) and the target p1 (the LJ13 Boltzmann law) are supported.
    The base x0 and the latent z are then the same kind of object -- centered
    Gaussians in a *flat* subspace -- so no tangent-space / exp-map machinery is
    needed, unlike a curved manifold. Centering the latent is what keeps the whole
    path x_t = alpha x0 + beta x1 + gamma z on V (x0, x1, z all mean-zero, the map
    linear); an off-subspace z would inject a spurious center-of-mass at
    intermediate t even with mean-zero endpoints.

    The latent is kept INDEPENDENT of the (OT-coupled) base on purpose: the
    one-sided equivalence that would let one fold alpha x0 + gamma z into a single
    Gaussian only holds for an uncoupled base, and `equivariant_ot_couple` aligns
    x0 to x1. With z independent, the antithetic denoising score stays exact.

    Divergence/entropy: `entropy_estimate(method="div")` (and `sample(entropy="div")`)
    trace net_b by Hutchinson with the
    same centered draw, so the probes are v ~ N(0, P) and E[v^T J v] = tr(P J) is
    the divergence on the DOF = (N-1)*3 subspace -- the estimator analogue of
    `eesi.systems.lj13.dynamics.divergence`. Pair with two `LJ13Dynamics` fields,
    whose output is already mean-free.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force ISM off, regardless of what the caller passed. `learn_score` only
        # matters for gamma="none" (the base uses it to gate implicit score
        # matching), and ISM needs a differentiable divergence -- but this class
        # traces net_b with `_subspace_divergence`, which runs under no_grad (see
        # `_divergence`). Learning the score here would silently backprop through a
        # detached graph, so it is disabled outright.
        self.learn_score = False

    def _divergence(self, s: torch.Tensor, x_t: torch.Tensor, create_graph: bool = True,
                    t: torch.Tensor | None = None) -> torch.Tensor:
        """Exact subspace divergence of net_b, via `lj13_dynamics.divergence`.

        Traces the velocity field over the DOF = (N-1)*3 COM-free directions with
        forward-mode jvp -- the exact analogue of the base class's Hutchinson
        estimator, and the same routine used for the free-energy log-det. It
        re-evaluates net_b at `t`, so `s` (the precomputed output) is unused, as is
        `create_graph`: the estimator is `@torch.no_grad()` and never differentiable
        (which is why `learn_score` is forced off; see `__init__`). The live
        callers are `entropy_estimate(method="div")` and `sample(entropy="div")`.
        """
        return _subspace_divergence(self.net_b, t, x_t)

    def _noise_like(self, ref: torch.Tensor) -> torch.Tensor:
        """A COM-free standard-normal draw shaped like `ref` (B, N, 3).

        Removes the per-configuration center of mass (mean over the particle axis),
        projecting the draw onto the mean-zero subspace. Serves every Gaussian the
        base class samples: the latent z, the SDE diffusion term, and the Hutchinson
        probes.
        """
        g = torch.randn_like(ref)
        return g - g.mean(dim=-2, keepdim=True)
