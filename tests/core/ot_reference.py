"""Reference (scipy) implementation of the equivariant-OT coupling. Slow and obviously correct.

This module exists to be right, not to be fast. It is never used in training -- it is
the oracle that `eesi.ot` (the vectorized GPU path) is validated against. Every function
here is a plain Python loop over pairs, using scipy's `linear_sum_assignment` and
`Rotation.align_vectors` directly.

Ported from SemlaFlow's `semlaflow/data/interpolate.py` (`_ot_map`, `_match_mols`,
`_match_cost`), dropping `GeometricMol`, padding, and truncation, since LJ13 is a
homogeneous point cloud of fixed size.

Fidelity notes, all load-bearing:
  * Permutation FIRST, then rotation. Not the reverse. Klein's approximation is
    sequential, not alternating and not iterated (Klein et al. 2023, App. A.9).
  * The rotation is PROPER (SO(3)); `Rotation.align_vectors` guarantees det = +1.
    O(3) is available via `proper=False`, which is a deviation from Klein Eq. 16.
  * We return the ALIGNED configurations, not just the costs. Computing a cost with
    alignment and then interpolating the unaligned pair is the classic bug here.
  * We transform the NOISE (x0), never the data. See `eesi.ot` for why that is only
    valid when both marginals are G-invariant.
  * Reduction is `sum` (Klein's ||.||^2). SemlaFlow uses `mean`, which for fixed N is
    a uniform rescale of the cost matrix and cannot change any assignment -- but it
    does change the reported numbers. Keep `sum` consistent with `eesi.ot`.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().double().numpy()


def center(x: torch.Tensor) -> torch.Tensor:
    """Remove the centre of mass. x: (..., N, 3)."""
    return x - x.mean(dim=-2, keepdim=True)


# ---- single pair -----------------------------------------------------------


def match_pair(x0: np.ndarray, x1: np.ndarray, proper: bool = True):
    """Align noise x0 onto data x1 over S(N) x SO(3). Both (N, 3), centred.

    Returns (x0_aligned, cost) with cost = ||x0_aligned - x1||^2.
    """
    # 1. permutation: Hungarian on the N x N inter-particle squared distances
    d = x0[:, None, :] - x1[None, :, :]              # (N, N, 3)
    D = (d ** 2).sum(-1)                             # D[a, b] = |x0[a] - x1[b]|^2
    r, c = linear_sum_assignment(D)
    sigma = np.empty(len(r), dtype=int)
    sigma[c] = r                                     # x0 particle feeding x1 site b
    x0p = x0[sigma]

    # 2. rotation: Kabsch. align_vectors(a, b) minimizes ||a - C b||^2, so a=x1, b=x0p.
    if proper:
        R, _ = Rotation.align_vectors(x1, x0p)
        x0a = R.apply(x0p)
    else:
        # O(3): plain SVD with no determinant correction (deviation from Klein).
        U, _, Vt = np.linalg.svd(x1.T @ x0p)
        x0a = x0p @ (U @ Vt).T

    return x0a, float(((x0a - x1) ** 2).sum())


# ---- batch -----------------------------------------------------------------


def cost_matrix(x0: torch.Tensor, x1: torch.Tensor, proper: bool = True):
    """Full B x B aligned-cost matrix and the aligned noise for every pair.

    x0, x1: (B, N, 3). Returns (M, aligned) with M (B, B) numpy and
    aligned (B, B, N, 3) numpy, where aligned[i, j] is x0_i aligned onto x1_j.
    """
    a, b = _np(center(x0)), _np(center(x1))
    B, N, _ = a.shape
    M = np.empty((B, B))
    aligned = np.empty((B, B, N, 3))
    for i in range(B):
        for j in range(B):
            aligned[i, j], M[i, j] = match_pair(a[i], b[j], proper=proper)
    return M, aligned


def ot_map(x0: torch.Tensor, x1: torch.Tensor, proper: bool = True,
           align: bool = True, batch: bool = True):
    """Equivariant-OT coupling of a batch. x0, x1: (B, N, 3) -> (x0_out, x1_out).

    `align` toggles the group-alignment layer (OT over S(N) x SO(3), filling M).
    `batch` toggles the minibatch-OT layer (OT over the batch, choosing pairings).
    The 2x2 of these two flags is the ablation in EQOT_PLAN.md; the full coupling is
    align=True, batch=True.

    x1 is returned unchanged and in place -- only the noise is transformed and
    reordered. Returned as torch tensors matching x0's dtype/device.
    """
    B = x0.shape[0]
    x0c, x1c = center(x0), center(x1)

    if align:
        M, aligned = cost_matrix(x0c, x1c, proper=proper)
    else:
        a, b = _np(x0c), _np(x1c)
        # unaligned squared distance, and the "aligned" noise is just the noise
        M = ((a[:, None] - b[None, :]) ** 2).sum((-1, -2))
        aligned = np.broadcast_to(a[:, None], (B, B, *a.shape[1:]))

    if batch:
        r, c = linear_sum_assignment(M)
    else:
        r = c = np.arange(B)                          # identity pairing

    out = np.empty_like(_np(x1c))
    out[c] = aligned[r, c]                            # out[j] = noise assigned to x1_j
    x0_out = torch.as_tensor(out, dtype=x0.dtype, device=x0.device)
    return x0_out, x1c


def transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> float:
    """Mean per-sample ||x0 - x1||^2 of an already-coupled pair."""
    return float(((x0 - x1) ** 2).sum(dim=(-1, -2)).mean())


# ---- XY chain: O(2) x Z2^site ----------------------------------------------
#
# The oracle for Part C. Note the deliberate asymmetry with `eesi.ot.xy_ot_couple`:
# the rotation here is found by DENSE GRID SEARCH over phi, not by the closed form
# phi* = atan2(S, C). A closed form validated against itself proves nothing. The
# discrete part is likewise an explicit nested Python loop, not a stacked argmin.


def _wrap(d: np.ndarray) -> np.ndarray:
    """Wrap angles into (-pi, pi]."""
    return d - 2.0 * np.pi * np.round(d / (2.0 * np.pi))


def xy_chordal_cost(x0: np.ndarray, x1: np.ndarray) -> float:
    """sum_i (1 - cos(x0_i - x1_i)). The group-free chordal distance."""
    return float((1.0 - np.cos(x0 - x1)).sum())


def xy_match_pair(x0: np.ndarray, x1: np.ndarray, reflect: bool = True,
                  negate: bool = True, n_grid: int = 10_000):
    """Align noise x0 onto data x1 over O(2) x Z2^site. Both (N,).

    Returns (x0_aligned, cost). The site reversal is i -> N-1-i, the spin flip is
    theta -> -theta, and the rotation is a global phase found on a dense grid.
    """
    best = None
    for r in ((0, 1) if reflect else (0,)):
        for s in ((1.0, -1.0) if negate else (1.0,)):
            u = s * (x0[::-1] if r else x0)
            d = x1 - u
            phis = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
            costs = (1.0 - np.cos(phis[:, None] - d[None, :])).sum(1)
            k = int(costs.argmin())
            if best is None or costs[k] < best[1]:
                best = (_wrap(u + phis[k]), float(costs[k]))
    return best


def xy_ot_map(x0: torch.Tensor, x1: torch.Tensor, align: bool = True,
              batch: bool = True, reflect: bool = True, negate: bool = True):
    """Equivariant-OT coupling for XY chains. x0, x1: (B, N) -> (x0_out, x1).

    Same `align` / `batch` semantics as `ot_map`. Only the noise is transformed.
    """
    a, b = _np(x0), _np(x1)
    B, N = a.shape
    M = np.empty((B, B))
    aligned = np.empty((B, B, N))
    for i in range(B):
        for j in range(B):
            if align:
                aligned[i, j], M[i, j] = xy_match_pair(a[i], b[j], reflect=reflect,
                                                       negate=negate)
            else:
                aligned[i, j], M[i, j] = a[i], xy_chordal_cost(a[i], b[j])

    r, c = linear_sum_assignment(M) if batch else (np.arange(B), np.arange(B))
    out = np.empty_like(a)
    out[c] = aligned[r, c]
    return torch.as_tensor(out, dtype=x0.dtype, device=x0.device), x1


def xy_transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> float:
    """Mean per-sample sum_i (1 - cos(x0_i - x1_i)) of an already-coupled pair."""
    return float((1.0 - torch.cos(x0 - x1)).sum(-1).mean())


# ---- TAP chain: O(3), tail-anchored ----------------------------------------
#
# The oracle for the tangentially active polymer. Two deliberate asymmetries with
# `eesi.systems.tap.ot`, both of them the point of having an oracle at all:
#
#   * The rotation comes from `scipy.linalg.orthogonal_procrustes` (a direct LAPACK
#     SVD) rather than from our eigvalsh-of-H^T-H singular values. Validating a closed
#     form against a rearrangement of itself proves nothing.
#   * NO CENTERING, and no permutation stage. Both omissions are the physics: the tail
#     is pinned at the origin so O(3) acts about it, and the chain is directed so
#     monomers may not be relabelled. A port of `match_pair` that kept either would be
#     caught here.


def tap_match_pair(x0: np.ndarray, x1: np.ndarray, proper: bool = False):
    """Align noise x0 onto data x1 over O(3), about the ORIGIN. Both (N, 3), anchored.

    Returns (x0_aligned, cost) with cost = ||x0_aligned - x1||^2.
    """
    if proper:
        # SO(3): align_vectors(a, b) minimizes ||a - C b||^2 with det C = +1.
        R, _ = Rotation.align_vectors(x1, x0)
        x0a = R.apply(x0)
    else:
        # O(3): orthogonal_procrustes(A, B) minimizes ||A R - B||_F over orthogonal R,
        # with no determinant constraint.
        R, _ = orthogonal_procrustes(x0, x1)
        x0a = x0 @ R
    return x0a, float(((x0a - x1) ** 2).sum())


def tap_cost_matrix(x0: torch.Tensor, x1: torch.Tensor, proper: bool = False):
    """Full B x B aligned-cost matrix and the aligned noise for every pair. (B, N, 3)."""
    a, b = _np(x0), _np(x1)
    B, N, _ = a.shape
    M = np.empty((B, B))
    aligned = np.empty((B, B, N, 3))
    for i in range(B):
        for j in range(B):
            aligned[i, j], M[i, j] = tap_match_pair(a[i], b[j], proper=proper)
    return M, aligned


def tap_ot_map(x0: torch.Tensor, x1: torch.Tensor, proper: bool = False,
               align: bool = True, batch: bool = True):
    """Equivariant-OT coupling for TAP chains. x0, x1: (B, N, 3) -> (x0_out, x1).

    Same `align` / `batch` semantics as `ot_map`. Only the noise is transformed, and
    nothing is centered.
    """
    a, b = _np(x0), _np(x1)
    B = a.shape[0]

    if align:
        M, aligned = tap_cost_matrix(x0, x1, proper=proper)
    else:
        M = ((a[:, None] - b[None, :]) ** 2).sum((-1, -2))
        aligned = np.broadcast_to(a[:, None], (B, B, *a.shape[1:]))

    r, c = linear_sum_assignment(M) if batch else (np.arange(B), np.arange(B))
    out = np.empty_like(a)
    out[c] = aligned[r, c]
    return torch.as_tensor(out, dtype=x0.dtype, device=x0.device), x1
