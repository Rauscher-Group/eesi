"""Equivariant-OT coupling for XY chains over O(2) x Z2^site.

Vectorized, GPU-ready, and validated against `tests/ot_reference.py` (scipy CPU
version); see `tests/xy/test_ot.py`. Runs under `no_grad`: the coupling is a
data-pairing step and the regression loss sees the aligned pair as fixed
targets, so we never backprop through it.

The `align` / `batch` flags mean what they mean in `eesi.systems.lj13.ot`:
`align` optimizes over the symmetry group, `batch` over the batch assignment.
The coupling transforms x0 and returns x1 as-is, which is marginal-preserving
because both p0 (iid uniform on (-pi, pi]) and p1 (the open-chain Boltzmann law)
are G-invariant.

Note that `xy_ot_couple` optimizes over O(2) x Z2^site and NOT over S(N): the open XY
chain's nearest-neighbour energy is not permutation-invariant, so permuting the noise
would disrupt the interpolant/marginals. (`XYChainGNN` is not permutation-equivariant
either.)
"""
from __future__ import annotations

import torch

from ...ot import _outer_assignment
from .gnn import angle_wrap


# ---- XY chain: O(2) x Z2^site ----------------------------------------------
#
# Structurally the same as the LJ13 path, but SIMPLER, not harder:
#
#   * No Hungarian. The discrete part is small enough to enumerate EXACTLY -- four
#     elements, each with a closed-form optimal rotation. Klein's sequential
#     Hungarian->Kabsch approximation exists because S(N) x SO(3) cannot be enumerated;
#     this can. A welcome consequence: the alignment map is exactly equivariant here,
#     which the LJ13 one is not (see tests/test_ot.py::test_rotating_x1_alone_is_not_invariant).
#   * No SVD. The U(1) alignment is the circular mean, the S^1 analogue of Kabsch.
#
# The group, G = O(2) x Z2^site, with O(2) = U(1) semidirect Z2^spin:
#     global rotation phi:  theta_i -> theta_i + phi     energy depends only on differences
#     spin flip n:          theta_i -> -theta_i          d(theta') = -d(theta), cos is even
#     site reversal r:      theta_i -> theta_{N-1-i}     d(theta') = -d(theta), cos is even
# All three are exact symmetries of the open-chain NN energy J*sum_i cos(theta_{i+1}-theta_i),
# and all three are respected by XYChainGNN (the spin flip since plans/XY_FIX_PLAN.md Phase 2;
# tests/test_xygnn.py::test_spin_reflection_equivariance). The discrete part is
# Z2^spin x Z2^site = 4 elements, enumerated in `_xy_group_elements`; the continuous U(1)
# is the closed form below. If the chain ever becomes PERIODIC the group grows to
# O(2) x D_N (N cyclic translations x reversal, 2N discrete elements) -- still exact
# enumeration, still no Hungarian: `_xy_group_elements` becomes the only thing to change.
#
# The cost is CHORDAL (1 - cos) rather than wrapped-geodesic (wrap(d)^2). This is a
# deliberate choice: it is what makes the rotation closed-form. Per pair,
# 1-cos(d) = 2 sin^2(d/2) is a monotone function of |wrap(d)| on [0, pi], so the two
# agree on per-pair ordering; they differ on sums. Note the divergence from the
# geodesic cost `xyEESI._interpolant_sample` uses for the interpolant itself.


def _xy_group_elements(x: torch.Tensor, reflect: bool = True, negate: bool = True):
    """The discrete Z2^site x Z2^spin orbit of the noise. Yields (B, N) tensors.

    Up to four elements, in a FIXED order -- (x, reversed, negated, both) -- so the
    `argmin` index returned by `xy_cost_matrix` and the survivor `pick` applied in
    `xy_ot_couple` refer to the same element as long as both are called with the same
    flags. That consistency is the reason this is the single place the discrete group
    is defined. A periodic chain would enumerate the 2*N cyclic permutations *
    inversion * spin flip here instead.

    Honestly, the only reason I didn't make the periodic chain was becasue I didn't
    want to mess around with transfer matrices for the exact solutions....

    Args:
        x: (B, N) angles.
        reflect: include the site reversal theta_i -> theta_{N-1-i}.
        negate: include the spin flip theta_i -> -theta_i. Setting it False is the
            ablation isolating O(2) down to U(1) semidirect nothing, i.e. SO(2).
    """
    g = [x]
    if reflect:
        g.append(x.flip(-1))
    if negate:
        g.append(-x)
    if reflect and negate:
        g.append(-x.flip(-1))
    return tuple(g)


@torch.no_grad()
def xy_cost_matrix(x0: torch.Tensor, x1: torch.Tensor, reflect: bool = True,
                   negate: bool = True):
    """Aligned B x B cost matrix over O(2) x Z2^site. x0, x1: (B, N).

    Returns (M, index) with M (B, B) the minimum chordal cost and `index` (B, B) long
    the index into `_xy_group_elements` achieving it. M[i, j] pairs noise i with data j.

    The cost is the chordal distance between each pair of angles. 
    For d = x1_j - rho(r)x0_i the wrapped angle differences (for each Z2 orbit), 
    the circular mean, phi*, is the angle that minimizes the sum of these distances:

        min_phi sum_l (1 - cos(phi - d_l)) = N - sqrt(S^2 + C^2),   phi* = atan2(S, C)

    where S = sum_l sin(d_l), and C = sum_l cos(d_l). This is nice because the 
    cost doesn't need an atan2 calculation; only the B pairs the OT selects need phi*
    to apply the alignment. This is conceptually the same as LJ13 `svdvals` trick above: 
    cheap invariant costs for the B^2 entries, full factorization only for B optimal pairs.

    Using sin/cos difference formulas avoids building the full (B, B, N) tensor
    of angle differences, becomes four matmuls per group element. Useful for large N.
    Going from 2 to 4 elements therefore costs two extra (B,N)@(N,B) matmul pairs --
    ~2.6 MFLOP at B=256, N=10, i.e. nothing next to the training step.
    """
    N = x0.shape[-1]
    s1, c1 = torch.sin(x1), torch.cos(x1)                      # (B, N)
    Ms = []
    for u in _xy_group_elements(x0, reflect, negate):
        su, cu = torch.sin(u), torch.cos(u)                    # (B, N)
        S = cu @ s1.T - su @ c1.T                              # (B, B) sum_l sin(x1-u)
        C = cu @ c1.T + su @ s1.T                              # (B, B) sum_l cos(x1-u)
        Ms.append(N - torch.hypot(S, C))
    M = torch.stack(Ms, 0)                                     # (n_g, B, B)
    return M.min(0).values, M.argmin(0)                        # (B,B), (B,B)


@torch.no_grad()
def _xy_apply_alignment(u: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Rotate noise onto data by the closed-form circular mean. u, x1: (B, N).

    The only place atan2 is called -- for the B survivors, never for the B^2 entries.
    """
    d = x1 - u                                                        # (B,N)
    phi = torch.atan2(torch.sin(d).sum(-1), torch.cos(d).sum(-1))     # (B,)
    return angle_wrap(u + phi.unsqueeze(-1))


@torch.no_grad()
def xy_ot_couple(x0: torch.Tensor, x1: torch.Tensor, align: bool = True,
                 batch: bool = True, reflect: bool = True, negate: bool = True):
    """Equivariant-OT coupling for 1D XY chains. x0, x1: (B, N) -> (x0_aligned, x1).

    `reflect=False` drops the site reversal, `negate=False` the spin flip; both False
    restricts to SO(2) alone. Those are the ablations isolating each discrete layer.
    """
    B = x0.shape[0]
    if align:
        M, index = xy_cost_matrix(x0, x1, reflect=reflect, negate=negate)
    else:
        M = (1.0 - torch.cos(x0[:, None] - x1[None, :])).sum(-1)      # (B, B) chordal
        index = None

    # minibatch OT on the costs of pre-aligned samples
    # r,c are shape (B,); r = row indices for each column, c = column indices for each row
    r, c = _outer_assignment(M) if batch else (torch.arange(B, device=x0.device),) * 2

    # tensor to perform indexing on x0
    inv = torch.empty(B, dtype=torch.long, device=x0.device)
    inv[c] = r                                     # inv[j] = noise index paired with x1_j
    x0_sel = x0[inv]                               # (B, N)
    if align:
        # apply each survivor's own discrete element, then its closed-form rotation
        g = _xy_group_elements(x0_sel, reflect, negate)
        pick = index[inv, torch.arange(B, device=x0.device)]           # (B,)
        u = torch.stack(g, 0)[pick, torch.arange(B, device=x0.device)] # (B, N)
        x0_sel = _xy_apply_alignment(u, x1)
    return x0_sel, x1



# ---- transport cost --------------------------------------------------------


def xy_transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Mean per-sample sum_i (1 - cos(x0_i - x1_i)) of a coupled pair. (B, N)."""
    return (1.0 - torch.cos(x0 - x1)).sum(-1).mean()
