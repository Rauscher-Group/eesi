"""Equivariant-OT couplings for flow-matching / interpolant training. Vectorized, GPU-ready.

Two methods are implemented for two different systems and their symmetries:

    equivariant_ot_couple   LJ13 point clouds over S(N) x SO(3)   (Klein et al. 2023)
    xy_ot_couple            XY chains over O(2) x Z2^site

Both run under `no_grad`: the coupling is a data-pairing step, and the regression loss
sees the aligned pair as fixed targets, so we never backprop through this.

Both are validated against `tests/ot_reference.py` (scipy CPU version); see 
`tests/test_ot.py`.


Two Levels of OT
--------------
Each coupling has two independent optimizations, kciked off by separate flags
in case you want to do ablation:

    align=True    OT over the symmetry group: per pair of configs in the batch (i,j), 
                  min over g in G, filling M[i,j] cost matrix. For the LJ13 system this
                  is approximated as sequential S_n + SO(3) minimizations.
    batch=True    OT over the batch: choose which x0_i pairs with which x1_j, via
                  linear_sum_assignment on M.

The full equivariant-OT coupling is align=True, batch=True. `align=False, batch=True` is
plain minibatch OT.


Equivariance
--------------------------
Couplings transform x0 and return x1 as-is. This is marginal-preserving
because BOTH p0 and p1 are G-invariant, not just p0. Note that for the marginals, 
i.e. interpolants xt, to be equivariant, any added noise (latent variable) must 
also be G-invariant.

For fixed x1 the alignment map T(x0) = rho(g*(x0,x1)) x0 is invariant under the group
acting on x0 (g*(rho(h)x0, x1) = g*(x0,x1) h^-1), so it projects each orbit onto the
representation best matching x1. The averaging over x1 recovers the marginal: if
p1 is G-invariant, x1 orientations are spread uniformly over its orbit.

Note that `xy_ot_couple` optimizes over O(2) x Z2^site and NOT over S(N): the open XY
chain's nearest-neighbour energy is not permutation-invariant, so permuting the noise would
disrupt the interpolant/marginals. (`XYChainGNN` is not permutation-equivariant either.)

    system   p0                          p1                       group G
    LJ13     COM-free isotropic Gaussian LJ13 Boltzmann           S(13) x SO(3)
    XY       iid uniform on (-pi, pi]    open-chain Boltzmann     O(2) x Z2^site
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .models.xygnn import angle_wrap

try:                                    # optional: batched CUDA Hungarian
    from torch_linear_assignment import batch_linear_assignment
    _HAS_BLA = True
except ImportError:                     # pragma: no cover
    _HAS_BLA = False


# ---- helpers ---------------------------------------------------------------


def center(x: torch.Tensor) -> torch.Tensor:
    """Remove the center of mass. x: (..., N, d)."""
    return x - x.mean(dim=-2, keepdim=True)


def _outer_assignment(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minibatch OT: linear_sum_assignment on a (B, B) cost matrix.

    Just one scipy call, only about 1 ms at B=256, not worth moving to GPU. 
    Returns (rows, cols) as long tensors on M's device.
    """
    r, c = linear_sum_assignment(M.detach().cpu().numpy())
    return (torch.as_tensor(r, device=M.device, dtype=torch.long),
            torch.as_tensor(c, device=M.device, dtype=torch.long))


def _hungarian_nd(D: torch.Tensor) -> torch.Tensor:
    """Batched Hungarian on (K, N, N) costs -> (K, N) column index per row.

    Uses torch-linear-assignment on CUDA when available, else loops scipy.
    """
    if _HAS_BLA and D.is_cuda:
        # batch_linear_assignment returns (K, N): for each row, the assigned column.
        return batch_linear_assignment(D.contiguous())
    
    # CPU fallback
    out = np.empty(D.shape[:2], dtype=np.int64)
    Dn = D.detach().cpu().numpy()
    for k in range(Dn.shape[0]):
        _, c = linear_sum_assignment(Dn[k])
        out[k] = c
    return torch.as_tensor(out, device=D.device)


# ---- LJ13: S(N) x SO(3) ----------------------------------------------------

def _svdvals_3x3(H: torch.Tensor) -> torch.Tensor:
    """Singular values of a batch of 3x3 matrices, descending. (..., 3, 3) -> (..., 3).

    Overhead on `torch.linalg.svdvals(H)` makes it super slow. Instead, get singular 
    values of H as square roots of eigenvalues of H^T H from batched symmetric eigensolve.

    Not seeing any issues by increasing condition number, since LJ13 configurations 
    are not degenerate point clouds, so H is well conditioned.
    """
    e = torch.linalg.eigvalsh(H.transpose(-2, -1) @ H)     # ascending, >= 0 up to roundoff
    return e.clamp_min(0).sqrt().flip(-1)                  # -> descending, as svdvals


@torch.no_grad()
def lj_cost_matrix(x0: torch.Tensor, x1: torch.Tensor, proper: bool = True):
    """Aligned B x B cost matrix over S(N) x SO(3), plus the permutations achieving it.

    x0, x1: (B, N, 3), centred. Returns (M, perm) with M (B, B) and perm (B, B, N):
    perm[i, j] gathers x0_i onto x1_j's particle ordering.

    Only singular VALUES are computed here -- the B^2 rotations are never materialized.
    See `_apply_alignment` for the B survivors.
    """
    B, N, _ = x0.shape

    # 1. pairwise particle cost (B, B, N, N) via the expansion, NOT the (B,B,N,N,3) diff
    sq0 = (x0 ** 2).sum(-1)                                    # (B, N)
    sq1 = (x1 ** 2).sum(-1)                                    # (B, N)
    inner = torch.einsum('iad,jbd->ijab', x0, x1)              # (B, B, N, N)
    D = sq0[:, None, :, None] + sq1[None, :, None, :] - 2 * inner
    D = D.clamp_min(0)          # ||a||^2+||b||^2-2<a,b> loses precision near zero

    # 2. Hungarian per pair: for each x1 site b, which x0 particle feeds it
    cols = _hungarian_nd(D.reshape(B * B, N, N))               # (B*B, N) row -> col
    perm = torch.argsort(cols, dim=-1).reshape(B, B, N)        # invert: col -> row

    # 3. gather x0 into x1's ordering
    idx = perm.reshape(B, B, N, 1).expand(B, B, N, 3)
    x0p = x0[:, None].expand(B, B, N, 3).gather(2, idx)        # (B, B, N, 3)

    # 4. Kabsch cost from singular values only:
    #    c = ||X||^2 + ||Y||^2 - 2(s1 + s2 + d*s3),  d = sign(det H)
    H = torch.einsum('ijna,jnb->ijab', x0p, x1)                # (B, B, 3, 3)
    s = _svdvals_3x3(H)                                        # (B, B, 3), descending
    if proper:
        # sign(det(V U^T)) = sign(det H), since det H = det(U)det(V)*prod(s), prod(s)>=0
        d = torch.sign(torch.linalg.det(H))
        s = torch.cat([s[..., :2], (d * s[..., 2]).unsqueeze(-1)], dim=-1)
    M = sq0.sum(-1)[:, None] + sq1.sum(-1)[None, :] - 2 * s.sum(-1)
    return M, perm


@torch.no_grad()
def _apply_alignment(x0p: torch.Tensor, x1: torch.Tensor, proper: bool = True):
    """Kabsch-rotate already-permuted noise onto the data. x0p, x1: (B, N, 3) -> (B, N, 3).

    Minimizes ||x0p R^T - x1||^2. With H = x0p^T x1 = U S V^T, the maximizer of
    tr(R H) is R = V U^T, corrected to V diag(1,1,d) U^T with d = sign(det H) when a
    proper rotation is required.

    The only place a rotation matrix is ever constructed -- for the B survivors, never
    for the B^2 cost entries.
    """
    B = x0p.shape[0]
    H = torch.einsum('bna,bnc->bac', x0p, x1)                  # (B, 3, 3)
    U, _, Vh = torch.linalg.svd(H)                             # H = U diag(S) Vh
    V, Ut = Vh.transpose(-2, -1), U.transpose(-2, -1)
    if proper:
        d = torch.sign(torch.linalg.det(H))                    # = sign(det(V U^T))
        eye = torch.ones(B, 3, dtype=x0p.dtype, device=x0p.device)
        eye[:, 2] = d
        R = V @ torch.diag_embed(eye) @ Ut
    else:
        R = V @ Ut
    return torch.einsum('bij,bnj->bni', R, x0p)


@torch.no_grad()
def equivariant_ot_couple(x0: torch.Tensor, x1: torch.Tensor, align: bool = True,
                          batch: bool = True, proper: bool = True):
    """Equivariant OT coupling for LJ-13 and related systems. 
       x0, x1: (B, N, 3) -> (x0_aligned, x1_centered)

    TODO: This can be generalized to arbitrary d, not just d = 3. Might be nice
    for 2d systems of interest (e.g. polymer on surfaces, thin films, etc.)

    Transforms the noise (base distribution sample) only; x1 is returned 
    centered but otherwise unchanged.

    `proper=True` restricts rotations to SO(3) (Klein Eq. 16). `proper=False` gives O(3),
    which is generally fine since the LJ energy is reflection-invariant, and the EGNN is
    O(n)-equivariant.
    """
    B, N, _ = x0.shape
    x0, x1 = center(x0), center(x1)

    if align:
        M, perm = lj_cost_matrix(x0, x1, proper=proper)
    else:
        M = ((x0[:, None] - x1[None, :]) ** 2).sum((-1, -2))          # (B, B) unaligned
        perm = None

    r, c = _outer_assignment(M) if batch else (torch.arange(B, device=x0.device),) * 2

    # gather the B selected noise samples, in x1's batch order
    inv = torch.empty(B, dtype=torch.long, device=x0.device)
    inv[c] = r                                     # inv[j] = noise index paired with x1_j
    x0_sel = x0[inv]
    if align:
        p = perm[inv, torch.arange(B, device=x0.device)]               # (B, N)
        x0_sel = x0_sel.gather(1, p.unsqueeze(-1).expand(B, N, 3))
        x0_sel = _apply_alignment(x0_sel, x1, proper=proper)
    return x0_sel, x1


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


def transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Mean per-sample ||x0 - x1||^2 of an already-coupled pair. (B, N, d)."""
    return ((x0 - x1) ** 2).sum(dim=(-1, -2)).mean()


def xy_transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Mean per-sample sum_i (1 - cos(x0_i - x1_i)) of a coupled pair. (B, N)."""
    return (1.0 - torch.cos(x0 - x1)).sum(-1).mean()
