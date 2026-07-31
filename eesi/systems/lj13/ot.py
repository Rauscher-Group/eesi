"""Equivariant-OT coupling for LJ13 point clouds over S(N) x SO(3).

Vectorized, GPU-ready, and validated against `tests/ot_reference.py` (scipy CPU
version); see `tests/lj13/test_ot.py`. Runs under `no_grad`: the coupling is a
data-pairing step and the regression loss sees the aligned pair as fixed
targets, so we never backprop through it.


Two Levels of OT
--------------
Two independent optimizations, kicked off by separate flags in case you want to
do ablation:

    align=True    OT over the symmetry group: per pair of configs in the batch (i,j),
                  min over g in G, filling M[i,j] cost matrix. For the LJ13 system this
                  is approximated as sequential S_n + SO(3) minimizations.
    batch=True    OT over the batch: choose which x0_i pairs with which x1_j, via
                  linear_sum_assignment on M.

The full equivariant-OT coupling is align=True, batch=True. `align=False, batch=True` is
plain minibatch OT.


Equivariance
--------------------------
The coupling transforms x0 and returns x1 as-is. This is marginal-preserving
because BOTH p0 and p1 are G-invariant, not just p0. Note that for the marginals,
i.e. interpolants xt, to be equivariant, any added noise (latent variable) must
also be G-invariant.

For fixed x1 the alignment map T(x0) = rho(g*(x0,x1)) x0 is invariant under the group
acting on x0 (g*(rho(h)x0, x1) = g*(x0,x1) h^-1), so it projects each orbit onto the
representation best matching x1. The averaging over x1 recovers the marginal: if
p1 is G-invariant, x1 orientations are spread uniformly over its orbit.

    p0                          p1                group G
    COM-free isotropic Gaussian LJ13 Boltzmann    S(13) x SO(3)
"""
from __future__ import annotations

from typing import Tuple

import torch

from ...ot import _hungarian_nd, _outer_assignment, _svdvals_3x3, center


# ---- LJ13: S(N) x SO(3) ----------------------------------------------------


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
