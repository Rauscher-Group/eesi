"""Equivariant-OT couplings for flow-matching / interpolant training. Vectorized, GPU-ready.

Two systems, one shape:

    equivariant_ot_couple   LJ13 point clouds over S(N) x SO(3)   (Klein et al. 2023)
    xy_ot_couple            XY chains over Z2 x U(1)              (see EQOT_PLAN.md Part C)

Both run under `no_grad`: the coupling is a data-pairing step, and the regression loss
sees the aligned pair as fixed targets. Never backprop through this. A hard assignment
is correct here; a Sinkhorn relaxation would be strictly worse.

Both are validated against `eesi.ot_reference` (scipy oracle). See `tests/test_ot.py`.


THE TWO LAYERS
--------------
Each coupling has two independent optimization layers, exposed as separate flags so the
2x2 ablation in EQOT_PLAN.md is a parameter sweep rather than a code change:

    align=True    OT over the GROUP: per pair (i,j), min over g in G, filling M[i,j].
    batch=True    OT over the BATCH: choose which x0_i pairs with which x1_j, via
                  linear_sum_assignment on M.

The full equivariant-OT coupling is align=True, batch=True. `align=False, batch=True` is
plain minibatch OT. Comparing only those two credits the group alignment with everything
the batch layer also does -- hence the 2x2.


WHY WE TRANSFORM THE NOISE
--------------------------
Both couplings transform x0 and return x1 untouched. This is only marginal-preserving
because BOTH p0 and p1 are G-invariant -- not just the prior.

For fixed x1 the alignment map T(x0) = rho(g*(x0,x1)) x0 is *invariant* under the group
acting on x0 (g*(rho(h)x0, x1) = g*(x0,x1) h^-1), so it collapses each orbit onto the
single representative best matching x1. What undoes the collapse is averaging over x1: if
p1 is G-invariant, x1's own orientation is spread uniformly over its orbit and the
marginal is restored. If p1 is NOT G-invariant, it never is.

Exchangeability of the prior is necessary but NOT sufficient. Counterexample: p0 = N(0,I)
on R^2 (exchangeable), p1 = delta(1,0) (not). Aligning the noise over S(2) sorts every x0
descending; the marginal becomes the sorted-Gaussian law, not N(0,I).

This is why `xy_ot_couple` optimizes over Z2 x U(1) and NOT over S(L): the open XY chain's
nearest-neighbour energy is not permutation-invariant, so permuting the noise would leak
chain structure into the prior marginal. (`XYChainGNN` is not permutation-equivariant
either -- it is built on a static chain graph. Both facts point the same way.)

    system   p0                          p1                       admissible G
    LJ13     COM-free isotropic Gaussian LJ13 Boltzmann           S(13) x SO(3)
    XY       iid uniform on (-pi, pi]    open-chain NN Boltzmann  Z2 x U(1)   (S(L) is NOT)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .xygnn import angle_wrap

try:                                    # optional: batched CUDA Hungarian
    from torch_linear_assignment import batch_linear_assignment
    _HAS_BLA = True
except ImportError:                     # pragma: no cover
    _HAS_BLA = False


# ---- helpers ---------------------------------------------------------------


def center(x: torch.Tensor) -> torch.Tensor:
    """Remove the centre of mass. x: (..., N, d)."""
    return x - x.mean(dim=-2, keepdim=True)


def _outer_assignment(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minibatch OT: linear_sum_assignment on a (B, B) cost matrix.

    One scipy call, ~1 ms at B=256 -- not worth moving to GPU. Returns (rows, cols)
    as long tensors on M's device.
    """
    r, c = linear_sum_assignment(M.detach().cpu().numpy())
    return (torch.as_tensor(r, device=M.device, dtype=torch.long),
            torch.as_tensor(c, device=M.device, dtype=torch.long))


def _hungarian_nd(D: torch.Tensor) -> torch.Tensor:
    """Batched Hungarian on (K, N, N) costs -> (K, N) column index per row.

    Uses torch-linear-assignment on CUDA when available (it is a CUDA extension and
    offers no CPU path), else loops scipy.
    """
    if _HAS_BLA and D.is_cuda:
        # batch_linear_assignment returns (K, N): for each row, the assigned column.
        return batch_linear_assignment(D.contiguous())
    out = np.empty(D.shape[:2], dtype=np.int64)
    Dn = D.detach().cpu().numpy()
    for k in range(Dn.shape[0]):
        _, c = linear_sum_assignment(Dn[k])
        out[k] = c
    return torch.as_tensor(out, device=D.device)


# ---- LJ13: S(N) x SO(3) ----------------------------------------------------


def _svdvals_3x3(H: torch.Tensor) -> torch.Tensor:
    """Singular values of a batch of 3x3 matrices, descending. (..., 3, 3) -> (..., 3).

    `torch.linalg.svdvals(H)` computes exactly this and is the obvious call. It is also
    ~20x slower: cuSOLVER's batched SVD has per-matrix overhead that dwarfs a 3x3, to the
    point where it runs slower on the GPU than on the CPU (51 ms vs 33 ms for a (256,256)
    stack). Since the singular values of H are the square roots of the eigenvalues of
    H^T H, a batched symmetric eigensolve gets the same numbers for ~1/20th the time.
    Measured at B=256: 48.8 ms -> 2.24 ms. (EQOT_PLAN.md Phase B5.)

    Squaring the condition number is the standard objection to this route, and it would
    matter most for the SMALLEST singular value -- which is the one the SO(3) sign
    correction rides on. It does not bite here: against an fp64 svdvals reference on real
    LJ13 data this errs 5.44e-05 in fp32, versus svdvals' own 5.35e-05, and the outer
    assignment is unchanged. LJ13 configurations are not degenerate point clouds, so H is
    well conditioned. If that ever stops being true, swapping back is a one-line change.
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
    """Equivariant-OT coupling for LJ13. x0, x1: (B, N, 3) -> (x0_aligned, x1_centred).

    Transforms the NOISE only; x1 is returned centred but otherwise untouched. See the
    module docstring for why that is valid, and for the meaning of `align` / `batch`.

    `proper=True` restricts rotations to SO(3) (Klein Eq. 16). `proper=False` gives O(3),
    which is defensible (the LJ energy is reflection-invariant, and the EGNN is
    O(n)-equivariant) but is a deviation.
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


# ---- XY chain: Z2 x U(1) ---------------------------------------------------
#
# Structurally the same as the LJ13 path, but SIMPLER, not harder:
#
#   * No Hungarian. Z2 x U(1) is small enough to enumerate EXACTLY -- two reflections,
#     each with a closed-form optimal rotation. Klein's sequential Hungarian->Kabsch
#     approximation exists because S(N) x SO(3) cannot be enumerated; this can. A
#     welcome consequence: the alignment map is exactly equivariant here, which the
#     LJ13 one is not (see tests/test_ot.py::test_rotating_x1_alone_is_not_invariant).
#   * No SVD. The U(1) alignment is the circular mean, the S^1 analogue of Kabsch.
#
# The group:
#     global rotation phi:  theta_i -> theta_i + phi     energy depends only on differences
#     reflection r:         theta_i -> theta_{L-1-i}     d(theta') = -d(theta), cos is even
# Both are exact symmetries of the open-chain NN energy J*sum_i cos(theta_{i+1}-theta_i),
# and both are respected by XYChainGNN. If the chain ever becomes PERIODIC the group
# grows to D_L x U(1) (L cyclic translations x reflection, 2L elements) -- still exact
# enumeration, still no Hungarian: `_xy_group_elements` becomes the only thing to change.
#
# The cost is CHORDAL (1 - cos) rather than wrapped-geodesic (wrap(d)^2). This is a
# deliberate choice: it is what makes the rotation closed-form. Per pair,
# 1-cos(d) = 2 sin^2(d/2) is a monotone function of |wrap(d)| on [0, pi], so the two
# agree on per-pair ordering; they differ on sums. Note the divergence from the
# geodesic cost `xyEESI._interpolant_sample` uses for the interpolant itself.


def _xy_group_elements(x: torch.Tensor, reflect: bool = True):
    """The Z2 orbit of the noise: (x, x_reversed). Yields (B, L) tensors.

    The single place the discrete group is defined. A periodic chain would enumerate
    the 2L dihedral elements here instead, and nothing else would change.
    """
    return (x, x.flip(-1)) if reflect else (x,)


@torch.no_grad()
def xy_cost_matrix(x0: torch.Tensor, x1: torch.Tensor, reflect: bool = True):
    """Aligned B x B cost matrix over Z2 x U(1). x0, x1: (B, L).

    Returns (M, which) with M (B, B) the minimum chordal cost and `which` (B, B) long
    the index of the Z2 element achieving it. M[i, j] pairs noise i with data j.

    Closed form. For d = x1_j - rho(r)x0_i, S = sum_l sin(d_l), C = sum_l cos(d_l):

        min_phi sum_l (1 - cos(phi - d_l)) = L - sqrt(S^2 + C^2),   phi* = atan2(S, C)

    The COST needs no atan2 at all -- only the B pairs the outer OT selects need phi*
    materialized. This is exactly the LJ13 `svdvals` trick: cheap invariants for the
    B^2 entries, the full factorization only for the B survivors.

    Expanding sin/cos of the difference turns the reduction over L into matmuls, so the
    (B, B, L) tensor is never built -- four GEMMs per group element. That matters here
    far more than for LJ13: L is 200 in classicalXY's default, not 13.
    """
    L = x0.shape[-1]
    s1, c1 = torch.sin(x1), torch.cos(x1)                      # (B, L)
    Ms = []
    for u in _xy_group_elements(x0, reflect):
        su, cu = torch.sin(u), torch.cos(u)                    # (B, L)
        S = cu @ s1.T - su @ c1.T                              # (B, B) sum_l sin(x1-u)
        C = cu @ c1.T + su @ s1.T                              # (B, B) sum_l cos(x1-u)
        Ms.append(L - torch.hypot(S, C))
    M = torch.stack(Ms, 0)                                     # (n_g, B, B)
    return M.min(0).values, M.argmin(0)


@torch.no_grad()
def _xy_apply_alignment(u: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Rotate noise onto data by the closed-form circular mean. u, x1: (B, L).

    The only place atan2 is called -- for the B survivors, never for the B^2 entries.
    """
    d = x1 - u
    phi = torch.atan2(torch.sin(d).sum(-1), torch.cos(d).sum(-1))     # (B,)
    return angle_wrap(u + phi.unsqueeze(-1))


@torch.no_grad()
def xy_ot_couple(x0: torch.Tensor, x1: torch.Tensor, align: bool = True,
                 batch: bool = True, reflect: bool = True):
    """Equivariant-OT coupling for XY chains. x0, x1: (B, L) -> (x0_aligned, x1).

    Transforms the NOISE only; x1 is returned untouched. Valid because both marginals
    are Z2 x U(1)-invariant: the prior is i.i.d. uniform on (-pi, pi], and the
    open-chain Boltzmann energy depends only on wrapped differences and is even under
    reversal. See the module docstring -- and note this is exactly why S(L) is NOT in
    the group, despite being a symmetry of the prior.

    `reflect=False` restricts to U(1) alone: the ablation isolating the Z2 layer.
    """
    B = x0.shape[0]
    if align:
        M, which = xy_cost_matrix(x0, x1, reflect=reflect)
    else:
        M = (1.0 - torch.cos(x0[:, None] - x1[None, :])).sum(-1)      # (B, B) chordal
        which = None

    r, c = _outer_assignment(M) if batch else (torch.arange(B, device=x0.device),) * 2

    inv = torch.empty(B, dtype=torch.long, device=x0.device)
    inv[c] = r                                     # inv[j] = noise index paired with x1_j
    x0_sel = x0[inv]
    if align:
        # apply each survivor's own Z2 element, then its closed-form rotation
        g = _xy_group_elements(x0_sel, reflect)
        pick = which[inv, torch.arange(B, device=x0.device)]           # (B,)
        u = torch.stack(g, 0)[pick, torch.arange(B, device=x0.device)]
        x0_sel = _xy_apply_alignment(u, x1)
    return x0_sel, x1


# ---- transport cost --------------------------------------------------------


def transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Mean per-sample ||x0 - x1||^2 of an already-coupled pair. (B, N, d)."""
    return ((x0 - x1) ** 2).sum(dim=(-1, -2)).mean()


def xy_transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Mean per-sample sum_i (1 - cos(x0_i - x1_i)) of a coupled pair. (B, L)."""
    return (1.0 - torch.cos(x0 - x1)).sum(-1).mean()
