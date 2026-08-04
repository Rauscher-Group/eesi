"""Equivariant-OT coupling for TAP chains over O(3).

Vectorized, GPU-ready, and validated against `tests/core/ot_reference.py` (scipy CPU
version); see `tests/tap/test_tap_ot.py`. Runs under `no_grad`: the coupling is a
data-pairing step and the regression loss sees the aligned pair as fixed targets, so
we never backprop through it.

The `align` / `batch` flags mean what they mean in `eesi.systems.lj13.ot`: `align`
optimizes over the symmetry group, `batch` over the batch assignment. The coupling
transforms x0 and returns x1 as-is, which is marginal-preserving because both p0 (the
semiflexible prior) and p1 (the active steady state) are O(3)-invariant. Note what that
argument does NOT require: p0 being Gaussian, or a scale family, or its bonds being
independent of one another. The prior has since acquired a finite bond length, then a
bending potential correlating successive bonds, and this module has needed no change for
either -- all it asks is that rotating a whole configuration leaves its density alone.
The prior earns that by drawing its first bond isotropically and defining every later
one through dot products against its predecessor.


No permutation layer
--------------------
`tap_ot_couple` optimizes over O(3) and NOT over S(N). A tangentially active polymer
is a DIRECTED chain: the active force acts along the local tangent, so relabelling
monomers -- in particular exchanging head and tail -- changes the equations of motion.
Permuting the noise would therefore couple configurations that are not physically
equivalent, and the interpolant between them would be meaningless. (`TAPDynamics` is
not permutation-equivariant either, by construction: see its `index_feature`.)

This makes the coupling STRUCTURALLY SIMPLER than LJ13's, not harder. Klein's
sequential Hungarian-then-Kabsch approximation exists only because S(N) x SO(3) cannot
be optimized jointly; with the permutation gone, orthogonal Procrustes solves the
remaining group EXACTLY. A welcome consequence, shared with the XY chain: the
alignment is exactly equivariant, which the LJ13 one is not (compare
tests/core/test_ot.py::test_rotating_x1_alone_is_not_invariant).


Rotations are about the pinned tail, NOT the centroid
-----------------------------------------------------
The single most important difference from `eesi.systems.lj13.ot`, and the easiest
thing to get wrong when porting it. LJ13 centers both arguments before Kabsch, because
for a free cluster the optimal rigid map includes a translation. Here translation is
already gauge-fixed by x_0 = 0, and O(3) acts about the ORIGIN. Centering would move
the tail off zero, leaving the subspace the prior and the velocity field are defined
on. So the inputs are `anchor`ed (idempotent, removes float drift) and never centered,
and the Procrustes problem is solved on the raw coordinates.


O(3) rather than SO(3)
----------------------
`proper=False` (the default) allows reflections, so all three singular values enter
with a positive sign and no determinant correction is needed. This is the physically
right group: the chain is achiral, the tangential drive reflects along with the
configuration, and `eesi.egnn.EGNN` sees only distances so it cannot detect handedness
in the first place. `proper=True` restricts to SO(3) and exists as an ablation.

    p0                      p1                      group G
    semiflexible chain      TAP active steady state O(3)
"""
from __future__ import annotations

import torch

from ...ot import _outer_assignment, _svdvals_3x3
from .data import anchor


@torch.no_grad()
def tap_cost_matrix(x0: torch.Tensor, x1: torch.Tensor, proper: bool = False) -> torch.Tensor:
    """Aligned B x B cost matrix over O(3). x0, x1: (B, N, 3), anchored. -> (B, B).

    M[i, j] = min over R in O(3) of ||R x0_i - x1_j||^2, pairing noise i with data j.

    With H = x0_i^T x1_j = U S V^T, the maximizer of tr(R H) over orthogonal R is
    R = V U^T, attaining sum_k s_k, so

        M[i, j] = ||x0_i||^2 + ||x1_j||^2 - 2 sum_k s_k(H_ij).

    Only singular VALUES are computed here -- the B^2 rotations are never materialized;
    see `_apply_alignment` for the B survivors. Unlike LJ13 there is no permutation
    stage, so this is a single einsum plus a batched 3x3 eigensolve.
    """
    sq0 = (x0 ** 2).sum(dim=(-1, -2))                          # (B,)
    sq1 = (x1 ** 2).sum(dim=(-1, -2))                          # (B,)
    H = torch.einsum('ina,jnb->ijab', x0, x1)                  # (B, B, 3, 3)
    s = _svdvals_3x3(H)                                        # (B, B, 3), descending
    if proper:
        # sign(det(V U^T)) = sign(det H), since det H = det(U)det(V)*prod(s), prod(s)>=0
        d = torch.sign(torch.linalg.det(H))
        s = torch.cat([s[..., :2], (d * s[..., 2]).unsqueeze(-1)], dim=-1)
    return sq0[:, None] + sq1[None, :] - 2 * s.sum(-1)


@torch.no_grad()
def _apply_alignment(x0: torch.Tensor, x1: torch.Tensor, proper: bool = False) -> torch.Tensor:
    """Rotate noise onto data about the origin. x0, x1: (B, N, 3) -> (B, N, 3).

    Minimizes ||R x0 - x1||^2 over O(3) (or SO(3) with `proper`). The only place a
    rotation matrix is ever constructed -- for the B survivors, never for the B^2
    cost entries.
    """
    B = x0.shape[0]
    H = torch.einsum('bna,bnc->bac', x0, x1)                   # (B, 3, 3)
    U, _, Vh = torch.linalg.svd(H)                             # H = U diag(S) Vh
    V, Ut = Vh.transpose(-2, -1), U.transpose(-2, -1)
    if proper:
        d = torch.sign(torch.linalg.det(H))                    # = sign(det(V U^T))
        eye = torch.ones(B, 3, dtype=x0.dtype, device=x0.device)
        eye[:, 2] = d
        R = V @ torch.diag_embed(eye) @ Ut
    else:
        R = V @ Ut
    return torch.einsum('bij,bnj->bni', R, x0)


@torch.no_grad()
def tap_ot_couple(x0: torch.Tensor, x1: torch.Tensor, align: bool = True,
                  batch: bool = True, proper: bool = False):
    """Equivariant-OT coupling for TAP chains. x0, x1: (B, N, 3) -> (x0_aligned, x1).

    Transforms the noise only; x1 is returned anchored but otherwise unchanged.
    `align=False` drops the O(3) layer, `batch=False` the minibatch-OT layer; the 2x2
    of those flags is the ablation, and the full coupling is both True.
    """
    B = x0.shape[0]
    x0, x1 = anchor(x0), anchor(x1)

    if align:
        M = tap_cost_matrix(x0, x1, proper=proper)
    else:
        M = ((x0[:, None] - x1[None, :]) ** 2).sum((-1, -2))           # (B, B) unaligned

    r, c = _outer_assignment(M) if batch else (torch.arange(B, device=x0.device),) * 2

    # gather the B selected noise samples, in x1's batch order
    inv = torch.empty(B, dtype=torch.long, device=x0.device)
    inv[c] = r                                     # inv[j] = noise index paired with x1_j
    x0_sel = x0[inv]
    if align:
        x0_sel = _apply_alignment(x0_sel, x1, proper=proper)
    return x0_sel, x1
