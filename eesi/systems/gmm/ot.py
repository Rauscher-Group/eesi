"""Minibatch-OT coupling for the GMM system: plain squared-Euclidean cost, no group.

The simplest of the three couplings in this package, and the only one that needs no
invariance argument. `eesi.systems.lj13.ot` optimizes over S(N) x SO(3) and
`eesi.systems.xy.ot` over O(2) x Z2^site because in both cases p0 AND p1 are invariant
under that group, so transforming the noise leaves the prior marginal alone and the
learned field is equivariant. Here the target is a generic mixture in R^d -- rotate it
and you get a different distribution -- so the only layer left is the batch assignment
itself, and the cost is the bare pairwise distance:

    M[i, j] = ||x0_i - x1_j||^2

The base N(0, I) *is* rotationally invariant, so an SO(d) alignment layer would in fact
be marginal-preserving. It is deliberately not here: with a non-invariant p1 it would
rotate each noise sample onto its own data point, collapsing the coupling toward a
radial map and throwing away exactly the orientation information the flow has to learn.
The `align` flag of the other two systems therefore has no analogue in this module;
`batch` (the ablation switch) is the only knob.

Squared distance rather than plain distance: it is the cost whose OT map is the
displacement interpolant flow matching assumes, and it is what the rest of the package
already reports (`eesi.ot.transport_cost`, `lj13.ot.lj_cost_matrix`).

Runs under `no_grad`, like the others -- the coupling is a data-pairing step and the
regression loss sees the paired endpoints as fixed targets.

Measured on the 40-dimensional 16-component mixture of `experiments/GMM/GMM.ipynb`,
transport cost vs the independent pairing: -26.9% at B=256, -28.8% at B=1000. The scipy
Hungarian is O(B^3) and runs on the CPU, so the coupling costs ~2.7 ms at B=256 but
~80 ms at B=1000 -- at that batch size it is comparable to the training step itself.
"""
from __future__ import annotations

import torch

from ...ot import _outer_assignment


@torch.no_grad()
def gmm_cost_matrix(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Squared-Euclidean B x B cost. x0, x1: (B, d).

    M[i, j] = ||x0_i - x1_j||^2, pairing noise i with data j.

    Expanded as |a|^2 + |b|^2 - 2 a.b, one matmul instead of a (B, B, d) difference
    tensor -- the same trick as `lj13.ot.lj_cost_matrix`, with the same `clamp_min(0)`
    afterwards because that form can go slightly negative on near-coincident points.
    Trailing dimensions are flattened, so a (B, N, d) batch works too, but the
    documented contract (and everything the GMM nets accept) is (B, d).
    """
    a, b = x0.flatten(1), x1.flatten(1)
    M = (a * a).sum(-1)[:, None] + (b * b).sum(-1)[None, :] - 2.0 * (a @ b.T)
    return M.clamp_min(0)


@torch.no_grad()
def gmm_ot_couple(x0: torch.Tensor, x1: torch.Tensor, batch: bool = True):
    """Minibatch-OT coupling. x0, x1: (B, d) -> (x0_permuted, x1).

    Reorders the noise to minimize sum_j ||x0_sigma(j) - x1_j||^2 over permutations
    sigma; the data is returned untouched. `batch=False` is the ablation arm: the
    identity pairing, i.e. the independent coupling p0 (x) p1.

    A permutation of the batch preserves the prior marginal exactly -- the coupled
    batch is the same multiset of noise samples, not merely an equal-in-distribution
    one -- which is why this module needs none of the invariance machinery that
    `xy_ot_couple` and `equivariant_ot_couple` do.
    """
    B = x0.shape[0]
    if B != x1.shape[0]:
        # A rectangular cost matrix would give `_outer_assignment` a partial matching,
        # leaving part of `inv` uninitialized. Fail loudly instead.
        raise ValueError(f"x0 and x1 must share a batch size; got {B} and {x1.shape[0]}")
    if not batch:
        return x0, x1

    M = gmm_cost_matrix(x0, x1)
    # r,c are shape (B,); r = row indices for each column, c = column indices for each row
    r, c = _outer_assignment(M)

    # tensor to perform indexing on x0
    inv = torch.empty(B, dtype=torch.long, device=x0.device)
    inv[c] = r                                     # inv[j] = noise index paired with x1_j
    return x0[inv], x1


# ---- transport cost --------------------------------------------------------


def gmm_transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Mean per-sample ||x0 - x1||^2 of an already-coupled pair. (B, d).

    The core `eesi.ot.transport_cost` sums over `dim=(-1, -2)`, so it assumes the
    (B, N, d) point clouds of LJ13; this is the same quantity for flat (B, d) samples.
    """
    return ((x0 - x1) ** 2).flatten(1).sum(-1).mean()
