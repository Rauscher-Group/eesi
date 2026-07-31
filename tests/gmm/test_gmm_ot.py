"""Tests for `eesi.systems.gmm.ot`: the plain minibatch-OT coupling.

No symmetry group here, so there is no alignment layer to validate against an oracle --
the whole content of the module is the squared-Euclidean cost and the batch assignment.
These check both against independent implementations (`torch.cdist`, a scipy
`linear_sum_assignment` on a numpy cost matrix built by a double loop), plus the
properties the coupling has to keep: it is a permutation of the noise, it lowers the
transport cost, and it carries no gradient.

Runs as either pytest or a plain script:

    pytest tests/gmm/test_gmm_ot.py
    python tests/gmm/test_gmm_ot.py

(Named `test_gmm_ot` rather than `test_ot`: the test dirs carry no `__init__.py`, so
pytest imports by basename and would collide with `tests/core/test_ot.py`.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from eesi.systems.gmm.ot import gmm_cost_matrix, gmm_ot_couple, gmm_transport_cost

# No `torch.set_default_dtype` -- it is global state and leaks into every other test
# module in the same pytest session. Everything below is explicitly float64.
DT = torch.float64


# ---- helpers ---------------------------------------------------------------


def _pair(B: int = 16, d: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, d, generator=g, dtype=DT),
            2.0 + torch.randn(B, d, generator=g, dtype=DT))


def _oracle_cost(x0, x1) -> np.ndarray:
    """The B x B cost the obvious, slow way: one Python loop per entry."""
    a, b = x0.numpy(), x1.numpy()
    B = a.shape[0]
    M = np.empty((B, B))
    for i in range(B):
        for j in range(B):
            M[i, j] = ((a[i] - b[j]) ** 2).sum()
    return M


# ---- the cost matrix -------------------------------------------------------


def test_cost_matrix_matches_cdist():
    """Entrywise agreement with `torch.cdist` squared -- the expanded
    |a|^2 + |b|^2 - 2 a.b form is an optimization, not a different cost."""
    x0, x1 = _pair(16, 5, seed=0)
    M = gmm_cost_matrix(x0, x1)
    assert M.shape == (16, 16)
    # ~1e-12 at these magnitudes: cdist is the stable form, the expansion is not.
    assert (M - torch.cdist(x0, x1) ** 2).abs().max() < 1e-10


def test_cost_matrix_matches_loop_oracle():
    """And with a plain double loop, which shares no code path with either."""
    x0, x1 = _pair(12, 3, seed=1)
    M = gmm_cost_matrix(x0, x1)
    assert (M - torch.as_tensor(_oracle_cost(x0, x1))).abs().max() < 1e-10


def test_cost_floor():
    """A sample against itself costs zero, and the clamp keeps it from going
    negative in the subtraction."""
    x0, _ = _pair(16, 5, seed=2)
    M = gmm_cost_matrix(x0, x0)
    # Not 0 exactly: |a|^2 + |b|^2 - 2 a.b cancels to ~eps * |x|^2, i.e. ~2e-15 at
    # these magnitudes. That cancellation is the price of the one-matmul form and the
    # reason for the clamp on the next line.
    assert M.diagonal().abs().max() < 1e-13
    assert (M >= 0).all()


def test_cost_is_shape_agnostic():
    """Trailing dims are flattened, so a (B, N, d) batch gives the (B, N*d) cost.
    The documented contract is (B, d); this keeps the extra generality honest."""
    g = torch.Generator().manual_seed(3)
    x0 = torch.randn(8, 4, 3, generator=g, dtype=DT)
    x1 = torch.randn(8, 4, 3, generator=g, dtype=DT)
    M = gmm_cost_matrix(x0, x1)
    M_flat = gmm_cost_matrix(x0.reshape(8, 12), x1.reshape(8, 12))
    assert (M - M_flat).abs().max() < 1e-12


# ---- the assignment --------------------------------------------------------


def test_permutation_matches_scipy_oracle():
    """The coupled noise is exactly what scipy's Hungarian picks on an
    independently built cost matrix."""
    x0, x1 = _pair(24, 5, seed=4)
    a, _ = gmm_ot_couple(x0, x1)
    r, c = linear_sum_assignment(_oracle_cost(x0, x1))
    inv = np.empty(24, dtype=np.int64)
    inv[c] = r                                  # inv[j] = noise index paired with x1_j
    assert torch.equal(a, x0[torch.as_tensor(inv)])


def test_coupling_is_a_permutation_of_the_noise():
    """The coupled batch is the SAME multiset of noise samples, so the prior marginal
    is preserved exactly -- not merely in distribution, as for the group couplings.
    The data is returned untouched."""
    x0, x1 = _pair(32, 4, seed=5)
    a, b = gmm_ot_couple(x0, x1)
    assert a.shape == x0.shape
    # sort rows lexicographically via their first coordinate (distinct a.s.)
    assert torch.equal(a[a[:, 0].argsort()], x0[x0[:, 0].argsort()])
    assert torch.equal(b, x1)


def test_ot_lowers_transport_cost():
    """The point of the whole module. `batch=False` is the independent coupling."""
    x0, x1 = _pair(64, 5, seed=6)
    a, b = gmm_ot_couple(x0, x1, batch=True)
    a0, b0 = gmm_ot_couple(x0, x1, batch=False)
    assert torch.equal(a0, x0), "batch=False must be the identity pairing"
    assert gmm_transport_cost(a, b) < gmm_transport_cost(a0, b0)


def test_assignment_is_optimal_over_random_permutations():
    """Stronger than the previous test: no random permutation beats it, and the
    optimum is strictly better than every one of them (no ties at B=8)."""
    x0, x1 = _pair(8, 3, seed=7)
    best = gmm_transport_cost(*gmm_ot_couple(x0, x1)).item()
    g = torch.Generator().manual_seed(8)
    for _ in range(200):
        p = torch.randperm(8, generator=g)
        assert gmm_transport_cost(x0[p], x1).item() > best - 1e-12


def test_transport_cost_matches_the_coupled_cost_matrix_entries():
    """`gmm_transport_cost` is the mean of the selected M entries, so the two agree
    by construction -- a guard against one of them drifting."""
    x0, x1 = _pair(16, 5, seed=9)
    M = gmm_cost_matrix(x0, x1)
    r, c = linear_sum_assignment(M.numpy())
    a, b = gmm_ot_couple(x0, x1)
    assert abs(gmm_transport_cost(a, b).item() - M[r, c].mean().item()) < 1e-10


# ---- gradient hygiene ------------------------------------------------------


def test_no_gradient_flows_through_the_coupling():
    """The coupling is a data-pairing step; the loss sees its output as fixed."""
    x0, x1 = _pair(16, 5, seed=10)
    x0.requires_grad_(True)
    x1.requires_grad_(True)
    a, b = gmm_ot_couple(x0, x1)
    assert not a.requires_grad
    assert not gmm_cost_matrix(x0, x1).requires_grad
    # x1 is passed straight through, so it keeps whatever it arrived with.
    assert b is x1


def test_device_and_dtype_are_preserved():
    """float32 is the GMM default; the scipy round-trip must not silently upcast."""
    x0, x1 = _pair(16, 5, seed=11)
    a, b = gmm_ot_couple(x0.float(), x1.float())
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.device == x0.device


def test_rejects_mismatched_batch_sizes():
    """A rectangular cost matrix would leave `_outer_assignment` with a partial
    matching and part of the gather index uninitialized -- silent garbage."""
    x0, _ = _pair(16, 5, seed=12)
    x1, _ = _pair(8, 5, seed=13)
    with pytest.raises(ValueError, match="batch size"):
        gmm_ot_couple(x0, x1)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
