"""Tests for `eesi.ot` (vectorized) against `eesi.ot_reference` (scipy oracle).

Covers Phase B3 of EQOT_PLAN.md: oracle equivalence, cost floor, monotonicity,
invariance, the SO(3)/O(3) sign correction, the 2x2 ablation, and marginal
preservation.

Runs as either pytest or a plain script:

    pytest tests/test_ot.py
    python tests/test_ot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from scipy.stats import kstest

from eesi import ot, ot_reference as ref
from eesi.lj13 import sample_prior

# NB: deliberately no `torch.set_default_dtype` here. It is global state and leaks
# into every other test module in the same pytest session. `sample_prior` already
# defaults to float64, so everything below is float64 without touching the default.

DT = torch.float64
SD = np.sqrt(12 / 13)          # Var(x_i - xbar) = 1 - 1/13 for a COM-free unit Gaussian


# ---- helpers ---------------------------------------------------------------


def _pair(B: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return sample_prior(B, generator=g), sample_prior(B, generator=g)


def _rot(x: torch.Tensor, seed: int) -> torch.Tensor:
    R = torch.as_tensor(Rotation.random(random_state=seed).as_matrix())
    return torch.einsum('ij,bnj->bni', R, x)


def _perm(x: torch.Tensor, seed: int) -> torch.Tensor:
    p = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(seed))
    return x[:, p]


def _cost(x0, x1, **kw) -> float:
    """Aligned cost of the (i, i) pairing, no batch OT: the pure alignment cost."""
    a, b = ot.equivariant_ot_couple(x0, x1, batch=False, **kw)
    return ot.transport_cost(a, b).item()


# ---- oracle equivalence ----------------------------------------------------


def test_cost_matrix_matches_oracle():
    """Entrywise agreement of the B x B aligned cost matrix -- stronger than
    comparing totals, and it implies the permutations agree too (absent ties)."""
    x0, x1 = _pair(16, seed=0)
    M_g, _ = ot.lj_cost_matrix(ot.center(x0), ot.center(x1))
    M_o = torch.as_tensor(ref.cost_matrix(x0, x1)[0])
    assert (M_g - M_o).abs().max() < 1e-8


def test_all_ablation_cells_match_oracle():
    """All four align/batch combinations agree with the scipy oracle."""
    x0, x1 = _pair(16, seed=1)
    for align in (False, True):
        for batch in (False, True):
            a_g, b_g = ot.equivariant_ot_couple(x0, x1, align=align, batch=batch)
            a_o, b_o = ref.ot_map(x0, x1, align=align, batch=batch)
            assert abs(ot.transport_cost(a_g, b_g).item()
                       - ref.transport_cost(a_o, b_o)) < 1e-6, (align, batch)


def test_o3_branch_matches_oracle():
    """The O(3) deviation also agrees with the oracle."""
    x0, x1 = _pair(12, seed=2)
    a_g, b_g = ot.equivariant_ot_couple(x0, x1, proper=False)
    a_o, b_o = ref.ot_map(x0, x1, proper=False)
    assert abs(ot.transport_cost(a_g, b_g).item() - ref.transport_cost(a_o, b_o)) < 1e-6


def test_cuda_hungarian_matches_scipy():
    """The CUDA path uses the torch-linear-assignment extension; the CPU path loops
    scipy. Training runs the former and every other test here runs the latter, so
    without this the extension's index convention is never checked -- and a mismatch
    would silently degrade the coupling rather than raise.
    """
    if not (torch.cuda.is_available() and ot._HAS_BLA):
        return                                  # skipped: no CUDA extension available
    x0, x1 = _pair(24, seed=12)
    M_c, p_c = ot.lj_cost_matrix(ot.center(x0), ot.center(x1))
    M_g, p_g = ot.lj_cost_matrix(ot.center(x0).cuda(), ot.center(x1).cuda())
    assert (M_c - M_g.cpu()).abs().max() < 1e-9
    assert torch.equal(p_c, p_g.cpu())


# ---- cost properties -------------------------------------------------------


def test_cost_floor():
    """c(x, x) == 0: identity permutation, identity rotation."""
    x, _ = _pair(8, seed=3)
    assert _cost(x, x) < 1e-16


def test_cost_monotonicity():
    """The symmetry-aware cost can only improve on the naive one. A violation
    means a sign or transpose error in Kabsch."""
    x0, x1 = _pair(16, seed=4)
    naive = _cost(x0, x1, align=False)
    assert _cost(x0, x1) <= naive + 1e-9


# ---- invariance ------------------------------------------------------------
#
# NB: Klein's approximation is SEQUENTIAL -- the Hungarian runs on unrotated
# coordinates. Only transformations leaving the Hungarian's cost matrix unchanged
# are exact symmetries. Rotating x1 ALONE is NOT one of them; see
# test_rotating_x1_alone_is_not_invariant and EQOT_PLAN.md Phase B3 test 4.


def test_invariant_to_permuting_x1():
    """Permuting x1 column-permutes the cost matrix: same optimum, exactly."""
    x0, x1 = _pair(8, seed=5)
    base = _cost(x0, x1)
    for s in range(5):
        assert abs(_cost(x0, _perm(x1, s)) - base) < 1e-12


def test_invariant_to_permuting_x0():
    """Permuting x0 row-permutes the cost matrix: same optimum, exactly."""
    x0, x1 = _pair(8, seed=6)
    base = _cost(x0, x1)
    for s in range(5):
        assert abs(_cost(_perm(x0, s), x1) - base) < 1e-12


def test_invariant_to_simultaneous_rotation():
    """A frame change leaves every |x0[a] - x1[b]|^2 unchanged, so the Hungarian
    picks the same permutation and Kabsch is equivariant.

    This is THE test that catches wrong-side and transpose bugs in Kabsch.
    """
    x0, x1 = _pair(8, seed=7)
    base = _cost(x0, x1)
    for s in range(5):
        assert abs(_cost(_rot(x0, s), _rot(x1, s)) - base) < 1e-10


def test_rotating_x1_alone_is_not_invariant():
    """Documents the approximation gap rather than asserting it away.

    The EXACT minimum over S(N) x SO(3) is rotation-invariant; Klein's sequential
    approximation is not, because the permutation is chosen before the rotation.
    The gap takes BOTH signs, which is what distinguishes an approximation from a bug.
    """
    x0, x1 = _pair(8, seed=8)
    base = _cost(x0, x1)
    d = [_cost(x0, _rot(x1, s)) - base for s in range(30)]
    assert max(np.abs(d)) > 1e-3, "no gap at all -> the Hungarian is not running"
    assert min(d) < 0 < max(d), "one-sided gap -> suspect a bug, not an approximation"


# ---- SO(3) vs O(3) ---------------------------------------------------------


def test_kabsch_sign_correction_is_live():
    """Isolate the Kabsch step with the permutation held at identity.

    Through the FULL pipeline this test is meaningless: reflecting x1 changes the
    Hungarian's permutation, so the O(3) branch cannot recover the unreflected cost.
    With x1 = R * mirror(x0): SO(3) must be strictly positive (the mirror is
    unreachable), O(3) must be ~0 (the mirror is free).
    """
    x, _ = _pair(1, seed=9)
    x = ot.center(x)
    mirror = _rot(x * torch.tensor([1.0, 1.0, -1.0], dtype=DT), seed=11)
    c_so3 = ((ot._apply_alignment(x, mirror, proper=True) - mirror) ** 2).sum().item()
    c_o3 = ((ot._apply_alignment(x, mirror, proper=False) - mirror) ** 2).sum().item()
    assert c_so3 > 1e-3, "SO(3) reached a mirror image -> sign correction is dead code"
    assert c_o3 < 1e-16, "O(3) failed to reach a mirror image -> Kabsch is wrong"


# ---- the 2x2 ablation ------------------------------------------------------


def test_cost_reduction_ablation():
    """Both layers help, and they compose. Klein's claim is that `align` dominates
    for LJ13, where S(13) x SO(3) is far too large for batch OT to stumble onto a
    matching permutation."""
    x0, x1 = _pair(32, seed=10)

    def cell(align, batch):
        a, b = ot.equivariant_ot_couple(x0, x1, align=align, batch=batch)
        return ot.transport_cost(a, b).item()

    none = cell(False, False)                        # random pairing: the baseline
    batch_only = cell(False, True)                   # plain minibatch OT
    align_only = cell(True, False)                   # group alignment alone
    both = cell(True, True)                          # full equivariant OT

    assert batch_only < none                         # the batch layer helps
    assert align_only < none                         # the align layer helps
    assert both <= min(align_only, batch_only) + 1e-9        # they compose
    assert align_only < batch_only                   # align dominates, for LJ13


# ---- marginal preservation -------------------------------------------------


def _invariant_p1(n: int, seed: int, invariant: bool = True) -> torch.Tensor:
    """A G-invariant p1: one cluster, randomly rotated AND permuted per sample.
    invariant=False fixes orientation and labelling -> NOT G-invariant."""
    base = ot.center(sample_prior(1, generator=torch.Generator().manual_seed(99)))
    out = base.repeat(n, 1, 1)
    if invariant:
        R = torch.as_tensor(Rotation.random(n, random_state=seed).as_matrix())
        out = torch.einsum('bij,bnj->bni', R, out)
        g = torch.Generator().manual_seed(seed)
        p = torch.stack([torch.randperm(13, generator=g) for _ in range(n)])
        out = out.gather(1, p.unsqueeze(-1).expand(n, 13, 3))
    return ot.center(out)


def _aligned_noise(invariant: bool, B: int = 24, n_batch: int = 48) -> torch.Tensor:
    """Accumulate aligned noise over many small batches -- the coupling is O(B^2)
    in memory and is a small-batch method, so this is also how training calls it."""
    acc = []
    for k in range(n_batch):
        x0 = sample_prior(B, generator=torch.Generator().manual_seed(1000 + k))
        a, _ = ot.equivariant_ot_couple(x0, _invariant_p1(B, k, invariant))
        acc.append(a)
    return torch.cat(acc)


def test_marginal_preserved_when_p1_is_invariant():
    """The gate that makes the noise-transforming convention legitimate.

    The statistic must be NON-G-invariant or it is preserved trivially: a raw
    coordinate is N(0, 12/13) under the COM-free prior.
    """
    s = _aligned_noise(invariant=True)[:, 0, 0].numpy()
    assert kstest(s / SD, "norm").pvalue > 0.01, "aligned noise is not prior-distributed"
    assert abs(s.mean()) < 0.05 and abs(s.std() - SD) < 0.05


def test_marginal_broken_when_p1_is_not_invariant():
    """Negative control: proves the test above has power.

    Exchangeability of the prior is necessary but NOT sufficient -- p1 must be
    G-invariant too, or aligning the noise leaks p1's structure into the marginal.
    """
    s = _aligned_noise(invariant=False)[:, 0, 0].numpy()
    assert kstest(s / SD, "norm").pvalue < 1e-6, "control did not break -> test is blind"


# ============================================================================
# Part C -- XY chain, Z2 x U(1)
# ============================================================================


def _xy_pair(B: int = 16, L: int = 12, seed: int = 0):
    """Prior: i.i.d. uniform on (-pi, pi]. Data: a chain-correlated random walk with a
    uniform global phase -- NOT S(L)-invariant (it has chain structure), but exactly
    Z2 x U(1)-invariant: a uniform phase gives U(1), and iid symmetric increments give
    reversal (reversing negates and reverses the increments, same law)."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand(B, L, generator=g, dtype=DT) * 2 * np.pi - np.pi
    step = 0.35 * torch.randn(B, L, generator=g, dtype=DT)
    x1 = ot.angle_wrap(torch.cumsum(step, 1)
                       + 2 * np.pi * torch.rand(B, 1, generator=g, dtype=DT))
    return x0, x1


def _xy_cost(x0, x1, **kw) -> float:
    a, b = ot.xy_ot_couple(x0, x1, batch=False, **kw)
    return ot.xy_transport_cost(a, b).item()


def _nn_corr(x: torch.Tensor) -> float:
    """<cos(theta_{i+1} - theta_i)>: 0 for the i.i.d. uniform prior, strongly positive
    for chain-correlated data. The statistic that detects structure leaking into the
    aligned noise."""
    return torch.cos(x[:, 1:] - x[:, :-1]).mean().item()


def test_xy_closed_form_matches_grid_oracle():
    """L - sqrt(S^2+C^2) against a dense phi grid. The closed form is the whole reason
    the XY path needs no SVD, so it is validated against something that is NOT itself."""
    x0, x1 = _xy_pair(12, seed=0)
    M_g, _ = ot.xy_cost_matrix(x0, x1)
    M_o = np.array([[ref.xy_match_pair(x0[i].numpy(), x1[j].numpy())[1]
                     for j in range(12)] for i in range(12)])
    assert np.abs(M_g.numpy() - M_o).max() < 1e-5      # grid resolution is ~6e-4


def test_xy_all_ablation_cells_match_oracle():
    x0, x1 = _xy_pair(12, seed=1)
    for reflect in (False, True):
        for align in (False, True):
            for batch in (False, True):
                a_g, b_g = ot.xy_ot_couple(x0, x1, align=align, batch=batch,
                                           reflect=reflect)
                a_o, b_o = ref.xy_ot_map(x0, x1, align=align, batch=batch,
                                         reflect=reflect)
                assert abs(ot.xy_transport_cost(a_g, b_g).item()
                           - ref.xy_transport_cost(a_o, b_o)) < 2e-3, \
                    (reflect, align, batch)


def test_xy_cost_floor():
    x, _ = _xy_pair(8, seed=2)
    assert _xy_cost(x, x) < 1e-16


def test_xy_cost_monotonicity():
    x0, x1 = _xy_pair(16, seed=3)
    assert _xy_cost(x0, x1) <= _xy_cost(x0, x1, align=False) + 1e-9


def test_xy_invariance_is_EXACT_unlike_lj13():
    """The XY group is enumerated exactly, so -- unlike Klein's sequential
    approximation for LJ13 -- rotating or reversing x1 alone IS an exact symmetry.

    This is the asymmetry worth remembering: the small-group system has the rigorous
    guarantee, the big-group system does not. Contrast
    test_rotating_x1_alone_is_not_invariant.
    """
    x0, x1 = _xy_pair(8, seed=4)
    base = _xy_cost(x0, x1)
    for psi in (0.3, 1.7, -2.9):                       # rotate x1 alone: absorbed by phi*
        assert abs(_xy_cost(x0, ot.angle_wrap(x1 + psi)) - base) < 1e-10
    assert abs(_xy_cost(x0, x1.flip(-1)) - base) < 1e-10          # reverse x1 alone
    for psi in (0.5, -1.2):                            # rotate x0 alone
        assert abs(_xy_cost(ot.angle_wrap(x0 + psi), x1) - base) < 1e-10
    assert abs(_xy_cost(x0.flip(-1), x1) - base) < 1e-10          # reverse x0 alone


def test_xy_reflection_branch_is_live():
    """With reflect=False, reversing x1 must CHANGE the cost -- otherwise the Z2 branch
    is dead code. (The LJ13 analogue is test_kabsch_sign_correction_is_live.)"""
    x0, x1 = _xy_pair(8, seed=5)
    base = _xy_cost(x0, x1, reflect=False)
    assert abs(_xy_cost(x0, x1.flip(-1), reflect=False) - base) > 1e-3
    # and enabling Z2 can only help
    assert _xy_cost(x0, x1, reflect=True) <= base + 1e-9


def test_xy_cost_reduction_ablation():
    """Both layers help. Unlike LJ13, `align` does NOT dominate here: Z2 x U(1) is a
    tiny group, so most of the win is ordinary minibatch OT wearing a symmetry-aware
    cost. A modest alignment gain is the expected result, not a failure."""
    x0, x1 = _xy_pair(32, seed=6)

    def cell(align, batch):
        a, b = ot.xy_ot_couple(x0, x1, align=align, batch=batch)
        return ot.xy_transport_cost(a, b).item()

    none, batch_only = cell(False, False), cell(False, True)
    align_only, both = cell(True, False), cell(True, True)
    assert batch_only < none and align_only < none
    assert both <= min(align_only, batch_only) + 1e-9


def _xy_aligned_noise(permute: bool, B: int = 24, n_batch: int = 64) -> torch.Tensor:
    """Accumulate aligned noise over small batches. `permute=True` adds an S(L)
    Hungarian step -- the inadmissible group, as a negative control."""
    acc = []
    for k in range(n_batch):
        x0, x1 = _xy_pair(B, seed=5000 + k)
        if permute:
            for i in range(B):                        # S(L) on the chordal cost
                C = 1.0 - torch.cos(x0[i][:, None] - x1[i][None, :])
                r, c = linear_sum_assignment(C.numpy())
                sigma = np.empty(len(r), dtype=int)
                sigma[c] = r
                x0[i] = x0[i][sigma]
        acc.append(ot.xy_ot_couple(x0, x1)[0])
    return torch.cat(acc)


def test_xy_marginal_preserved_over_z2_u1():
    """⚠️ The gate that makes Part C legitimate.

    Both p0 and p1 are Z2 x U(1)-invariant, so aligning the noise preserves its
    marginal. <cos(theta_{i+1}-theta_i)> is 0 for the i.i.d. uniform prior; if chain
    structure leaks into the aligned noise, this catches it.
    """
    assert abs(_nn_corr(_xy_aligned_noise(permute=False))) < 0.02


def test_xy_marginal_BROKEN_by_permutation():
    """Negative control, and the experiment behind the invariance table: adding an S(L)
    step -- a symmetry of the prior but NOT of the open chain's energy -- leaks chain
    structure into the prior marginal. This is why S(L) is not in the group.
    """
    clean = abs(_nn_corr(_xy_aligned_noise(permute=False)))
    broken = _nn_corr(_xy_aligned_noise(permute=True))
    assert broken > 0.1, "S(L) control did not break the marginal -> test is blind"
    assert broken > 5 * clean


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
