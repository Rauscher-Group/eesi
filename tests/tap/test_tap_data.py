"""Tests for `eesi.systems.tap.data`: the ideal-chain prior and subspace geometry.

Runs as either pytest or a plain script:

    pytest tests/tap/test_tap_data.py
    python tests/tap/test_tap_data.py

The prior is where a silent convention error would hide -- a stray factor of 3 from
the spatial dimension, or (N-1) vs N bonds, produces a perfectly plausible-looking
chain with the wrong length scale, and nothing downstream would complain. So these
tests are quantitative and check the definition directly: ReSqr IS the mean squared
end-to-end distance, and `log_prior` IS the normalized density of what `sample_prior`
draws.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch

from eesi.systems.tap.data import (
    DOF,
    anchor,
    bond_sigma,
    bond_vectors,
    dof,
    end_to_end_sq,
    gyration_sq,
    load_ref_data,
    log_prior,
    sample_prior,
    subspace_dirs,
)

N, RE_SQR = 20, 4.0
BIG = 200_000       # enough that the MC checks below are tight but still ~1 s


def _gen(seed: int = 0):
    return torch.Generator().manual_seed(seed)


# ---- the defining convention ----------------------------------------------


def test_end_to_end_matches_re_sqr():
    """E|x_{N-1} - x_0|^2 == ReSqr. This is the definition of the parameter.

    Catches a missing factor of n_dims in `bond_sigma` (would give 3x) and an
    off-by-one in the bond count (would give ~N/(N-1)x).
    """
    x = sample_prior(BIG, RE_SQR, n_particles=N, generator=_gen(0))
    got = end_to_end_sq(x).mean().item()
    # relative std of the mean is sqrt(2/(N-1)) / sqrt(BIG) ~ 7e-4; 2% is ample room
    assert abs(got - RE_SQR) / RE_SQR < 0.02, f"E[Re^2]={got}, want {RE_SQR}"


def test_bond_mean_square_is_re_sqr_over_n_bonds():
    """E|b|^2 == ReSqr / (N-1), the per-bond share."""
    x = sample_prior(BIG, RE_SQR, n_particles=N, generator=_gen(1))
    got = (bond_vectors(x) ** 2).sum(-1).mean().item()
    assert abs(got - RE_SQR / (N - 1)) / (RE_SQR / (N - 1)) < 0.01


def test_re_sqr_scaling_is_linear():
    """Doubling ReSqr doubles the mean squared end-to-end distance."""
    a = end_to_end_sq(sample_prior(50_000, 1.0, n_particles=N, generator=_gen(2))).mean()
    b = end_to_end_sq(sample_prior(50_000, 2.0, n_particles=N, generator=_gen(2))).mean()
    assert abs((b / a).item() - 2.0) < 0.05


def test_bonds_are_iid_and_isotropic():
    """Bond vectors are uncorrelated across bonds and isotropic within a bond.

    An ideal chain has no bending rigidity; a nonzero bond-bond correlation would mean
    the sampler had accidentally correlated successive increments.
    """
    b = bond_vectors(sample_prior(100_000, RE_SQR, n_particles=N, generator=_gen(3)))
    var = b.var(dim=(0, 1))                       # per-component
    assert torch.allclose(var, var.mean().expand(3), rtol=0.05), var.tolist()
    corr = (b[:, :-1] * b[:, 1:]).sum(-1).mean().item()
    assert abs(corr) < 0.01 * (RE_SQR / (N - 1))


def test_positions_are_correlated_not_iid():
    """Positions are a cumulative sum, so they are NOT iid -- var grows along the chain.

    Guards against a sampler that forgot the cumsum and returned raw bonds as
    positions, which would still pass a naive per-bond variance check.
    """
    x = sample_prior(50_000, RE_SQR, n_particles=N, generator=_gen(4))
    v = (x ** 2).sum(-1).mean(0)                  # E|x_k|^2 per particle
    assert v[1] < v[N // 2] < v[-1]
    # E|x_k|^2 = k * ReSqr/(N-1) for an ideal chain anchored at x_0
    expect = torch.arange(N, dtype=v.dtype) * (RE_SQR / (N - 1))
    assert torch.allclose(v, expect, rtol=0.05, atol=1e-3), v.tolist()


# ---- the subspace ----------------------------------------------------------


def test_tail_is_exactly_zero():
    """Row 0 is a literal zero, not a zero up to roundoff."""
    x = sample_prior(64, RE_SQR, n_particles=N, generator=_gen(5))
    assert (x[:, 0] == 0).all()


def test_anchor_is_idempotent_and_fixes_the_tail():
    x = torch.randn(16, N, 3, dtype=torch.float64)
    a = anchor(x)
    assert a[:, 0].abs().max() == 0
    assert torch.equal(anchor(a), a)


def test_dof_and_basis():
    """DOF = 3(N-1), and the basis is orthonormal and spans {v : v_0 = 0}."""
    assert DOF == 3 * (N - 1) == 57
    assert dof(6, 3) == 15
    dirs = subspace_dirs(6, 3)
    assert dirs.shape == (15, 6, 3)
    G = dirs.reshape(15, -1)
    assert torch.allclose(G @ G.T, torch.eye(15))
    assert dirs[:, 0].abs().max() == 0          # never touches the pinned tail


# ---- the density -----------------------------------------------------------


def test_log_prior_matches_scipy_on_the_bond_representation():
    """log_prior equals a plain Gaussian density on the bonds, i.e. |det J| = 1.

    The independent check of the module's central claim: the cumulative-sum map from
    bonds to positions is unit lower triangular, so no Jacobian term is needed.
    """
    from scipy.stats import multivariate_normal

    n = 6
    x = sample_prior(32, RE_SQR, n_particles=n, generator=_gen(6))
    sigma = bond_sigma(RE_SQR, n)
    b = bond_vectors(x).reshape(32, -1).numpy()
    ref = multivariate_normal.logpdf(b, mean=np.zeros(b.shape[1]), cov=sigma ** 2)
    got = log_prior(x, RE_SQR).numpy()
    assert np.allclose(got, ref, atol=1e-9), (got[:3], ref[:3])


def test_log_prior_is_normalized_via_the_entropy():
    """-E[log p] equals the analytic Gaussian entropy DOF/2 * (1 + log(2 pi sigma^2)).

    An independent check of the NORMALIZER specifically: the quadratic term could be
    right while the constant is wrong, and every relative comparison would still work
    while every absolute entropy would be off by a fixed offset.
    """
    x = sample_prior(BIG, RE_SQR, n_particles=N, generator=_gen(7))
    got = -log_prior(x, RE_SQR).mean().item()
    sigma2 = bond_sigma(RE_SQR, N) ** 2
    want = 0.5 * DOF * (1.0 + math.log(2.0 * math.pi * sigma2))
    # Var(log p) = DOF/2, so the std of the mean is sqrt(DOF/2 / BIG) ~ 0.012
    assert abs(got - want) < 0.06, f"{got} vs {want}"


def test_log_prior_is_o3_invariant():
    """Rotating (and reflecting) a configuration leaves its prior density unchanged."""
    from scipy.spatial.transform import Rotation

    x = sample_prior(16, RE_SQR, n_particles=N, generator=_gen(8))
    R = torch.tensor(Rotation.random(random_state=0).as_matrix(), dtype=x.dtype)
    lp = log_prior(x, RE_SQR)
    assert torch.allclose(log_prior(x @ R.T, RE_SQR), lp, atol=1e-10)
    assert torch.allclose(log_prior(-x, RE_SQR), lp, atol=1e-10)   # a reflection


def test_log_prior_reads_bonds_not_positions():
    """Shifting a whole chain leaves its density unchanged, because bonds are shift-free.

    Regression guard with a specific culprit in mind: a density written on raw
    positions, `-sum_k |x_k|^2 / 2 sigma^2`, would pass every other test in this file
    (it is still O(3)-invariant, still normalized on the subspace) and would fail only
    here. Translating the chain is the one probe that separates the two formulas.
    """
    x = sample_prior(8, RE_SQR, n_particles=N, generator=_gen(9))
    shifted = x + torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype)
    assert torch.allclose(log_prior(shifted, RE_SQR), log_prior(x, RE_SQR), atol=1e-10)
    # and re-anchoring undoes the shift exactly, which is what `load_ref_data` relies on
    assert torch.allclose(anchor(shifted), x, atol=1e-12)


# ---- observables and IO ----------------------------------------------------


def test_gyration_smaller_than_end_to_end():
    """For an ideal chain <Rg^2> = <Re^2>/6."""
    x = sample_prior(BIG, RE_SQR, n_particles=N, generator=_gen(10))
    ratio = (end_to_end_sq(x).mean() / gyration_sq(x).mean()).item()
    assert abs(ratio - 6.0) / 6.0 < 0.05, ratio


@pytest.mark.parametrize("flat", [True, False])
def test_load_ref_data_shapes_and_anchors(tmp_path, flat):
    """Accepts (M, N*3) or (M, N, 3), slices with `n`, and re-anchors."""
    raw = np.random.default_rng(0).normal(size=(50, N, 3)) + 5.0   # deliberately offset
    p = tmp_path / "tap.npy"
    np.save(p, raw.reshape(50, -1) if flat else raw)

    x = load_ref_data(p, n=10, n_particles=N)
    assert x.shape == (10, N, 3)
    assert x[:, 0].abs().max() < 1e-12
    # anchoring preserves internal geometry
    assert np.allclose(bond_vectors(x).numpy(), np.diff(raw[:10], axis=1), atol=1e-12)


def test_load_ref_data_missing_file_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="TAP reference data not found"):
        load_ref_data(tmp_path / "nope.npy")


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_end_to_end_matches_re_sqr,
        test_bond_mean_square_is_re_sqr_over_n_bonds,
        test_re_sqr_scaling_is_linear,
        test_bonds_are_iid_and_isotropic,
        test_positions_are_correlated_not_iid,
        test_tail_is_exactly_zero,
        test_anchor_is_idempotent_and_fixes_the_tail,
        test_dof_and_basis,
        test_log_prior_matches_scipy_on_the_bond_representation,
        test_log_prior_is_normalized_via_the_entropy,
        test_log_prior_is_o3_invariant,
        test_log_prior_reads_bonds_not_positions,
        test_gyration_smaller_than_end_to_end,
    ]
    import tempfile

    for _flat in (True, False):
        tests.append(lambda f=_flat: test_load_ref_data_shapes_and_anchors(
            Path(tempfile.mkdtemp()), f))
    tests.append(lambda: test_load_ref_data_missing_file_is_loud(Path(tempfile.mkdtemp())))

    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {getattr(t, '__name__', 'lambda')}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {getattr(t, '__name__', 'lambda')}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {getattr(t, '__name__', 'lambda')}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
