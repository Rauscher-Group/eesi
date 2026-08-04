"""Tests for `eesi.systems.tap.data`: the harmonic-bond prior and subspace geometry.

Runs as either pytest or a plain script:

    pytest tests/tap/test_tap_data.py
    python tests/tap/test_tap_data.py

The prior is where a silent convention error would hide -- a stray factor of 3 from
the spatial dimension, or (N-1) vs N bonds, produces a perfectly plausible-looking
chain with the wrong length scale, and nothing downstream would complain. So these
tests are quantitative and check the definition directly.

Three independent handles on the same law, which is what makes the file worth its
runtime. `bond_moments` is checked against numerical quadrature, `sample_bond_lengths`
against `bond_moments`, and `log_prior` against both -- via the entropy identity, which
is sensitive to the NORMALIZER specifically. A wrong Z1 would leave every relative
comparison intact and every absolute entropy off by a fixed offset, so it needs a test
that no downstream metric would notice.

The b = 0 slice is tested against the closed-form Gaussian it must reduce to. That is
the migration guard: the prior used to be an ideal chain parameterized by ReSqr, and
b = 0 with k = 3(N-1)/ReSqr has to reproduce it exactly, not approximately.
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
    bond_moments,
    bond_vectors,
    dof,
    end_to_end_mean_sq,
    end_to_end_sq,
    gyration_sq,
    load_ref_data,
    log_prior,
    prior_entropy,
    sample_bond_lengths,
    sample_prior,
    subspace_dirs,
)

N, K, B = 20, 5.0, 1.0
BIG = 200_000       # enough that the MC checks below are tight but still ~1 s

# Spanning pairs for the closed forms: b = 0 (the Maxwell edge case), stiff springs
# where the law is nearly Gaussian about b, and floppy ones where the Q^2 Jacobian
# still dominates and the truncation at Q = 0 actually bites.
PAIRS = [(1.0, 0.0), (K, B), (1.0, 2.5), (100.0, 1.0), (0.5, 3.0), (2.0, 0.3)]


def _gen(seed: int = 0):
    return torch.Generator().manual_seed(seed)


def _radial_quad(k, b, power):
    """int_0^inf Q^power exp(-(k/2)(Q-b)^2) dQ, numerically. The independent oracle."""
    from scipy.integrate import quad

    return quad(lambda q: q ** power * math.exp(-0.5 * k * (q - b) ** 2), 0, np.inf)[0]


# ---- the bond law, against quadrature --------------------------------------


def test_bond_moments_match_quadrature():
    """Z1, E[Q] and E[Q^2] agree with numerical integration of the radial law.

    The closed forms come from a recursion on truncated Gaussian moments; this is the
    check that the recursion, the binomial expansion and the sigma powers are all
    right, over a range wide enough that a wrong truncation term could not hide.
    """
    for k, b in PAIRS:
        z1, mean_q, mean_q_sq = bond_moments(k, b)
        assert abs(z1 - _radial_quad(k, b, 2)) < 1e-9 * z1, (k, b, z1)
        want_q = _radial_quad(k, b, 3) / _radial_quad(k, b, 2)
        want_q_sq = _radial_quad(k, b, 4) / _radial_quad(k, b, 2)
        assert abs(mean_q - want_q) < 1e-9 * want_q, (k, b, mean_q, want_q)
        assert abs(mean_q_sq - want_q_sq) < 1e-9 * want_q_sq, (k, b, mean_q_sq)


def test_bond_moments_satisfy_the_virial_identity():
    """E[Q^2] == b E[Q] + 3/k, exactly.

    Integration by parts on Q^3 exp(-(k/2)(Q-b)^2); the boundary term vanishes at both
    ends. An algebraic relation between the two moments, so it catches an error in one
    that quadrature agreement in the other would not localize.
    """
    for k, b in PAIRS:
        _, mean_q, mean_q_sq = bond_moments(k, b)
        want = b * mean_q + 3.0 / k
        assert abs(mean_q_sq - want) < 1e-10 * max(1.0, want), (k, b, mean_q_sq, want)


def test_sampled_bond_lengths_match_the_moments():
    """The rejection sampler reproduces the closed-form moments, and never returns Q <= 0.

    The sampler and `bond_moments` share no code, so this is a genuine cross-check of
    the acceptance criterion -- a wrong log-acceptance would tilt the distribution
    while leaving it perfectly plausible-looking.
    """
    for k, b in PAIRS:
        _, mean_q, mean_q_sq = bond_moments(k, b)
        q = sample_bond_lengths(100_000, k, b, generator=_gen(0))
        assert q.shape == (100_000,)
        assert (q > 0).all(), (k, b)
        assert abs(q.mean().item() - mean_q) < 0.015 * mean_q, (k, b, q.mean().item())
        assert abs((q ** 2).mean().item() - mean_q_sq) < 0.02 * mean_q_sq, (k, b)


def test_sampler_is_reproducible_under_a_fixed_generator():
    """Rejection consumes a data-dependent number of variates, but is still seedable."""
    a = sample_bond_lengths(1000, K, B, generator=_gen(3))
    b = sample_bond_lengths(1000, K, B, generator=_gen(3))
    assert torch.equal(a, b)
    assert not torch.equal(a, sample_bond_lengths(1000, K, B, generator=_gen(4)))


# ---- the chain built from it ------------------------------------------------


def test_end_to_end_matches_the_analytic_mean_square():
    """E|x_{N-1} - x_0|^2 == (N-1) E[Q^2], the iid-bond prediction.

    Catches an off-by-one in the bond count (would give ~N/(N-1)x) and any mismatch
    between the magnitudes the sampler draws and the law `bond_moments` describes.
    """
    x = sample_prior(BIG, K, B, n_particles=N, generator=_gen(0))
    got = end_to_end_sq(x).mean().item()
    want = end_to_end_mean_sq(K, B, N)
    assert abs(got - want) / want < 0.02, f"E[Re^2]={got}, want {want}"


def test_bond_mean_square_is_the_analytic_moment():
    """E|bond|^2 == E[Q^2], i.e. the direction factor is a genuine unit vector."""
    x = sample_prior(BIG, K, B, n_particles=N, generator=_gen(1))
    got = (bond_vectors(x) ** 2).sum(-1).mean().item()
    want = bond_moments(K, B)[2]
    assert abs(got - want) / want < 0.01, (got, want)


def test_growing_b_grows_the_chain():
    """At fixed k, a longer equilibrium bond gives a longer chain -- analytically and
    in the samples, together.

    The replacement for the old linear-ReSqr-scaling test. The prior is no longer a
    scale family, so there is no exact scaling law left to assert; what survives is
    monotonicity, plus the requirement that the closed form and the sampler move in
    lockstep rather than merely both increasing.
    """
    sizes = [end_to_end_mean_sq(K, b, N) for b in (0.0, 1.0, 2.0)]
    assert sizes[0] < sizes[1] < sizes[2], sizes
    for b, want in zip((0.0, 1.0, 2.0), sizes):
        x = sample_prior(50_000, K, b, n_particles=N, generator=_gen(2))
        assert abs(end_to_end_sq(x).mean().item() - want) / want < 0.03, (b, want)


def test_bonds_are_iid_and_isotropic():
    """Bond vectors are uncorrelated across bonds and isotropic within a bond.

    The prior has no bending rigidity; a nonzero bond-bond correlation would mean the
    sampler had accidentally correlated successive increments. Isotropy is separately
    load-bearing: it is what makes the prior O(3)-invariant, and hence what makes the
    alignment layer of `eesi.systems.tap.ot` marginal-preserving.
    """
    bonds = bond_vectors(sample_prior(100_000, K, B, n_particles=N, generator=_gen(3)))
    var = bonds.var(dim=(0, 1))                   # per-component
    assert torch.allclose(var, var.mean().expand(3), rtol=0.05), var.tolist()
    corr = (bonds[:, :-1] * bonds[:, 1:]).sum(-1).mean().item()
    assert abs(corr) < 0.01 * bond_moments(K, B)[2]


def test_positions_are_correlated_not_iid():
    """Positions are a cumulative sum, so they are NOT iid -- var grows along the chain.

    Guards against a sampler that forgot the cumsum and returned raw bonds as
    positions, which would still pass a naive per-bond variance check.
    """
    x = sample_prior(50_000, K, B, n_particles=N, generator=_gen(4))
    v = (x ** 2).sum(-1).mean(0)                  # E|x_k|^2 per particle
    assert v[1] < v[N // 2] < v[-1]
    # E|x_k|^2 = k E[Q^2]: the bonds are iid and isotropic, so cross terms vanish
    expect = torch.arange(N, dtype=v.dtype) * bond_moments(K, B)[2]
    assert torch.allclose(v, expect, rtol=0.05, atol=1e-3), v.tolist()


# ---- the subspace ----------------------------------------------------------


def test_tail_is_exactly_zero():
    """Row 0 is a literal zero, not a zero up to roundoff."""
    x = sample_prior(64, K, B, n_particles=N, generator=_gen(5))
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


def test_log_prior_normalizer_matches_quadrature():
    """The constant `log_prior` subtracts is exactly (N-1) log(4 pi Z1).

    Peels the quadratic term off an actual evaluation and compares what is left to
    quadrature, so the normalizer is checked without going through `bond_moments`.
    The 4 pi is the point: dropping it (integrating the radial law and forgetting the
    sphere) is the single most likely error here, and it is a clean (N-1) log 4 pi
    offset that nothing else in this file would catch.
    """
    n = 6
    x = sample_prior(32, K, B, n_particles=n, generator=_gen(6))
    q = torch.linalg.vector_norm(bond_vectors(x), dim=-1)
    quad_term = -0.5 * K * ((q - B) ** 2).sum(dim=-1)
    const = (log_prior(x, K, B) - quad_term).numpy()
    want = -(n - 1) * math.log(4.0 * math.pi * _radial_quad(K, B, 2))
    assert np.allclose(const, want, atol=1e-9), (const[:3], want)


def test_log_prior_is_normalized_via_the_entropy():
    """-E[log p] equals `prior_entropy`, the closed-form S[p0].

    An independent check of the NORMALIZER: the quadratic term could be right while
    the constant is wrong, and every relative comparison would still work while every
    absolute entropy -- the number the whole TAP experiment reports -- would be off by
    a fixed offset. Also the only test of `prior_entropy` against sampled data.
    """
    for k, b in [(K, B), (1.0, 0.0), (2.0, 2.0)]:
        x = sample_prior(BIG, k, b, n_particles=N, generator=_gen(7))
        got = -log_prior(x, k, b).mean().item()
        want = prior_entropy(k, b, N)
        # the std of the mean is ~0.012 at these sizes; 0.06 is ample room
        assert abs(got - want) < 0.06, f"k={k} b={b}: {got} vs {want}"


def test_log_prior_is_o3_invariant():
    """Rotating (and reflecting) a configuration leaves its prior density unchanged."""
    from scipy.spatial.transform import Rotation

    x = sample_prior(16, K, B, n_particles=N, generator=_gen(8))
    R = torch.tensor(Rotation.random(random_state=0).as_matrix(), dtype=x.dtype)
    lp = log_prior(x, K, B)
    assert torch.allclose(log_prior(x @ R.T, K, B), lp, atol=1e-10)
    assert torch.allclose(log_prior(-x, K, B), lp, atol=1e-10)   # a reflection


def test_log_prior_reads_bonds_not_positions():
    """Shifting a whole chain leaves its density unchanged, because bonds are shift-free.

    Regression guard with a specific culprit in mind: a density written on raw
    positions would pass every other test in this file (it is still O(3)-invariant,
    still normalized on the subspace) and would fail only here. Translating the chain
    is the one probe that separates the two formulas.
    """
    x = sample_prior(8, K, B, n_particles=N, generator=_gen(9))
    shifted = x + torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype)
    assert torch.allclose(log_prior(shifted, K, B), log_prior(x, K, B), atol=1e-10)
    # and re-anchoring undoes the shift exactly, which is what `load_ref_data` relies on
    assert torch.allclose(anchor(shifted), x, atol=1e-12)


# ---- b = 0: the ideal chain the prior used to be ---------------------------


def test_b_zero_reproduces_the_ideal_chain_density():
    """At b = 0 the bond law is N(0, I/k), so log_prior IS the Gaussian on the bonds.

    The migration guard, and what is left of the old scipy comparison: it still checks
    the module's central claim, that the cumsum from bonds to positions is unit lower
    triangular and needs no Jacobian term. That argument never mentioned the bond law,
    so the b = 0 slice tests it just as well as the Gaussian prior did.
    """
    from scipy.stats import multivariate_normal

    n, k = 6, 3.0
    sigma = 1.0 / math.sqrt(k)
    x = sample_prior(32, k, 0.0, n_particles=n, generator=_gen(10))
    bonds = bond_vectors(x).reshape(32, -1).numpy()
    ref = multivariate_normal.logpdf(bonds, mean=np.zeros(bonds.shape[1]), cov=sigma ** 2)
    got = log_prior(x, k, 0.0).numpy()
    assert np.allclose(got, ref, atol=1e-9), (got[:3], ref[:3])


def test_b_zero_reproduces_the_ideal_chain_constants():
    """The closed forms collapse to the Gaussian ones, and ReSqr maps onto k exactly.

    Pins the correspondence the docstrings claim: the old ReSqr parameterization is
    the b = 0 slice with k = 3(N-1)/ReSqr. Anyone reproducing an old run needs this to
    be exact, not close.
    """
    k = 3.0
    sigma_sq = 1.0 / k
    assert abs(bond_moments(k, 0.0)[2] - 3.0 * sigma_sq) < 1e-12
    want_entropy = 0.5 * DOF * (1.0 + math.log(2.0 * math.pi * sigma_sq))
    assert abs(prior_entropy(k, 0.0, N) - want_entropy) < 1e-9

    re_sqr = 44.2769
    k_equiv = 3.0 * (N - 1) / re_sqr
    assert abs(end_to_end_mean_sq(k_equiv, 0.0, N) - re_sqr) < 1e-9


# ---- observables and IO ----------------------------------------------------


def test_gyration_smaller_than_end_to_end():
    """Rg^2 < Re^2 always; and the ideal chain's <Re^2>/<Rg^2> = 6 survives at b = 0.

    The ratio-6 identity is ideal-chain-only -- it comes from the Gaussian chain's
    continuum limit -- so at finite b it is not expected to hold and is not asserted.
    """
    x = sample_prior(BIG, K, B, n_particles=N, generator=_gen(11))
    assert gyration_sq(x).mean().item() < end_to_end_sq(x).mean().item()

    x0 = sample_prior(BIG, 1.0, 0.0, n_particles=N, generator=_gen(12))
    ratio = (end_to_end_sq(x0).mean() / gyration_sq(x0).mean()).item()
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


# ---- invalid input is loud -------------------------------------------------


def test_invalid_bond_parameters_are_loud():
    """k <= 0 and b < 0 are rejected, in the sampler as well as the closed forms.

    k <= 0 is not merely meaningless: it would send the rejection loop in
    `sample_bond_lengths` spinning forever rather than failing, so it has to be caught
    at the door.
    """
    for k, b in [(0.0, 1.0), (-1.0, 1.0), (1.0, -0.5)]:
        with pytest.raises(ValueError):
            bond_moments(k, b)
        with pytest.raises(ValueError):
            sample_bond_lengths(4, k, b)


def test_non_3d_is_rejected():
    """The prior is 3D-only; the Q^2 Jacobian says so and the error message explains it."""
    with pytest.raises(ValueError, match="n_dims=3 only"):
        sample_prior(4, K, B, n_particles=5, n_dims=2)
    with pytest.raises(ValueError, match="n_dims=3 only"):
        prior_entropy(K, B, n_particles=5, n_dims=2)
    with pytest.raises(ValueError, match="n_dims=3 only"):
        log_prior(torch.zeros(2, 5, 2, dtype=torch.float64), K, B)


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_bond_moments_match_quadrature,
        test_bond_moments_satisfy_the_virial_identity,
        test_sampled_bond_lengths_match_the_moments,
        test_sampler_is_reproducible_under_a_fixed_generator,
        test_end_to_end_matches_the_analytic_mean_square,
        test_bond_mean_square_is_the_analytic_moment,
        test_growing_b_grows_the_chain,
        test_bonds_are_iid_and_isotropic,
        test_positions_are_correlated_not_iid,
        test_tail_is_exactly_zero,
        test_anchor_is_idempotent_and_fixes_the_tail,
        test_dof_and_basis,
        test_log_prior_normalizer_matches_quadrature,
        test_log_prior_is_normalized_via_the_entropy,
        test_log_prior_is_o3_invariant,
        test_log_prior_reads_bonds_not_positions,
        test_b_zero_reproduces_the_ideal_chain_density,
        test_b_zero_reproduces_the_ideal_chain_constants,
        test_gyration_smaller_than_end_to_end,
        test_invalid_bond_parameters_are_loud,
        test_non_3d_is_rejected,
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
