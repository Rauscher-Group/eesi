"""Tests for `eesi.systems.tap.data`: the semiflexible prior and subspace geometry.

Runs as either pytest or a plain script:

    pytest tests/tap/test_tap_data.py
    python tests/tap/test_tap_data.py

The prior is where a silent convention error would hide -- a stray factor of 3 from
the spatial dimension, or (N-1) vs N bonds, produces a perfectly plausible-looking
chain with the wrong length scale, and nothing downstream would complain. So these
tests are quantitative and check the definition directly.

The prior has two independent laws -- a radial one for the bond magnitudes and an
angular one for the bends -- and each gets the same three independent handles, which is
what makes the file worth its runtime. `bond_moments` and `angle_moments` are checked
against numerical quadrature, the two samplers against those closed forms, and
`log_prior` against everything at once via the entropy identity, which is sensitive to
the NORMALIZER specifically. A wrong Z1 or Z_ang, or the 4 pi / 2 pi split applied to the
wrong count of bonds, would leave every relative comparison intact and every absolute
entropy off by a fixed offset -- so it needs a test that no downstream metric would
notice.

Two nested migration guards, both required to be exact rather than close, so that older
runs stay reproducible:

    gamma = 0            the freely-jointed harmonic-bond chain
    gamma = 0 and b = 0  the original ideal chain, ReSqr = 3(N-1)/k

The sharpest test of the direction sampler is not any moment but the Markov identity
<u_i . u_j> = <u>^|i-j|. A frame construction that is subtly wrong still reproduces the
nearest-neighbour correlation and fails at longer lags.
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
    angle_moments,
    bending_deltas,
    bond_cosines,
    bond_moments,
    bond_vectors,
    dof,
    end_to_end_mean_sq,
    end_to_end_sq,
    gyration_sq,
    load_ref_data,
    log_prior,
    prior_energy,
    prior_entropy,
    prior_free_energy,
    rouse_mode_moments,
    rouse_modes,
    sample_bond_cosines,
    sample_bond_lengths,
    sample_prior,
    solve_cos_theta_0,
    subspace_dirs,
)

N, K, B = 20, 5.0, 1.0
G, C0 = 2.0, 0.5                    # the prior's bending constant and equilibrium cosine
BIG = 200_000       # enough that the MC checks below are tight but still ~1 s

# Spanning pairs for the radial closed forms: b = 0 (the Maxwell edge case), stiff springs
# where the law is nearly Gaussian about b, and floppy ones where the Q^2 Jacobian
# still dominates and the truncation at Q = 0 actually bites.
PAIRS = [(1.0, 0.0), (K, B), (1.0, 2.5), (100.0, 1.0), (0.5, 3.0), (2.0, 0.3)]

# And for the angular ones: gamma = 0 (uniform), the stiff limit where the truncation is
# irrelevant, and floppy cases where it dominates -- including u_0 at both endpoints of
# its permitted range, where one tail is cut off entirely.
ANGLES = [(0.0, 0.0), (G, C0), (1.0, 0.5), (5.0, 0.9), (50.0, 0.95), (0.3, -0.4),
          (200.0, 1.0), (2.0, 0.0), (10.0, -1.0)]


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


def _angular_quad(gamma, cos_theta_0, weight):
    """int_{-1}^{1} weight(u) exp(-gamma (u-u_0)^2) du, numerically. The oracle."""
    from scipy.integrate import quad

    f = lambda u: weight(u) * math.exp(-gamma * (u - cos_theta_0) ** 2)
    return quad(f, -1.0, 1.0)[0]


def test_angle_moments_match_quadrature():
    """Z_ang, E[u] and E[(u-u_0)^2] agree with numerical integration of the bending law.

    The angular counterpart of `test_bond_moments_match_quadrature`. Spans gamma = 0
    (uniform), the stiff limit, and u_0 at both ends of its range, where the truncation
    removes one tail entirely and the standard-normal formulas are least forgiving.
    """
    for gamma, c0 in ANGLES:
        z_ang, mean_u, mean_sq_dev = angle_moments(gamma, c0)
        norm = _angular_quad(gamma, c0, lambda u: 1.0)
        want_u = _angular_quad(gamma, c0, lambda u: u) / norm
        want_dev = _angular_quad(gamma, c0, lambda u: (u - c0) ** 2) / norm
        assert abs(z_ang - norm) < 1e-9 * norm, (gamma, c0, z_ang, norm)
        assert abs(mean_u - want_u) < 1e-9 * max(1e-3, abs(want_u)), (gamma, c0, mean_u)
        assert abs(mean_sq_dev - want_dev) < 1e-9 * want_dev, (gamma, c0, mean_sq_dev)


def test_angle_moments_at_zero_gamma_are_the_uniform_values():
    """gamma = 0 is branched, not a limit, so its exact values are pinned separately.

    Z_ang = 2, E[u] = 0 and E[(u-u_0)^2] = 1/3 + u_0^2 for the uniform law on [-1, 1].
    Note the last still depends on u_0 -- it is the second moment about the potential's
    centre, not the distribution's own -- and a version that dropped the u_0^2 would
    still pass every test that only looks at samples.
    """
    for c0 in (0.0, 0.5, -1.0, 1.0):
        z_ang, mean_u, mean_sq_dev = angle_moments(0.0, c0)
        assert z_ang == 2.0 and mean_u == 0.0
        assert abs(mean_sq_dev - (1.0 / 3.0 + c0 ** 2)) < 1e-15


def test_sampled_bond_cosines_match_the_moments():
    """The inverse-CDF draw reproduces the closed forms and stays inside [-1, 1].

    The [-1, 1] postcondition is not cosmetic: `sample_prior` feeds these to
    sqrt(1 - u^2), which would return NaN on any excursion. Includes gamma = 500, stiff
    enough that the erfinv argument saturates and the clamps actually engage.
    """
    for gamma, c0 in [(G, C0), (20.0, 1.0), (0.0, 0.0), (5.0, -0.3), (500.0, 0.8)]:
        _, mean_u, mean_sq_dev = angle_moments(gamma, c0)
        u = sample_bond_cosines(200_000, gamma, c0, generator=_gen(0))
        assert u.shape == (200_000,)
        assert (u >= -1.0).all() and (u <= 1.0).all(), (gamma, c0)
        assert abs(u.mean().item() - mean_u) < 0.01 * max(0.05, abs(mean_u)), (gamma, c0)
        got_dev = ((u - c0) ** 2).mean().item()
        assert abs(got_dev - mean_sq_dev) < 0.03 * mean_sq_dev, (gamma, c0, got_dev)


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
    x = sample_prior(BIG, K, B, G, C0, n_particles=N, generator=_gen(0))
    got = end_to_end_sq(x).mean().item()
    want = end_to_end_mean_sq(K, B, G, C0, N)
    assert abs(got - want) / want < 0.02, f"E[Re^2]={got}, want {want}"


def test_bond_mean_square_is_the_analytic_moment():
    """E|bond|^2 == E[Q^2], i.e. the direction factor is a genuine unit vector."""
    x = sample_prior(BIG, K, B, G, C0, n_particles=N, generator=_gen(1))
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
    sizes = [end_to_end_mean_sq(K, b, G, C0, N) for b in (0.0, 1.0, 2.0)]
    assert sizes[0] < sizes[1] < sizes[2], sizes
    for b, want in zip((0.0, 1.0, 2.0), sizes):
        x = sample_prior(50_000, K, b, G, C0, n_particles=N, generator=_gen(2))
        assert abs(end_to_end_sq(x).mean().item() - want) / want < 0.03, (b, want)


def test_bond_directions_are_markov():
    """<u_i . u_j> == <u>^|i-j| across lags, the freely-rotating-chain identity.

    THE test of the sampler's frame construction, and the reason it is worth stating
    separately from any moment check. The uniform azimuth averages away every component
    perpendicular to the previous bond, which is what makes the directions an exact
    Markov chain; a perpendicular basis that is subtly wrong -- not orthogonal to the
    previous bond, or not orthonormal to itself -- still reproduces the lag-1
    correlation and goes astray from lag 2 on.

    Checked at the working parameters and in the stiff regime, where the powers stay
    far from zero and a wrong decay rate has nowhere to hide.
    """
    for gamma, c0 in [(G, C0), (20.0, 1.0), (1.0, -0.3)]:
        x = sample_prior(100_000, K, B, gamma, c0, n_particles=N, generator=_gen(3))
        u = bond_vectors(x)
        u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        mean_u = angle_moments(gamma, c0)[1]
        for lag in (1, 2, 3, 5):
            got = (u[:, :-lag] * u[:, lag:]).sum(-1).mean().item()
            assert abs(got - mean_u ** lag) < 0.006, (gamma, c0, lag, got, mean_u ** lag)


def test_perpendicular_basis_is_orthonormal_including_at_the_poles():
    """(e1, e2, u) is an orthonormal triad for every input, the pole included.

    White-box, and deliberately so: the pole is where this can fail and it is exactly
    where random sampling never looks. Projecting z-hat out of a direction parallel to
    z-hat leaves the zero vector, and normalizing that is NaN -- which is why
    `_perpendicular_basis` switches to x-hat once |u_z| exceeds 0.9. A statistical test
    cannot catch the regression: a chain of 200k samples has probability ~0 of landing
    close enough to the axis to notice, so the guard would rot silently.

    The near-pole cases matter too, and for a different reason: there the residual is
    tiny rather than zero, so normalizing it amplifies roundoff into a direction that is
    no longer perpendicular to anything in particular.
    """
    from eesi.systems.tap.data import _perpendicular_basis

    u = torch.tensor([
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],           # the poles themselves
        [1e-9, 0.0, 1.0], [0.0, -1e-9, -1.0],        # just off them
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],            # the equator
        [0.5773502691896258] * 3,                    # a generic direction
    ], dtype=torch.float64)
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
    e1, e2 = _perpendicular_basis(u)

    assert torch.isfinite(e1).all() and torch.isfinite(e2).all(), (e1, e2)
    one = torch.ones(u.shape[0], dtype=u.dtype)
    assert torch.allclose(torch.linalg.vector_norm(e1, dim=-1), one, atol=1e-12)
    assert torch.allclose(torch.linalg.vector_norm(e2, dim=-1), one, atol=1e-12)
    zero = torch.zeros(u.shape[0], dtype=u.dtype)
    assert torch.allclose((e1 * u).sum(-1), zero, atol=1e-12), "e1 not perpendicular to u"
    assert torch.allclose((e2 * u).sum(-1), zero, atol=1e-12), "e2 not perpendicular to u"
    assert torch.allclose((e1 * e2).sum(-1), zero, atol=1e-12), "e1 not perpendicular to e2"


def test_bonds_are_isotropic_and_correlated():
    """Each bond is isotropic on its own; successive bonds are correlated by the bending.

    Isotropy is load-bearing and survives the bending potential: the first bond is drawn
    uniformly and every later one is defined relative to it, so each bond's MARGINAL is
    still uniform on the sphere. That is what keeps the prior O(3)-invariant and hence
    the alignment layer of `eesi.systems.tap.ot` marginal-preserving.

    What does NOT survive is independence. <b_i . b_j> = E[Q]^2 <u>^|i-j| now, and the
    lag-1 value is asserted against that rather than against zero -- the old test
    demanded it vanish, which is exactly the behaviour this prior is designed not to
    have. At gamma = 0 it does vanish, and that case is checked too.
    """
    _, mean_q, _ = bond_moments(K, B)
    for gamma, c0 in [(G, C0), (0.0, 1.0)]:
        bonds = bond_vectors(sample_prior(100_000, K, B, gamma, c0, n_particles=N,
                                          generator=_gen(3)))
        var = bonds.var(dim=(0, 1))                   # per-component
        assert torch.allclose(var, var.mean().expand(3), rtol=0.05), var.tolist()
        got = (bonds[:, :-1] * bonds[:, 1:]).sum(-1).mean().item()
        want = mean_q ** 2 * angle_moments(gamma, c0)[1]
        assert abs(got - want) < 0.01 * bond_moments(K, B)[2] + 0.005, (gamma, got, want)


def test_positions_are_correlated_not_iid():
    """Positions are a cumulative sum, so they are NOT iid -- var grows along the chain.

    Guards against a sampler that forgot the cumsum and returned raw bonds as positions,
    which would still pass a naive per-bond variance check.

    E|x_j|^2 is just the mean squared end-to-end distance of the first j bonds, so the
    expectation comes from `end_to_end_mean_sq` at n_particles = j+1. That reuse is the
    point: with correlated bonds there is no longer a simple j E[Q^2] to write down, and
    a hand-rolled second formula here could drift from the one the module ships.
    """
    x = sample_prior(50_000, K, B, G, C0, n_particles=N, generator=_gen(4))
    v = (x ** 2).sum(-1).mean(0)                  # E|x_j|^2 per particle
    assert v[1] < v[N // 2] < v[-1]
    expect = torch.tensor([0.0] + [end_to_end_mean_sq(K, B, G, C0, j + 1)
                                   for j in range(1, N)], dtype=v.dtype)
    assert torch.allclose(v, expect, rtol=0.05, atol=1e-3), v.tolist()


# ---- the subspace ----------------------------------------------------------


def test_tail_is_exactly_zero():
    """Row 0 is a literal zero, not a zero up to roundoff."""
    x = sample_prior(64, K, B, G, C0, n_particles=N, generator=_gen(5))
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
    """The constant is exactly log(4 pi Z1) + (N-2) log(2 pi Z1 Z_ang).

    Peels BOTH quadratic terms off an actual evaluation and compares what is left to
    quadrature, so the normalizer is checked without going through `bond_moments` or
    `angle_moments` at all.

    The 4 pi / 2 pi split is the point, and it is the likeliest thing to get wrong here.
    Only the first bond normalizes over the full sphere; for the other N-2 the polar
    integral is already inside Z_ang, leaving just the azimuth. Using 4 pi throughout
    would be a clean (N-2) log 2 offset, and using N-1 angular factors instead of N-2
    another fixed shift -- both invisible to every relative comparison, and to every
    test in this file that does not look at the constant itself.
    """
    n = 6
    x = sample_prior(32, K, B, G, C0, n_particles=n, generator=_gen(6))
    q = torch.linalg.vector_norm(bond_vectors(x), dim=-1)
    radial = -0.5 * K * ((q - B) ** 2).sum(dim=-1)
    bending = -G * ((bond_cosines(x) - C0) ** 2).sum(dim=-1)
    const = (log_prior(x, K, B, G, C0) - radial - bending).numpy()

    z1 = _radial_quad(K, B, 2)
    z_ang = _angular_quad(G, C0, lambda u: 1.0)
    want = -(math.log(4.0 * math.pi * z1)
             + (n - 2) * math.log(2.0 * math.pi * z1 * z_ang))
    assert np.allclose(const, want, atol=1e-9), (const[:3], want)


def test_log_prior_is_normalized_via_the_entropy():
    """-E[log p] equals `prior_entropy`, the closed-form S[p0].

    An independent check of the NORMALIZER: the quadratic term could be right while
    the constant is wrong, and every relative comparison would still work while every
    absolute entropy -- the number the whole TAP experiment reports -- would be off by
    a fixed offset. Also the only test of `prior_entropy` against sampled data.
    """
    for k, b in [(K, B), (1.0, 0.0), (2.0, 2.0)]:
        x = sample_prior(BIG, k, b, G, C0, n_particles=N, generator=_gen(7))
        got = -log_prior(x, k, b, G, C0).mean().item()
        want = prior_entropy(k, b, G, C0, N)
        # the std of the mean is ~0.012 at these sizes; 0.06 is ample room
        assert abs(got - want) < 0.06, f"k={k} b={b}: {got} vs {want}"


def test_log_prior_is_o3_invariant():
    """Rotating (and reflecting) a configuration leaves its prior density unchanged."""
    from scipy.spatial.transform import Rotation

    x = sample_prior(16, K, B, G, C0, n_particles=N, generator=_gen(8))
    R = torch.tensor(Rotation.random(random_state=0).as_matrix(), dtype=x.dtype)
    lp = log_prior(x, K, B, G, C0)
    assert torch.allclose(log_prior(x @ R.T, K, B, G, C0), lp, atol=1e-10)
    assert torch.allclose(log_prior(-x, K, B, G, C0), lp, atol=1e-10)   # a reflection


def test_log_prior_reads_bonds_not_positions():
    """Shifting a whole chain leaves its density unchanged, because bonds are shift-free.

    Regression guard with a specific culprit in mind: a density written on raw
    positions would pass every other test in this file (it is still O(3)-invariant,
    still normalized on the subspace) and would fail only here. Translating the chain
    is the one probe that separates the two formulas.
    """
    x = sample_prior(8, K, B, G, C0, n_particles=N, generator=_gen(9))
    shifted = x + torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype)
    assert torch.allclose(log_prior(shifted, K, B, G, C0), log_prior(x, K, B, G, C0), atol=1e-10)
    # and re-anchoring undoes the shift exactly, which is what `load_ref_data` relies on
    assert torch.allclose(anchor(shifted), x, atol=1e-12)


# ---- the thermodynamic potentials and the bending bridge -------------------


@pytest.mark.parametrize("k,b", PAIRS)
@pytest.mark.parametrize("gamma,c0", ANGLES)
@pytest.mark.parametrize("n", [3, 5, N])
def test_entropy_is_energy_minus_free_energy(k, b, gamma, c0, n):
    """S = <U> - F, because p0 is Boltzmann and S = <U> + log Z.

    The sharpest available check that the three closed forms agree on the 4 pi / 2 pi
    split and on the N-1 / N-2 counts: `prior_free_energy` carries the whole normalizer
    and `prior_energy` none of it, so any slip in either lands here. n = 3 is the
    smallest chain with a bend at all (one angle), where an off-by-one in the angular
    count is a large relative error rather than a small one.
    """
    got = prior_energy(k, b, gamma, c0, n) - prior_free_energy(k, b, gamma, c0, n)
    assert abs(got - prior_entropy(k, b, gamma, c0, n)) < 1e-9


@pytest.mark.parametrize("gamma,c0", ANGLES)
def test_bending_deltas_match_the_prior_differences(gamma, c0):
    """The bridge equals the difference of the two states, at every (k, b).

    Two claims in one: the closed form is right, and it really is independent of the
    bond law -- the radial factors cancel term by term because p(Q) is identical in the
    freely-jointed and semiflexible states. `bending_deltas` computes only the angular
    part, so a k or b leaking into it would show up as a spread across PAIRS.
    """
    d_s, d_u, d_f = bending_deltas(gamma, c0, N)
    for k, b in PAIRS:
        # The reference state's cos_theta_0 is inert at gamma = 0 (pinned by
        # test_bending_deltas_vanish_at_zero_gamma), so 0.0 here is not a choice.
        assert abs((prior_entropy(k, b, gamma, c0, N)
                    - prior_entropy(k, b, 0.0, 0.0, N)) - d_s) < 1e-9, (k, b)
        assert abs((prior_energy(k, b, gamma, c0, N)
                    - prior_energy(k, b, 0.0, 0.0, N)) - d_u) < 1e-9, (k, b)
        assert abs((prior_free_energy(k, b, gamma, c0, N)
                    - prior_free_energy(k, b, 0.0, 0.0, N)) - d_f) < 1e-9, (k, b)


def test_bending_deltas_vanish_at_zero_gamma():
    """No bending potential, no shift -- exactly, and for any cos_theta_0.

    The reference state of the bridge is the gamma = 0 chain itself, so this is the
    identity element. Exact rather than close: log(Z_ang / 2) = log 1 = 0 and the energy
    term carries an explicit factor of gamma.
    """
    for c0 in (-1.0, 0.0, 0.5, 1.0):
        assert bending_deltas(0.0, c0, N) == (0.0, 0.0, 0.0), c0


@pytest.mark.parametrize("gamma,c0", ANGLES)
def test_bending_free_energy_matches_quadrature(gamma, c0):
    """dF against the free-energy perturbation it is: -log <exp(-U_bend)> over uniform u.

    The angular reference state is uniform on [-1, 1], so per angle
    <exp(-gamma (u-u_0)^2)> = Z_ang / 2, and dF = -(N-2) log of that. Done by
    deterministic quadrature on ONE angle and then scaled, which is both exact to ~1e-9
    and the right way round: a whole-chain FEP over 18 angles is variance-dominated
    (2e5 chains gives 12.78 against the true 12.88 -- useless as a test).
    """
    from scipy.integrate import quad

    mean_boltzmann = 0.5 * quad(lambda u: math.exp(-gamma * (u - c0) ** 2), -1.0, 1.0)[0]
    want = -(N - 2) * math.log(mean_boltzmann)
    assert abs(bending_deltas(gamma, c0, N)[2] - want) < 1e-9


def test_bending_energy_matches_prior_samples():
    """dU against the mean bending energy of an actual prior draw.

    Ties the analytic moment to the sampler that produces the training data: an
    E[(u-u_0)^2] taken about the wrong centre (the distribution's mean rather than the
    potential's) would pass every algebraic test above and fail here.
    """
    for gamma, c0 in [(G, C0), (5.0, 0.9), (0.3, -0.4)]:
        x = sample_prior(BIG, K, B, gamma, c0, n_particles=N, generator=_gen(21))
        got = (gamma * ((bond_cosines(x) - c0) ** 2).sum(-1)).mean().item()
        want = bending_deltas(gamma, c0, N)[1]
        assert abs(got - want) < 0.02 * abs(want) + 1e-6, (gamma, c0, got, want)


@pytest.mark.parametrize("gamma,c0", [(g, c) for g, c in ANGLES if g > 0.0])
def test_bending_deltas_signs(gamma, c0):
    """Switching on a constraint: entropy falls, free energy and energy rise.

    Worth pinning because it is the statement a reader will take on trust when reading a
    reconciliation that comes out with the wrong sign.
    """
    d_s, d_u, d_f = bending_deltas(gamma, c0, N)
    assert d_s < 0.0 and d_u > 0.0 and d_f > 0.0, (d_s, d_u, d_f)


# ---- the nested priors this one must still reproduce -----------------------


def test_gamma_zero_reproduces_the_freely_jointed_chain():
    """At gamma = 0 the bending drops out and the closed forms collapse to the old ones.

    The migration guard for THIS change, and required exact rather than close: the
    freely-jointed formulas are what the previous prior shipped, and a run made against
    them has to stay reproducible. Written out here rather than imported so the test
    fails if someone edits the general formula in a way that breaks the special case.

        E[Re^2] = (N-1) E[Q^2]                             no cross terms
        S       = (N-1) [log(4 pi Z1) + 3/2 - (k b/2)(E[Q] - b)]
        log p0  = -sum (k/2)(Q-b)^2 - (N-1) log(4 pi Z1)

    The entropy one is the sharp case: at gamma = 0, log Z_ang = log 2 has to absorb
    the per-angle 2 pi back into a 4 pi, so an off-by-one in the N-2 angular count
    shows up here as a clean multiple of log 2.
    """
    for k, b in [(K, B), (91.46, 1.0241), (3.0, 0.0)]:
        z1, mean_q, mean_q_sq = bond_moments(k, b)

        assert abs(end_to_end_mean_sq(k, b, 0.0, 1.0, N) - (N - 1) * mean_q_sq) < 1e-9
        want_s = (N - 1) * (math.log(4.0 * math.pi * z1) + 1.5
                            - 0.5 * k * b * (mean_q - b))
        assert abs(prior_entropy(k, b, 0.0, 1.0, N) - want_s) < 1e-9, (k, b)

        x = sample_prior(64, k, b, 0.0, 1.0, n_particles=N, generator=_gen(20))
        q = torch.linalg.vector_norm(bond_vectors(x), dim=-1)
        want_lp = (-0.5 * k * ((q - b) ** 2).sum(-1)
                   - (N - 1) * math.log(4.0 * math.pi * z1))
        assert torch.allclose(log_prior(x, k, b, 0.0, 1.0), want_lp, atol=1e-10), (k, b)


def test_gamma_zero_is_independent_of_cos_theta_0():
    """With no bending there is no equilibrium angle, so cos_theta_0 must not matter.

    It still enters `angle_moments`' third return, E[(u-u_0)^2] = 1/3 + u_0^2, so the
    entropy has a u_0-dependent term that is multiplied by gamma. If that multiplication
    were dropped the value would silently drift with a parameter that has no meaning at
    gamma = 0.
    """
    vals = [prior_entropy(K, B, 0.0, c0, N) for c0 in (-1.0, 0.0, 0.5, 1.0)]
    assert max(vals) - min(vals) == 0.0, vals
    sizes = [end_to_end_mean_sq(K, B, 0.0, c0, N) for c0 in (-1.0, 0.0, 0.5, 1.0)]
    assert max(sizes) - min(sizes) == 0.0, sizes


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
    x = sample_prior(32, k, 0.0, 0.0, 1.0, n_particles=n, generator=_gen(10))
    bonds = bond_vectors(x).reshape(32, -1).numpy()
    ref = multivariate_normal.logpdf(bonds, mean=np.zeros(bonds.shape[1]), cov=sigma ** 2)
    got = log_prior(x, k, 0.0, 0.0, 1.0).numpy()
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
    assert abs(prior_entropy(k, 0.0, 0.0, 1.0, N) - want_entropy) < 1e-9

    re_sqr = 44.2769
    k_equiv = 3.0 * (N - 1) / re_sqr
    assert abs(end_to_end_mean_sq(k_equiv, 0.0, 0.0, 1.0, N) - re_sqr) < 1e-9


# ---- calibration -----------------------------------------------------------


def test_solve_cos_theta_0_round_trips():
    """The solved angle reproduces the requested E[Re^2] through the forward formula.

    The calibration path in miniature: gamma comes from the data's var(cos theta) and
    cos_theta_0 is fitted to the data's size, so a solver that missed would quietly
    hand back a prior of the wrong length.
    """
    k, b, gamma = 91.46, 1.0241, 2.331
    for target in (25.0, 44.443, 60.0):
        c0 = solve_cos_theta_0(k, b, gamma, target, N)
        assert -1.0 <= c0 <= 1.0
        assert abs(end_to_end_mean_sq(k, b, gamma, c0, N) - target) < 1e-6, (target, c0)


def test_solve_cos_theta_0_rejects_unreachable_targets():
    """An unreachable size fails loudly, naming the interval and the remedy.

    Not a numerical failure but a physical one: at this gamma the chain cannot be made
    that stiff, and the fix is a larger gamma, not a cosine outside [-1, 1]. A silent
    clamp to the endpoint would be the worst outcome -- a prior quietly not matching the
    size it was asked for.
    """
    k, b, gamma = 91.46, 1.0241, 2.331
    reachable = (end_to_end_mean_sq(k, b, gamma, -1.0, N),
                 end_to_end_mean_sq(k, b, gamma, 1.0, N))
    with pytest.raises(ValueError, match="unreachable"):
        solve_cos_theta_0(k, b, gamma, reachable[1] * 2.0, N)
    with pytest.raises(ValueError, match="unreachable"):
        solve_cos_theta_0(k, b, gamma, reachable[0] * 0.5, N)


def test_solve_cos_theta_0_is_monotone_in_gamma():
    """A stiffer bending constant needs a smaller equilibrium cosine for the same size.

    Sanity on the direction of the knob: gamma and cos_theta_0 both increase <u>, so at
    a fixed target they trade off against each other. Someone raising gamma to widen the
    reachable range should expect the solved angle to fall.
    """
    k, b, target = 91.46, 1.0241, 44.443
    solved = [solve_cos_theta_0(k, b, g, target, N) for g in (2.0, 5.0, 20.0)]
    assert solved[0] > solved[1] > solved[2], solved


# ---- observables and IO ----------------------------------------------------


def test_gyration_smaller_than_end_to_end():
    """Rg^2 < Re^2 always; and the ideal chain's <Re^2>/<Rg^2> = 6 survives at b = 0.

    The ratio-6 identity is ideal-chain-only -- it comes from the Gaussian chain's
    continuum limit -- so at finite b it is not expected to hold and is not asserted.
    """
    x = sample_prior(BIG, K, B, G, C0, n_particles=N, generator=_gen(11))
    assert gyration_sq(x).mean().item() < end_to_end_sq(x).mean().item()

    x0 = sample_prior(BIG, 1.0, 0.0, 0.0, 1.0, n_particles=N, generator=_gen(12))
    ratio = (end_to_end_sq(x0).mean() / gyration_sq(x0).mean()).item()
    assert abs(ratio - 6.0) / 6.0 < 0.05, ratio


# ---- Rouse modes -----------------------------------------------------------
#
# All exact algebra, no statistics. The transform is a fixed orthogonal-ish matrix, so
# every property it should have holds sample by sample to machine precision, and a
# tolerance-based test would only be hiding a convention error behind sampling noise.
# The conventions worth pinning are the ones a reader is most likely to assume
# differently: which mode is the centre of mass, which modes are translation
# invariant, and that the sqrt(2/N) normalization is NOT the orthonormal DCT-II.

_NR = 7             # deliberately not N: the transform is chain-length agnostic


def _chain(B: int = 40, n: int = _NR, seed: int = 0) -> torch.Tensor:
    return torch.randn(B, n, 3, generator=_gen(seed), dtype=torch.float64)


def test_rouse_matches_the_defining_sum():
    """The matrix form equals the literal double sum it is meant to implement.

    The definition has two off-by-one traps in it -- j runs from 1, so the argument
    carries (j - 1/2) against a 0-based tensor index, and k runs from 0 -- and getting
    either wrong still produces a plausible-looking spectrum.
    """
    x = _chain(B=3, seed=1)
    got = rouse_modes(x)
    want = torch.zeros_like(x)
    for k in range(_NR):
        for j in range(1, _NR + 1):
            want[:, k] += x[:, j - 1] * math.cos(math.pi * k * (j - 0.5) / _NR)
    want *= math.sqrt(2.0 / _NR)
    assert torch.allclose(got, want, atol=1e-14)


def test_rouse_k0_is_the_center_of_mass():
    """X_0 = sqrt(2N) * mean_j R_j -- not a structural mode, and not translation invariant."""
    x = _chain(seed=2)
    assert torch.allclose(rouse_modes(x)[:, 0], math.sqrt(2.0 * _NR) * x.mean(-2),
                          atol=1e-14)


def test_rouse_higher_modes_are_translation_invariant():
    """Shifting the whole chain moves X_0 and leaves every k >= 1 untouched.

    Both halves matter. The invariance is what lets modes be compared between samples
    that are only gauge-fixed up to the pinned tail; the second assert stops the test
    passing on a transform that annihilates everything.
    """
    x = _chain(seed=3)
    shift = torch.randn(1, 1, 3, generator=_gen(4), dtype=torch.float64)
    before, after = rouse_modes(x), rouse_modes(x + shift)
    assert torch.allclose(after[:, 1:], before[:, 1:], atol=1e-13)
    assert (after[:, 0] - before[:, 0]).abs().max() > 1e-3


def test_rouse_parseval_carries_the_extra_com_term():
    """sum_k |X_k|^2 = sum_j |R_j|^2 + N |mean_j R_j|^2.

    The transform is NOT orthonormal: a uniform sqrt(2/N) leaves row 0 with norm
    sqrt(2) rather than 1, so Parseval picks up one extra centre-of-mass term. This is
    the polymer literature's normalization, kept deliberately, and this test is what
    makes the choice explicit rather than accidental -- swapping in the orthonormal
    DCT-II would drop the second term and fail here.
    """
    x = _chain(seed=5)
    got = (rouse_modes(x) ** 2).sum((-1, -2))
    want = (x ** 2).sum((-1, -2)) + _NR * (x.mean(-2) ** 2).sum(-1)
    assert torch.allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("proper", [True, False])
def test_rouse_is_o3_equivariant(proper):
    """X_k -> R X_k, so |X_k|^2 and X_k . X_l are O(3)-INVARIANT.

    Which is the property that makes the moments legitimate to compare between prior,
    generated and reference samples: those are only ever defined up to a rotation, and
    the OT coupling rotates them freely.
    """
    from scipy.spatial.transform import Rotation

    x = _chain(seed=6)
    R = torch.tensor(Rotation.random(random_state=2).as_matrix(), dtype=x.dtype)
    if not proper:
        R = R * torch.tensor([1.0, 1.0, -1.0], dtype=x.dtype)
    assert torch.allclose(rouse_modes(x @ R.T), rouse_modes(x) @ R.T, atol=1e-13)


def test_rouse_modes_are_batch_shape_agnostic():
    """(N, d), (B, N, d) and (A, B, N, d) all work and agree, as the other observables do."""
    x = _chain(B=5, seed=7)
    assert rouse_modes(x[0]).shape == (_NR, 3)
    assert torch.allclose(rouse_modes(x[0]), rouse_modes(x)[0], atol=1e-14)
    stacked = torch.stack([x, x + 1.0])
    assert rouse_modes(stacked).shape == (2, 5, _NR, 3)
    assert torch.allclose(rouse_modes(stacked)[0], rouse_modes(x), atol=1e-14)


def test_rouse_mode_moments_agree_with_the_modes():
    """msq is the diagonal of cov, cov is symmetric, and both are second moments
    about the ORIGIN rather than about the sample mean.

    The last is the one a reader is likely to "fix": these describe a gauge-fixed
    distribution in which <X_k> is genuinely non-zero, so centering would subtract a
    physical quantity. A centered covariance would fail the explicit check below.
    """
    x = _chain(B=64, seed=8) + 0.7          # offset, so a centered version would differ
    msq, cov = rouse_mode_moments(x)
    modes = rouse_modes(x)

    assert msq.shape == (_NR,) and cov.shape == (_NR, _NR)
    assert torch.allclose(msq, torch.diagonal(cov), atol=1e-14)
    assert torch.allclose(cov, cov.T, atol=1e-13)
    assert torch.allclose(msq, (modes ** 2).sum(-1).mean(0), atol=1e-13)
    centered = modes - modes.mean(0, keepdim=True)
    assert not torch.allclose(msq, (centered ** 2).sum(-1).mean(0), atol=1e-3)


def test_rouse_mode_moments_rejects_unbatched_input():
    with pytest.raises(ValueError, match=r"expected a \(B, N, d\) sample"):
        rouse_mode_moments(torch.randn(_NR, 3, dtype=torch.float64))


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


def test_invalid_bending_parameters_are_loud():
    """gamma < 0 and |cos_theta_0| > 1 are rejected by the closed form and the sampler.

    The cosine bound is the load-bearing one and the message says why: outside [-1, 1]
    the truncated normal sits entirely beyond the interval, its mass underflows to zero,
    and every moment formula divides by it. Silently returning inf or nan there would
    poison the entropy without any obvious symptom.
    """
    for gamma, c0 in [(-1.0, 0.0), (1.0, 1.5), (1.0, -1.5)]:
        with pytest.raises(ValueError):
            angle_moments(gamma, c0)
        with pytest.raises(ValueError):
            sample_bond_cosines(4, gamma, c0)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        angle_moments(1.0, 1.5)


def test_non_3d_is_rejected():
    """The prior is 3D-only; the Q^2 Jacobian says so and the error message explains it."""
    with pytest.raises(ValueError, match="n_dims=3 only"):
        sample_prior(4, K, B, G, C0, n_particles=5, n_dims=2)
    with pytest.raises(ValueError, match="n_dims=3 only"):
        prior_entropy(K, B, G, C0, n_particles=5, n_dims=2)
    with pytest.raises(ValueError, match="n_dims=3 only"):
        log_prior(torch.zeros(2, 5, 2, dtype=torch.float64), K, B, G, C0)


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_bond_moments_match_quadrature,
        test_bond_moments_satisfy_the_virial_identity,
        test_sampled_bond_lengths_match_the_moments,
        test_angle_moments_match_quadrature,
        test_angle_moments_at_zero_gamma_are_the_uniform_values,
        test_sampled_bond_cosines_match_the_moments,
        test_sampler_is_reproducible_under_a_fixed_generator,
        test_end_to_end_matches_the_analytic_mean_square,
        test_bond_mean_square_is_the_analytic_moment,
        test_growing_b_grows_the_chain,
        test_bond_directions_are_markov,
        test_perpendicular_basis_is_orthonormal_including_at_the_poles,
        test_bonds_are_isotropic_and_correlated,
        test_positions_are_correlated_not_iid,
        test_tail_is_exactly_zero,
        test_anchor_is_idempotent_and_fixes_the_tail,
        test_dof_and_basis,
        test_log_prior_normalizer_matches_quadrature,
        test_log_prior_is_normalized_via_the_entropy,
        test_log_prior_is_o3_invariant,
        test_log_prior_reads_bonds_not_positions,
        test_bending_deltas_vanish_at_zero_gamma,
        test_bending_energy_matches_prior_samples,
        test_gamma_zero_reproduces_the_freely_jointed_chain,
        test_gamma_zero_is_independent_of_cos_theta_0,
        test_b_zero_reproduces_the_ideal_chain_density,
        test_b_zero_reproduces_the_ideal_chain_constants,
        test_solve_cos_theta_0_round_trips,
        test_solve_cos_theta_0_rejects_unreachable_targets,
        test_solve_cos_theta_0_is_monotone_in_gamma,
        test_gyration_smaller_than_end_to_end,
        test_rouse_matches_the_defining_sum,
        test_rouse_k0_is_the_center_of_mass,
        test_rouse_higher_modes_are_translation_invariant,
        test_rouse_parseval_carries_the_extra_com_term,
        test_rouse_modes_are_batch_shape_agnostic,
        test_rouse_mode_moments_agree_with_the_modes,
        test_rouse_mode_moments_rejects_unbatched_input,
        test_invalid_bond_parameters_are_loud,
        test_invalid_bending_parameters_are_loud,
        test_non_3d_is_rejected,
    ]
    import tempfile

    for _proper in (True, False):
        tests.append(lambda p=_proper: test_rouse_is_o3_equivariant(p))
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
