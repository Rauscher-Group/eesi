"""Tests for `eesi.datasets.xy`: the exact sampler against theory and against MC.

`sample_p1_exact` claims that the open XY chain factorises into a uniform first
angle and independent von Mises bonds. That claim is what makes it a drop-in
replacement for the Metropolis sampler, so it is checked two ways here: against
the closed-form bond density, and against `mcxy` itself.

Runs as either pytest or a plain script:

    pytest tests/test_xy_dataset.py
    python tests/test_xy_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from eesi.datasets.xy import energy, mcxy, sample_p1_exact


# ---- helpers ---------------------------------------------------------------


def _bonds(theta: np.ndarray) -> np.ndarray:
    """Wrapped nearest-neighbour differences of a [B, N] configuration -> [B, N-1]."""
    d = np.diff(theta, axis=1)
    return (d + np.pi) % (2 * np.pi) - np.pi


def _bessel_ratio(J: float) -> float:
    """I1(J) / I0(J) by quadrature: I_n(J) = (1/pi) int_0^pi e^{J cos u} cos(nu) du."""
    u = np.linspace(0.0, np.pi, 20_001)
    w = np.exp(J * np.cos(u))
    return np.trapezoid(w * np.cos(u), u) / np.trapezoid(w, u)


# ---- the exact sampler -----------------------------------------------------


def test_sample_p1_exact_shape_and_range():
    """Returns [B, N] angles wrapped onto (-pi, pi], for N down to 1."""
    x = sample_p1_exact(64, 7, 2.0, np.random.default_rng(0))
    assert x.shape == (64, 7)
    assert (x > -np.pi - 1e-12).all() and (x <= np.pi + 1e-12).all()
    # N = 1 is the degenerate no-bond case: a single uniform angle
    assert sample_p1_exact(16, 1, 2.0, np.random.default_rng(0)).shape == (16, 1)


def test_sample_p1_exact_is_reproducible():
    """The generator is the only source of randomness."""
    a = sample_p1_exact(32, 6, 1.5, np.random.default_rng(7))
    b = sample_p1_exact(32, 6, 1.5, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_bond_density_matches_von_mises():
    """The bond histogram matches vonMises(0, J) = exp(J cos d) / Z."""
    J, B, N = 2.0, 20_000, 8
    d = _bonds(sample_p1_exact(B, N, J, np.random.default_rng(1))).ravel()

    edges = np.linspace(-np.pi, np.pi, 25)
    got, _ = np.histogram(d, bins=edges, density=True)

    grid = np.linspace(-np.pi, np.pi, 4001)          # normalise the density on-grid
    dens = np.exp(J * np.cos(grid))
    dens /= np.trapezoid(dens, grid)
    want = np.array([                                 # bin-average the exact density
        np.trapezoid(np.interp(g := np.linspace(lo, hi, 201), grid, dens), g) / (hi - lo)
        for lo, hi in zip(edges[:-1], edges[1:])
    ])
    assert np.abs(got - want).max() < 0.02, f"max bin error {np.abs(got - want).max():.4f}"


def test_first_angle_marginal_is_uniform():
    """theta_1 ~ U(-pi, pi]: mean resultant length ~ 0 (no preferred direction)."""
    x = sample_p1_exact(20_000, 6, 2.0, np.random.default_rng(2))
    r = np.hypot(np.cos(x[:, 0]).mean(), np.sin(x[:, 0]).mean())
    assert r < 0.03, f"|<e^{{i theta_1}}>| = {r:.4f}"


def test_bonds_are_independent():
    """Neighbouring bonds are uncorrelated -- the factorisation's whole content."""
    d = _bonds(sample_p1_exact(20_000, 10, 2.0, np.random.default_rng(3)))
    c = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
    assert abs(c) < 0.03, f"corr(Delta_i, Delta_i+1) = {c:.4f}"


def test_mean_energy_matches_closed_form():
    """<U> = -(N-1) J I1(J)/I0(J), the number the notebook plots against."""
    J, B, N = 2.0, 20_000, 10
    x = sample_p1_exact(B, N, J, np.random.default_rng(4))
    got = np.array([energy(row, J) for row in x]).mean()
    want = -(N - 1) * J * _bessel_ratio(J)
    assert abs(got - want) < 0.05, f"<U> = {got:.4f} vs exact {want:.4f}"


# ---- agreement with the Monte Carlo sampler --------------------------------


def test_agrees_with_mcxy():
    """The exact sampler and `mcxy` produce the same bond distribution.

    This is the remaining use for `mcxy`: an independent check of the sampler the
    training data now comes from. Kept short -- Metropolis here is a Python loop.
    """
    J, N = 2.0, 8
    np.random.seed(0)
    mc, _ = mcxy(N=N, J=J, n_eq=20_000, n_prod=200_000, n_save=2_000)
    ex = sample_p1_exact(20_000, N, J, np.random.default_rng(5))

    d_mc, d_ex = _bonds(mc), _bonds(ex)
    # first two circular moments of the bond distribution
    for k in (1, 2):
        m_mc, m_ex = np.cos(k * d_mc).mean(), np.cos(k * d_ex).mean()
        assert abs(m_mc - m_ex) < 0.02, f"<cos {k}d>: mc {m_mc:.4f} vs exact {m_ex:.4f}"
        s_mc, s_ex = np.sin(k * d_mc).mean(), np.sin(k * d_ex).mean()
        assert abs(s_mc - s_ex) < 0.02, f"<sin {k}d>: mc {s_mc:.4f} vs exact {s_ex:.4f}"


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_sample_p1_exact_shape_and_range,
        test_sample_p1_exact_is_reproducible,
        test_bond_density_matches_von_mises,
        test_first_angle_marginal_is_uniform,
        test_bonds_are_independent,
        test_mean_energy_matches_closed_form,
        test_agrees_with_mcxy,
    ]
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
