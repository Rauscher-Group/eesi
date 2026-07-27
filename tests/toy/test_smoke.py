"""Smoke tests for the toy Gaussian-mixture base distribution.

Runs as either pytest or a plain script:

    pytest tests/toy/test_smoke.py
    python tests/toy/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from eesi import GaussianMixture


SEED = 0


def test_gaussian_mixture_sample_and_log_prob():
    """`GaussianMixture` (d=2, illustrative size) produces correctly shaped,
    finite-log-prob samples."""
    n_dim = 2
    base = GaussianMixture(d=n_dim, n=3, sigma=4.0, seed=SEED)
    batch_size = 100
    x = base.sample(batch_size)
    assert x.shape == (batch_size, n_dim)
    lp = base.log_prob(x)
    assert lp.shape == (batch_size,)
    assert torch.isfinite(lp).all().item()
    # Frozen parameters: weights are a valid simplex point.
    assert base.weights.shape == (3,)
    assert torch.allclose(base.weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_gaussian_mixture_high_dim():
    """`GaussianMixture` also holds up at d=128, the "challenging" regime."""
    n_dim = 128
    base = GaussianMixture(d=n_dim, n=8, sigma=4.0, seed=SEED)
    batch_size = 16
    x = base.sample((batch_size,))
    assert x.shape == (batch_size, n_dim)
    lp = base.log_prob(x)
    assert lp.shape == (batch_size,)
    assert torch.isfinite(lp).all().item()
    # Covariances C_k = (1/d) W^T W + I are symmetric positive-definite.
    cov = base.covariances
    assert cov.shape == (8, n_dim, n_dim)
    eigs = torch.linalg.eigvalsh(cov)
    assert (eigs > 0).all().item()


def test_gaussian_mixture_reproducible():
    """A fixed `seed` reproduces the same frozen parameters and samples."""
    a = GaussianMixture(d=5, n=4, sigma=2.0, seed=SEED)
    b = GaussianMixture(d=5, n=4, sigma=2.0, seed=SEED)
    assert torch.allclose(a.weights, b.weights)
    assert torch.allclose(a.means, b.means)
    assert torch.allclose(a.covariances, b.covariances)


if __name__ == "__main__":
    tests = [
        test_gaussian_mixture_sample_and_log_prob,
        test_gaussian_mixture_high_dim,
        test_gaussian_mixture_reproducible,
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
