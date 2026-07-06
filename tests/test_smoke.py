"""Smoke tests for eesi core components: base, ot, data.

Runs as either pytest or a plain script:

    pytest tests/test_smoke.py
    python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from eesi import (
    GaussianMixture,
    ParticleDataset,
    make_loader,
)


L = 5.0
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


def test_dataset_and_loader():
    """`ParticleDataset` + `make_loader` yield correctly shaped batches."""
    torch.manual_seed(SEED)
    species_one = torch.tensor([0, 0, 1, 1, 1])
    positions = torch.rand(20, 5, 3) * L
    species = species_one.unsqueeze(0).repeat(20, 1)
    ds = ParticleDataset(positions, species, L=L)
    assert len(ds) == 20
    assert ds[0][0].shape == (5, 3)
    assert ds[0][1].shape == (5,)
    loader = make_loader(ds, batch_size=4, shuffle=False)
    bx, ba = next(iter(loader))
    assert bx.shape == (4, 5, 3)
    assert ba.shape == (4, 5)


if __name__ == "__main__":
    tests = [
        test_gaussian_mixture_sample_and_log_prob,
        test_gaussian_mixture_high_dim,
        test_gaussian_mixture_reproducible,
        test_dataset_and_loader,
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
