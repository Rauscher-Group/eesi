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


def _mixture(dim, n_mixes, loc_scaling=4.0, **kw):
    """Build a CPU mixture; `device` defaults to "cuda" in the constructor."""
    return GaussianMixture(
        dim=dim,
        n_mixes=n_mixes,
        loc_scaling=loc_scaling,
        seed=SEED,
        device="cpu",
        **kw,
    )


def test_gaussian_mixture_sample_and_log_prob():
    """`GaussianMixture` (dim=2, illustrative size) produces correctly shaped,
    finite-log-prob samples."""
    n_dim = 2
    base = _mixture(dim=n_dim, n_mixes=3)
    batch_size = 100
    x = base.sample((batch_size,))
    assert x.shape == (batch_size, n_dim)
    lp = base.log_prob(x)
    assert lp.shape == (batch_size,)
    assert torch.isfinite(lp).all().item()
    # Frozen parameters: one unnormalized weight per component, normalized by
    # the Categorical into a valid simplex point.
    assert base.cat_probs.shape == (3,)
    probs = base.distribution.mixture_distribution.probs
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_gaussian_mixture_high_dim():
    """`GaussianMixture` also holds up at dim=128, the "challenging" regime."""
    n_dim = 128
    base = _mixture(dim=n_dim, n_mixes=8)
    batch_size = 16
    x = base.sample((batch_size,))
    assert x.shape == (batch_size, n_dim)
    lp = base.log_prob(x)
    assert lp.shape == (batch_size,)
    assert torch.isfinite(lp).all().item()
    # Covariances C_k = L L^T are symmetric positive-definite.
    assert base.scale_trils.shape == (8, n_dim, n_dim)
    cov = base.scale_trils @ base.scale_trils.transpose(-1, -2)
    eigs = torch.linalg.eigvalsh(cov)
    assert (eigs > 0).all().item()


def test_gaussian_mixture_reproducible():
    """A fixed `seed` reproduces the same frozen parameters."""
    a = _mixture(dim=5, n_mixes=4, loc_scaling=2.0)
    b = _mixture(dim=5, n_mixes=4, loc_scaling=2.0)
    assert torch.allclose(a.cat_probs, b.cat_probs)
    assert torch.allclose(a.locs, b.locs)
    assert torch.allclose(a.scale_trils, b.scale_trils)


def test_gaussian_mixture_score():
    """`score` returns a finite gradient of log p with the shape of x."""
    n_dim = 4
    base = _mixture(dim=n_dim, n_mixes=3)
    x = base.sample((8,))
    s = base.score(x)
    assert s.shape == (8, n_dim)
    assert torch.isfinite(s).all().item()


def test_gaussian_mixture_joint_helpers():
    """`get_sample_and_logp` / `get_sample_and_score` agree with the pieces."""
    n_dim = 4
    base = _mixture(dim=n_dim, n_mixes=3)
    x, lp = base.get_sample_and_logp((8,))
    assert x.shape == (8, n_dim) and lp.shape == (8,)
    assert torch.allclose(lp, base.log_prob(x))

    y, s = base.get_sample_and_score((8,))
    assert y.shape == (8, n_dim) and s.shape == (8, n_dim)
    assert torch.isfinite(s).all().item()


def test_gaussian_mixture_call_counter():
    """`call_time` accumulates the evaluated batch sizes, unless opted out."""
    base = _mixture(dim=3, n_mixes=2)
    assert base.call_time == 0
    x = base.sample((10,))
    assert base.call_time == 10
    base.log_prob(x)
    assert base.call_time == 20
    base.log_prob(x, count_call=False)
    assert base.call_time == 20


if __name__ == "__main__":
    tests = [
        test_gaussian_mixture_sample_and_log_prob,
        test_gaussian_mixture_high_dim,
        test_gaussian_mixture_reproducible,
        test_gaussian_mixture_score,
        test_gaussian_mixture_joint_helpers,
        test_gaussian_mixture_call_counter,
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
