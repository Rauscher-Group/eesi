"""Tests for `eesi.model`: divergence estimators and score loss.

Runs as either pytest or a plain script:

    pytest tests/test_model.py
    python tests/test_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from eesi.egnn import EGNN
from eesi.model import SI, _div_exact, _div_hutchinson


# ---- helpers ---------------------------------------------------------------


def _make_egnn(d: int, r_cut: float, *, seed: int = 0) -> EGNN:
    torch.manual_seed(seed)
    return EGNN(
        d=d, r_cut=r_cut,
        hidden=16, n_layers=2, k_attn=1, n_global_tokens=2,
    ).eval()


def _make_si(
    d: int = 2,
    r_cut: float = 2.0,
    n_particles: int = 8,
    path: str = "linear",
    gamma: str = "quad",
    score_div_method: str = "hutchinson",
    n_hutchinson_probes: int = 1,
    seed: int = 0,
) -> SI:
    net_b = _make_egnn(d, r_cut, seed=seed)
    net_s = _make_egnn(d, r_cut, seed=seed + 1)
    return SI(
        net_b, net_s,
        n_particles=n_particles, d=d,
        path=path, gamma=gamma,
        score_div_method=score_div_method,
        n_hutchinson_probes=n_hutchinson_probes,
    )


def _random_batch(B: int, N: int, d: int, L: float, seed: int = 0):
    """Returns (x1, x0) tensors."""
    g = torch.Generator().manual_seed(seed)
    x1 = torch.rand(B, N, d, generator=g) * L
    x0 = torch.rand(B, N, d, generator=g) * L
    return x1, x0


# ---- divergence estimators on known fields ---------------------------------


def test_div_exact_known_divergence():
    """_div_exact is exact on a scalar-linear field s = c*x where div(s) = c*N*d."""
    torch.manual_seed(0)
    B, N, d, c = 3, 4, 2, 3.0
    x_t = torch.rand(B, N, d).requires_grad_(True)
    s = c * x_t
    div = _div_exact(s, x_t)
    expected = torch.full((B,), c * N * d)
    assert torch.allclose(div, expected, atol=1e-4), (
        f"exact div: got {div.tolist()}, expected {expected.tolist()}"
    )


def test_div_hutchinson_unbiased():
    """_div_hutchinson converges to c*N*d on a linear field with many probes."""
    torch.manual_seed(42)
    B, N, d, c = 1, 4, 2, 3.0
    x_t = torch.rand(B, N, d).requires_grad_(True)
    s = c * x_t
    div = _div_hutchinson(s, x_t, n_probes=2000)
    expected = c * N * d
    assert abs(float(div[0]) - expected) < 1.0, (
        f"hutchinson div: got {float(div[0]):.4f}, expected {expected:.4f}"
    )


# ---- end-to-end score loss via SI.loss() -----------------------------------


def test_score_loss_hutchinson_finite_and_differentiable():
    """loss['s'] with hutchinson is a finite scalar and backpropagates into net_s."""
    torch.manual_seed(0)
    model = _make_si(score_div_method="hutchinson")
    x1, x0 = _random_batch(B=2, N=8, d=2, L=5.0)

    losses = model.loss(x1, x0)

    assert losses["s"] is not None
    assert losses["s"].shape == torch.Size([]), "loss_s must be a scalar"
    assert losses["s"].isfinite(), f"loss_s is not finite: {losses['s'].item()}"

    losses["s"].backward()
    assert any(p.grad is not None for p in model.net_s.parameters()), (
        "no net_s parameter received a gradient"
    )


def test_score_loss_exact_finite_and_differentiable():
    """loss['s'] with exact autograd trace is a finite scalar and backpropagates into net_s."""
    torch.manual_seed(0)
    model = _make_si(score_div_method="exact")
    x1, x0 = _random_batch(B=2, N=8, d=2, L=5.0)

    losses = model.loss(x1, x0)

    assert losses["s"] is not None
    assert losses["s"].shape == torch.Size([]), "loss_s must be a scalar"
    assert losses["s"].isfinite(), f"loss_s is not finite: {losses['s'].item()}"

    losses["s"].backward()
    assert any(p.grad is not None for p in model.net_s.parameters()), (
        "no net_s parameter received a gradient"
    )


def test_score_loss_net_b_unaffected():
    """Backpropping loss_s alone must not deposit gradients into net_b (x_t detach)."""
    torch.manual_seed(0)
    model = _make_si(score_div_method="hutchinson")
    x1, x0 = _random_batch(B=2, N=8, d=2, L=5.0)

    losses = model.loss(x1, x0)
    losses["s"].backward()

    assert all(p.grad is None for p in model.net_b.parameters()), (
        "net_b received gradients from loss_s — x_t detach is broken"
    )


def test_score_loss_invalid_method():
    """SI raises ValueError for an unrecognised score_div_method."""
    with pytest.raises(ValueError, match="score_div_method"):
        _make_si(score_div_method="bad_method")


# ---- interpolant path / gamma selection ------------------------------------


@pytest.mark.parametrize("path", ["linear", "trig", "encdec"])
@pytest.mark.parametrize("gamma", ["none", "quad", "sqrt"])
def test_loss_finite_across_paths_and_gammas(path, gamma):
    """Every (path, gamma) combo yields finite b/s losses that backprop into net_b."""
    torch.manual_seed(0)
    model = _make_si(path=path, gamma=gamma)
    x1, x0 = _random_batch(B=2, N=8, d=2, L=5.0)

    losses = model.loss(x1, x0)
    assert losses["b"].isfinite(), f"loss_b not finite for {path}/{gamma}"
    assert losses["s"].isfinite(), f"loss_s not finite for {path}/{gamma}"

    losses["b"].backward()
    assert any(p.grad is not None for p in model.net_b.parameters()), (
        f"no net_b gradient for {path}/{gamma}"
    )


def test_invalid_path_and_gamma():
    """SI raises ValueError for unrecognised path / gamma names."""
    with pytest.raises(ValueError, match="path"):
        _make_si(path="bad_path")
    with pytest.raises(ValueError, match="gamma"):
        _make_si(gamma="bad_gamma")


# ---- runner ----------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_div_exact_known_divergence,
        test_div_hutchinson_unbiased,
        test_score_loss_hutchinson_finite_and_differentiable,
        test_score_loss_exact_finite_and_differentiable,
        test_score_loss_net_b_unaffected,
        test_score_loss_invalid_method,
        test_invalid_path_and_gamma,
    ]
    for _path in ("linear", "trig", "encdec"):
        for _gamma in ("none", "quad", "sqrt"):
            tests.append(
                lambda p=_path, g=_gamma: test_loss_finite_across_paths_and_gammas(p, g)
            )
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
