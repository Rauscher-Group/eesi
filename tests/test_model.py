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

from eesi.mlp import TimeMLP
from eesi.model import EESI, _div_exact, _div_hutchinson


# ---- helpers ---------------------------------------------------------------


def _make_mlp(d: int, *, seed: int = 0) -> TimeMLP:
    torch.manual_seed(seed)
    return TimeMLP(d=d, hidden=16, n_layers=2).eval()


def _make_si(
    d: int = 8,
    path: str = "linear",
    gamma: str = "quad",
    score_div_method: str = "hutchinson",
    n_hutchinson_probes: int = 1,
    seed: int = 0,
) -> EESI:
    net_b = _make_mlp(d, seed=seed)
    net_s = _make_mlp(d, seed=seed + 1)
    return EESI(
        net_b, net_s,
        d=d,
        path=path, gamma=gamma,
        score_div_method=score_div_method,
        n_hutchinson_probes=n_hutchinson_probes,
    )


def _random_batch(B: int, d: int, L: float, seed: int = 0):
    """Returns (x1, x0) tensors of shape [B, d]."""
    g = torch.Generator().manual_seed(seed)
    x1 = torch.rand(B, d, generator=g) * L
    x0 = torch.rand(B, d, generator=g) * L
    return x1, x0


# ---- divergence estimators on known fields ---------------------------------


def test_div_exact_known_divergence():
    """_div_exact is exact on a scalar-linear field s = c*x where div(s) = c*d."""
    torch.manual_seed(0)
    B, d, c = 3, 8, 3.0
    x_t = torch.rand(B, d).requires_grad_(True)
    s = c * x_t
    div = _div_exact(s, x_t)
    expected = torch.full((B,), c * d)
    assert torch.allclose(div, expected, atol=1e-4), (
        f"exact div: got {div.tolist()}, expected {expected.tolist()}"
    )


def test_div_hutchinson_unbiased():
    """_div_hutchinson converges to c*d on a linear field with many probes."""
    torch.manual_seed(42)
    B, d, c = 1, 8, 3.0
    x_t = torch.rand(B, d).requires_grad_(True)
    s = c * x_t
    div = _div_hutchinson(s, x_t, n_probes=2000)
    expected = c * d
    assert abs(float(div[0]) - expected) < 1.0, (
        f"hutchinson div: got {float(div[0]):.4f}, expected {expected:.4f}"
    )


# ---- end-to-end score loss via EESI.loss() ---------------------------------


def test_score_loss_hutchinson_finite_and_differentiable():
    """ISM loss['s'] with hutchinson (gamma='none') is finite and backprops into net_s."""
    torch.manual_seed(0)
    model = _make_si(gamma="none", score_div_method="hutchinson")
    x1, x0 = _random_batch(B=2, d=8, L=5.0)

    losses = model.loss(x1, x0)

    assert losses["s"] is not None
    assert losses["s"].shape == torch.Size([]), "loss_s must be a scalar"
    assert losses["s"].isfinite(), f"loss_s is not finite: {losses['s'].item()}"

    losses["s"].backward()
    assert any(p.grad is not None for p in model.net_s.parameters()), (
        "no net_s parameter received a gradient"
    )


def test_score_loss_exact_finite_and_differentiable():
    """ISM loss['s'] with exact trace (gamma='none') is finite and backprops into net_s."""
    torch.manual_seed(0)
    model = _make_si(gamma="none", score_div_method="exact")
    x1, x0 = _random_batch(B=2, d=8, L=5.0)

    losses = model.loss(x1, x0)

    assert losses["s"] is not None
    assert losses["s"].shape == torch.Size([]), "loss_s must be a scalar"
    assert losses["s"].isfinite(), f"loss_s is not finite: {losses['s'].item()}"

    losses["s"].backward()
    assert any(p.grad is not None for p in model.net_s.parameters()), (
        "no net_s parameter received a gradient"
    )


@pytest.mark.parametrize("gamma", ["none", "quad"])
def test_score_loss_net_b_unaffected(gamma):
    """Backpropping loss_s alone must not deposit gradients into net_b.

    Holds for both the ISM path (gamma='none', via x_t detach) and the antithetic
    denoising path (gamma!='none', because loss_s only calls net_s).
    """
    torch.manual_seed(0)
    model = _make_si(gamma=gamma, score_div_method="hutchinson")
    x1, x0 = _random_batch(B=2, d=8, L=5.0)

    losses = model.loss(x1, x0)
    losses["s"].backward()

    assert all(p.grad is None for p in model.net_b.parameters()), (
        f"net_b received gradients from loss_s (gamma={gamma})"
    )


def test_score_loss_invalid_method():
    """EESI raises ValueError for an unrecognised score_div_method."""
    with pytest.raises(ValueError, match="score_div_method"):
        _make_si(score_div_method="bad_method")


# ---- interpolant path / gamma selection ------------------------------------


@pytest.mark.parametrize("path", ["linear", "trig", "encdec"])
@pytest.mark.parametrize("gamma", ["none", "quad", "sqrt"])
def test_loss_finite_across_paths_and_gammas(path, gamma):
    """Every (path, gamma) combo yields finite b/s losses that backprop into net_b/net_s."""
    torch.manual_seed(0)
    model = _make_si(path=path, gamma=gamma)
    x1, x0 = _random_batch(B=2, d=8, L=5.0)

    losses = model.loss(x1, x0)
    assert losses["b"].isfinite(), f"loss_b not finite for {path}/{gamma}"
    assert losses["s"].isfinite(), f"loss_s not finite for {path}/{gamma}"

    (losses["b"] + losses["s"]).backward()
    assert any(p.grad is not None for p in model.net_b.parameters()), (
        f"no net_b gradient for {path}/{gamma}"
    )
    assert any(p.grad is not None for p in model.net_s.parameters()), (
        f"no net_s gradient for {path}/{gamma}"
    )


def test_invalid_path_and_gamma():
    """EESI raises ValueError for unrecognised path / gamma names."""
    with pytest.raises(ValueError, match="path"):
        _make_si(path="bad_path")
    with pytest.raises(ValueError, match="gamma"):
        _make_si(gamma="bad_gamma")


# ---- antithetic denoising losses (non-zero gamma) --------------------------


@pytest.mark.parametrize("path", ["linear", "trig", "encdec"])
@pytest.mark.parametrize("gamma", ["quad", "sqrt"])
def test_antithetic_losses_finite_and_differentiable(path, gamma):
    """Antithetic DSM path: loss_b/loss_s are finite and backprop into net_b/net_s."""
    torch.manual_seed(0)
    model = _make_si(path=path, gamma=gamma)
    x1, x0 = _random_batch(B=2, d=8, L=5.0)

    losses = model.loss(x1, x0)
    assert losses["b"].shape == torch.Size([]) and losses["s"].shape == torch.Size([])
    assert losses["b"].isfinite() and losses["s"].isfinite(), (
        f"non-finite loss for {path}/{gamma}: {losses}"
    )

    losses["b"].backward(retain_graph=True)
    assert any(p.grad is not None for p in model.net_b.parameters()), "no net_b gradient"
    losses["s"].backward()
    assert any(p.grad is not None for p in model.net_s.parameters()), "no net_s gradient"


def test_antithetic_finite_near_endpoints():
    """With gamma='sqrt' (gamma'->inf, 1/gamma->inf at the ends) losses stay finite.

    Antithetic sampling must cancel the endpoint singularities across many draws
    of t in [eps, 1-eps], including t very close to the boundary.
    """
    for seed in range(25):
        model = _make_si(gamma="sqrt", seed=seed)
        x1, x0 = _random_batch(B=4, d=8, L=5.0, seed=seed)
        torch.manual_seed(seed)  # drives the internal t / z draws
        losses = model.loss(x1, x0)
        assert losses["b"].isfinite(), f"loss_b not finite at seed {seed}: {losses['b']}"
        assert losses["s"].isfinite(), f"loss_s not finite at seed {seed}: {losses['s']}"


# ---- runner ----------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_div_exact_known_divergence,
        test_div_hutchinson_unbiased,
        test_score_loss_hutchinson_finite_and_differentiable,
        test_score_loss_exact_finite_and_differentiable,
        test_score_loss_invalid_method,
        test_invalid_path_and_gamma,
        test_antithetic_finite_near_endpoints,
    ]
    for _g in ("none", "quad"):
        tests.append(lambda g=_g: test_score_loss_net_b_unaffected(g))
    for _path in ("linear", "trig", "encdec"):
        for _gamma in ("none", "quad", "sqrt"):
            tests.append(
                lambda p=_path, g=_gamma: test_loss_finite_across_paths_and_gammas(p, g)
            )
        for _gamma in ("quad", "sqrt"):
            tests.append(
                lambda p=_path, g=_gamma: test_antithetic_losses_finite_and_differentiable(p, g)
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
