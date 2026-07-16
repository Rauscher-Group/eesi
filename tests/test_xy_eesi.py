"""Tests for `eesi.interpolant.xyEESI` and the `_interpolant_sample` refactor.

Runs as either pytest or a plain script:

    pytest tests/test_xy_eesi.py
    python tests/test_xy_eesi.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from eesi.interpolant import EESI, _min_image, xyEESI
from eesi.models.xygnn import XYChainGNN


# ---- helpers ---------------------------------------------------------------


def _nets(seed: int = 0):
    """Two independent nets. `XYChainGNN` is chain-length independent: it infers N
    from each input, so the chain length is not a constructor argument."""
    torch.manual_seed(seed)
    kw = dict(n_neighbors=2, hidden=16, n_layers=2, edge_order=2, time_order=2)
    return XYChainGNN(**kw), XYChainGNN(**kw)


def _make_xy(N: int, path: str = "linear", gamma: str = "quad", seed: int = 0) -> xyEESI:
    net_b, net_s = _nets(seed)
    return xyEESI(net_b, net_s, d=N, path=path, gamma=gamma)


def _random_angles(B: int, N: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, N, generator=g) * 2.0 * math.pi) - math.pi


# ---- _interpolant_sample geometry -----------------------------------------


def test_sample_geodesic_position_and_targets_no_noise():
    """With z=0: x_t = wrap(x0+beta*d), b_target = beta'*d, s_target = 0 (linear)."""
    model = _make_xy(4, path="linear", gamma="quad")
    # components 1 and 4 straddle the 0/2pi seam
    x0 = torch.tensor([[0.1, 1.0, 0.2, -3.0]])
    x1 = torch.tensor([[6.0, 1.2, 6.1, 3.0]])
    t = torch.full((1, 1), 0.3)
    z = torch.zeros(1, 4)

    x_t, b_target, s_target = model._interpolant_sample(t, x0, x1, z)
    d = _min_image(x1 - x0)
    assert d.abs().max() <= math.pi + 1e-6
    assert torch.allclose(x_t, _min_image(x0 + 0.3 * d), atol=1e-6)
    assert torch.allclose(b_target, d, atol=1e-6)            # beta_dot=1 for linear
    assert torch.allclose(s_target, torch.zeros_like(s_target), atol=1e-6)
    assert (x_t <= math.pi + 1e-6).all() and (x_t > -math.pi - 1e-6).all()
    # the straight-line drift target would be huge; the geodesic one is small
    assert b_target.abs().max() < (x1 - x0).abs().max()


def test_sample_targets_with_noise():
    """x_t, b_target, s_target match the closed forms using the model's own schedules."""
    model = _make_xy(4, path="linear", gamma="quad", seed=1)
    x0 = _random_angles(3, 4, seed=0)
    x1 = _random_angles(3, 4, seed=1)
    t = torch.rand(3, 1) * 0.8 + 0.1
    z = torch.randn(3, 4)

    x_t, b_target, s_target = model._interpolant_sample(t, x0, x1, z)
    _, beta, _, beta_dot = model._path(t)
    g, g_dot = model._scaled_gamma(t)
    d = _min_image(x1 - x0)
    assert torch.allclose(x_t, _min_image(x0 + beta * d + g * z), atol=1e-6)
    assert torch.allclose(b_target, beta_dot * d + g_dot * z, atol=1e-6)
    assert torch.allclose(s_target, -z / g.clamp_min(1e-12), atol=1e-6)
    assert (x_t <= math.pi + 1e-6).all() and (x_t > -math.pi - 1e-6).all()


def test_antithetic_sign_relations():
    """The +z / -z branches average to the deterministic drift; score flips sign."""
    model = _make_xy(4, path="linear", gamma="quad", seed=2)
    x0 = _random_angles(3, 4, seed=2)
    x1 = _random_angles(3, 4, seed=3)
    t = torch.rand(3, 1) * 0.8 + 0.1
    z = torch.randn(3, 4)

    _, b_p, s_p = model._interpolant_sample(t, x0, x1, z)
    _, b_m, s_m = model._interpolant_sample(t, x0, x1, -z)
    _, beta, _, beta_dot = model._path(t)
    d = _min_image(x1 - x0)
    assert torch.allclose(0.5 * (b_p + b_m), beta_dot * d, atol=1e-6)
    assert torch.allclose(s_p, -s_m, atol=1e-6)


def test_endpoints_recovered_mod_2pi():
    """At t=0 the position is x0; at t=1 it equals x1 modulo 2pi."""
    model = _make_xy(4, path="linear", gamma="quad")
    x0 = _random_angles(3, 4, seed=0)
    x1 = _random_angles(3, 4, seed=1)
    z = torch.zeros(3, 4)
    x0_t, _, _ = model._interpolant_sample(torch.zeros(3, 1), x0, x1, z)
    x1_t, _, _ = model._interpolant_sample(torch.ones(3, 1), x0, x1, z)
    assert torch.allclose(_min_image(x0_t - x0), torch.zeros_like(x0), atol=1e-5)
    assert torch.allclose(_min_image(x1_t - x1), torch.zeros_like(x1), atol=1e-5)


def test_drift_target_invariant_to_global_rotation():
    """b_target depends only on x1-x0 (and z), so a global shift leaves it unchanged."""
    model = _make_xy(4, path="linear", gamma="quad")
    x0 = _random_angles(3, 4, seed=2)
    x1 = _random_angles(3, 4, seed=3)
    t = torch.rand(3, 1) * 0.8 + 0.1
    z = torch.randn(3, 4)
    _, b0, _ = model._interpolant_sample(t, x0, x1, z)
    _, b1, _ = model._interpolant_sample(t, x0 + 0.7, x1 + 0.7, z)
    assert torch.allclose(b0, b1, atol=1e-5)


def test_reduces_to_eesi_without_wrapping():
    """With all differences inside (-pi, pi], the geodesic equals the straight line."""
    N = 4
    xy = _make_xy(N, path="linear", gamma="quad")
    euc = EESI(xy.net_b, xy.net_s, d=N, path="linear", gamma="quad")

    x0 = _random_angles(2, N, seed=4) * 0.3
    x1 = x0 + torch.tensor([[0.5, -0.5, 1.0, -1.0], [0.2, 0.3, -0.4, 0.1]])  # |delta| < pi
    t = torch.full((2, 1), 0.4)
    z = torch.randn(2, N)

    xt_a, b_a, s_a = xy._interpolant_sample(t, x0, x1, z)
    xt_e, b_e, s_e = euc._interpolant_sample(t, x0, x1, z)
    assert torch.allclose(b_a, b_e, atol=1e-6)
    assert torch.allclose(s_a, s_e, atol=1e-6)
    # positions agree modulo the (here inactive) wrap
    assert torch.allclose(_min_image(xt_a - xt_e), torch.zeros_like(xt_a), atol=1e-6)


# ---- EESI base sample + numerical parity of the refactor -------------------


def test_eesi_sample_closed_form():
    """The base (Euclidean) `_interpolant_sample` matches its closed forms."""
    N = 4
    net_b, net_s = _nets(seed=5)
    model = EESI(net_b, net_s, d=N, path="trig", gamma="quad")
    x0 = _random_angles(3, N, seed=0)
    x1 = _random_angles(3, N, seed=1)
    t = torch.rand(3, 1) * 0.8 + 0.1
    z = torch.randn(3, N)

    x_t, b_target, s_target = model._interpolant_sample(t, x0, x1, z)
    alpha, beta, alpha_dot, beta_dot = model._path(t)
    g, g_dot = model._scaled_gamma(t)
    assert torch.allclose(x_t, alpha * x0 + beta * x1 + g * z, atol=1e-6)
    assert torch.allclose(b_target, alpha_dot * x0 + beta_dot * x1 + g_dot * z, atol=1e-6)
    assert torch.allclose(s_target, -z / g.clamp_min(1e-12), atol=1e-6)


def test_loss_matches_antithetic_formula():
    """The refactored antithetic loss reproduces the original quad/lin expressions."""
    N, B = 5, 4
    net_b, net_s = _nets(seed=6)
    model = EESI(net_b, net_s, d=N, path="linear", gamma="quad")
    x1 = _random_angles(B, N, seed=20)
    x0 = _random_angles(B, N, seed=21)

    torch.manual_seed(99)
    out = model.loss(x1, x0)

    # Replay the same RNG stream (t, then z) and apply the pre-refactor formula.
    torch.manual_seed(99)
    t = torch.rand((B, 1)) * (1.0 - 2.0 * model.eps) + model.eps
    t_b = t.view(B)
    alpha, beta, alpha_dot, beta_dot = model._path(t)
    g, g_dot = model._scaled_gamma(t)
    I_t = alpha * x0 + beta * x1
    v_det = alpha_dot * x0 + beta_dot * x1
    z = torch.randn_like(x1)
    x_plus, x_minus = I_t + g * z, I_t - g * z
    b_plus, b_minus = model.net_b(t_b, x_plus), model.net_b(t_b, x_minus)
    quad_b = 0.25 * (b_plus.square() + b_minus.square()).sum(-1)
    lin_det = 0.5 * (v_det * (b_plus + b_minus)).sum(-1)
    lin_noise = 0.5 * (g_dot * z * (b_plus - b_minus)).sum(-1)
    loss_b = (quad_b - lin_det - lin_noise).mean()
    s_plus, s_minus = model.net_s(t_b, x_plus), model.net_s(t_b, x_minus)
    quad_s = 0.25 * (s_plus.square() + s_minus.square()).sum(-1)
    cross_s = ((s_plus - s_minus) * z).sum(-1) / (2.0 * g.view(B).clamp_min(1e-12))
    loss_s = (quad_s + cross_s).mean()

    assert torch.allclose(out["b"], loss_b, atol=1e-5)
    assert torch.allclose(out["s"], loss_s, atol=1e-5)


# ---- xyEESI integration ----------------------------------------------------


def test_xy_loss_runs_and_backprops():
    """Two XYChainGNNs give a finite xyEESI loss (gamma='quad') that backprops."""
    N, B = 6, 5
    model = _make_xy(N, path="linear", gamma="quad", seed=1)
    x1 = _random_angles(B, N, seed=10)
    x0 = _random_angles(B, N, seed=11)
    losses = model.loss(x1, x0)
    total = losses["b"] + losses["s"]
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_xy_ism_path_runs():
    """gamma='none' exercises the implicit-score-matching divergence path."""
    N, B = 6, 4
    net_b, net_s = _nets(seed=2)
    model = xyEESI(net_b, net_s, d=N, path="linear", gamma="none", n_hutchinson_probes=4)
    x1 = _random_angles(B, N, seed=12)
    x0 = _random_angles(B, N, seed=13)
    losses = model.loss(x1, x0)
    total = losses["b"] + losses["s"]
    assert torch.isfinite(total)
    total.backward()


def test_xy_loss_invariant_to_2pi_shifts():
    """The loss is unchanged under per-node 2pi shifts of the inputs (reseeded RNG)."""
    N, B = 6, 4
    model = _make_xy(N, path="linear", gamma="quad", seed=3)
    x1 = _random_angles(B, N, seed=20)
    x0 = _random_angles(B, N, seed=21)

    g = torch.Generator().manual_seed(7)
    m0 = torch.randint(-2, 3, (B, N), generator=g).to(x0.dtype) * (2.0 * math.pi)
    m1 = torch.randint(-2, 3, (B, N), generator=g).to(x1.dtype) * (2.0 * math.pi)

    torch.manual_seed(123)
    l0 = model.loss(x1, x0)
    s0 = l0["b"] + l0["s"]
    torch.manual_seed(123)
    l1 = model.loss(x1 + m1, x0 + m0)
    s1 = l1["b"] + l1["s"]
    assert torch.allclose(s0, s1, atol=1e-4), f"|Δ| = {(s0 - s1).abs().item():.3e}"


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_sample_geodesic_position_and_targets_no_noise,
        test_sample_targets_with_noise,
        test_antithetic_sign_relations,
        test_endpoints_recovered_mod_2pi,
        test_drift_target_invariant_to_global_rotation,
        test_reduces_to_eesi_without_wrapping,
        test_eesi_sample_closed_form,
        test_loss_matches_antithetic_formula,
        test_xy_loss_runs_and_backprops,
        test_xy_ism_path_runs,
        test_xy_loss_invariant_to_2pi_shifts,
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
