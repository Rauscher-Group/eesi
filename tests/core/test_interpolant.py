"""Tests for `eesi.interpolant`: divergence estimators and score loss.

Runs as either pytest or a plain script:

    pytest tests/test_model.py
    python tests/test_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from eesi.systems.gmm.mlp import TimeMLP
from eesi.interpolant import EESI, _div_exact, _div_hutchinson


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


# ---- entropy_estimate(method="zdot") ---------------------------------------


class _LinearField(torch.nn.Module):
    """b(t, x) = x @ A^T, a field with the exactly known divergence tr(A)."""

    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)

    def forward(self, t, x):
        return x @ self.A.T


def _pin_time(model: EESI, t_val: float) -> None:
    """Force every internal time draw to the constant `t_val` (endpoint stress test)."""
    def _draw(x, _t=t_val):
        B = x.shape[0]
        shape = (B,) + (1,) * (x.dim() - 1)
        t = torch.full(shape, _t, device=x.device, dtype=x.dtype)
        return t, t.reshape(B)
    model._draw_time = _draw


@pytest.mark.parametrize("path", ["linear", "trig", "encdec"])
@pytest.mark.parametrize("gamma", ["quad", "sqrt", "sin2"])
def test_zdot_finite_across_paths_and_gammas(path, gamma):
    """method='zdot' returns a finite [B] tensor for every (path, gamma) combo."""
    torch.manual_seed(0)
    model = _make_si(path=path, gamma=gamma)
    x1, x0 = _random_batch(B=4, d=8, L=5.0)

    ent = model.entropy_estimate(x1, x0, method="zdot")
    assert ent.shape == (4,)
    assert ent.isfinite().all(), f"non-finite zdot for {path}/{gamma}: {ent}"


def test_zdot_rejects_zero_gamma_and_unknown_method():
    """'zdot' needs a latent schedule; unknown method names still raise."""
    model = _make_si(gamma="none")
    x1, x0 = _random_batch(B=2, d=8, L=5.0)
    with pytest.raises(ValueError, match="gamma='none'"):
        model.entropy_estimate(x1, x0, method="zdot")
    with pytest.raises(ValueError, match="method"):
        model.entropy_estimate(x1, x0, method="bogus")


@pytest.mark.parametrize("gamma", ["quad", "sqrt"])
def test_zdot_unbiased_for_a_linear_field(gamma):
    """On b(t, x) = A x, 'zdot' has mean tr(A) -- the exact divergence.

    The antithetic pair reduces to z^T A z there, whose expectation over z ~ N(0, I)
    is tr(A) exactly; this is the tower-property claim (E[b.s] = E[b.(-z/gamma)])
    with every other source of error removed, since no network is involved and the
    true divergence is known in closed form. Run in float64 so the O(gamma)
    difference between the two branches is not lost to precision.
    """
    torch.manual_seed(0)
    d, B = 4, 20000
    A = torch.randn(d, d, dtype=torch.float64) / d
    net = _LinearField(A)
    model = EESI(net, net, d=d, path="linear", gamma=gamma, eps=1e-3)
    x1, x0 = _random_batch(B, d, L=5.0, seed=3)
    x1, x0 = x1.double(), x0.double()

    ent = model.entropy_estimate(x1, x0, method="zdot")
    mean, sem = ent.mean().item(), ent.std().item() / B ** 0.5
    assert abs(mean - A.trace().item()) < 4.0 * sem, (
        f"zdot mean {mean:.4f} is more than 4 sem ({sem:.4f}) from tr(A)={A.trace():.4f}"
    )


def test_zdot_matches_div_on_a_linear_field():
    """'zdot' and 'div' estimate the same quantity: agreement within Monte-Carlo error."""
    torch.manual_seed(0)
    d, B = 4, 20000
    A = torch.randn(d, d, dtype=torch.float64) / d
    net = _LinearField(A)
    model = EESI(net, net, d=d, gamma="quad", eps=1e-3, n_hutchinson_probes=1)
    x1, x0 = _random_batch(B, d, L=5.0, seed=4)
    x1, x0 = x1.double(), x0.double()

    z = model.entropy_estimate(x1, x0, method="zdot")
    v = model.entropy_estimate(x1, x0, method="div")
    sem = (z.var() / B + v.var() / B).sqrt().item()
    assert abs(z.mean().item() - v.mean().item()) < 4.0 * sem, (
        f"zdot {z.mean():.4f} vs div {v.mean():.4f} (sem {sem:.4f})"
    )


@pytest.mark.parametrize("gamma", ["quad", "sqrt", "sin2"])
def test_zdot_finite_at_the_time_endpoints(gamma):
    """Pinned at t = eps and t = 1 - eps, where a single branch's 1/gamma diverges.

    Only the antithetic average is finite there: the +z branch alone carries
    -(b.(-z/gamma)) with gamma -> 0. `eps` is the documented knob, so use the
    ~1e-3 recommended for this method rather than the 1e-6 training default.
    """
    eps = 1e-3
    net_b, net_s = _make_mlp(8, seed=0), _make_mlp(8, seed=1)
    for t_val in (eps, 1.0 - eps):
        model = EESI(net_b, net_s, d=8, gamma=gamma, eps=eps)
        _pin_time(model, t_val)
        x1, x0 = _random_batch(B=8, d=8, L=5.0, seed=5)
        ent = model.entropy_estimate(x1, x0, method="zdot")
        assert ent.isfinite().all(), f"non-finite zdot at t={t_val} ({gamma}): {ent}"


# ---- loss(entropy=...) : the training-time entropy channel ------------------


@pytest.mark.parametrize("gamma", ["quad", "sqrt"])
def test_loss_zdot_channel_matches_entropy_estimate(gamma):
    """loss(entropy='zdot') reproduces entropy_estimate(method='zdot') draw for draw.

    Both call `_draw_time` then `_noise_like`, in that order and nowhere else, so
    reseeding before each forces identical t and z -- the two must then agree to
    float64 round-off, not merely in expectation. This is what pins the training
    channel to the estimator it stands in for.
    """
    d, B = 8, 64
    x1, x0 = _random_batch(B, d, L=5.0, seed=7)
    x1, x0 = x1.double(), x0.double()
    net_b, net_s = _make_mlp(d, seed=0).double(), _make_mlp(d, seed=1).double()
    model = EESI(net_b, net_s, d=d, gamma=gamma, eps=1e-3)

    torch.manual_seed(11)
    from_loss = model.loss(x1, x0, entropy="zdot")["ent_zdot"]
    torch.manual_seed(11)
    from_estimate = model.entropy_estimate(x1, x0, method="zdot").mean()

    assert torch.allclose(from_loss, from_estimate, rtol=0, atol=1e-10), (
        f"ent_zdot {from_loss.item():.10f} != entropy_estimate {from_estimate.item():.10f}"
    )


def test_loss_dot_channel_agrees_with_entropy_estimate_in_the_mean():
    """The 'dot' channel and entropy_estimate('dot') estimate the same quantity.

    They cannot match exactly: the loss averages the +z and -z branches where the
    estimator uses one. Both are unbiased, so they agree within Monte-Carlo error --
    checked by batching over many independent draws.
    """
    torch.manual_seed(0)
    d, B, n = 4, 256, 200
    net_b, net_s = _make_mlp(d, seed=0).double(), _make_mlp(d, seed=1).double()
    model = EESI(net_b, net_s, d=d, gamma="quad", eps=1e-3)
    x1, x0 = _random_batch(B, d, L=5.0, seed=8)
    x1, x0 = x1.double(), x0.double()

    from_loss = torch.stack([model.loss(x1, x0, entropy="dot")["ent_dot"] for _ in range(n)])
    from_est = torch.stack([model.entropy_estimate(x1, x0, method="dot").mean() for _ in range(n)])
    sem = (from_loss.var() / n + from_est.var() / n).sqrt().item()
    assert abs(from_loss.mean().item() - from_est.mean().item()) < 4.0 * sem, (
        f"ent_dot {from_loss.mean():.4f} vs entropy_estimate {from_est.mean():.4f} (sem {sem:.4f})"
    )


@pytest.mark.parametrize("gamma", ["none", "quad"])
def test_entropy_channel_does_not_disturb_the_losses(gamma):
    """The channel is a read-out: same seed, same loss_b/loss_s, and still trainable.

    Also checks it stays out of the autograd graph -- a channel that carried a
    gradient would quietly train the nets on the entropy estimate.
    """
    d, B = 8, 16
    x1, x0 = _random_batch(B, d, L=5.0, seed=9)
    entropy = "dot" if gamma == "none" else "both"

    def _run(ent):
        model = _make_si(d=d, gamma=gamma, seed=0)
        torch.manual_seed(21)
        return model, model.loss(x1, x0, entropy=ent)

    _, plain = _run(None)
    model, rich = _run(entropy)

    for k in ("b", "s"):
        assert torch.equal(plain[k], rich[k]), f"loss {k!r} changed with entropy={entropy!r}"
    for k in rich:
        if k.startswith("ent_"):
            assert not rich[k].requires_grad, f"{k} is attached to the graph"
            assert rich[k].isfinite(), f"{k} not finite: {rich[k]}"

    (rich["b"] + rich["s"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.net_b.parameters())


def test_entropy_channel_guards():
    """Bad settings raise rather than reporting a meaningless number."""
    x1, x0 = _random_batch(B=4, d=8, L=5.0)

    with pytest.raises(ValueError, match="entropy must be"):
        _make_si().loss(x1, x0, entropy="bogus")
    with pytest.raises(ValueError, match="gamma='none'"):
        _make_si(gamma="none").loss(x1, x0, entropy="zdot")
    # net_s never receives a gradient here, so -b.s would read an untrained net.
    net_b, net_s = _make_mlp(8, seed=0), _make_mlp(8, seed=1)
    frozen = EESI(net_b, net_s, d=8, gamma="quad", learn_score=False)
    with pytest.raises(ValueError, match="learn_score"):
        frozen.loss(x1, x0, entropy="dot")
    assert "ent_zdot" in frozen.loss(x1, x0, entropy="zdot"), "zdot must survive learn_score=False"


def test_entropy_channel_absent_by_default():
    """The default return value is unchanged: exactly the two loss keys."""
    x1, x0 = _random_batch(B=4, d=8, L=5.0)
    assert set(_make_si().loss(x1, x0)) == {"b", "s"}
    assert set(_make_si(gamma="none").loss(x1, x0)) == {"b", "s"}


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
        test_zdot_rejects_zero_gamma_and_unknown_method,
        test_zdot_matches_div_on_a_linear_field,
        test_loss_dot_channel_agrees_with_entropy_estimate_in_the_mean,
        test_entropy_channel_guards,
        test_entropy_channel_absent_by_default,
    ]
    for _gamma in ("quad", "sqrt"):
        tests.append(lambda g=_gamma: test_loss_zdot_channel_matches_entropy_estimate(g))
    for _gamma in ("none", "quad"):
        tests.append(lambda g=_gamma: test_entropy_channel_does_not_disturb_the_losses(g))
    for _gamma in ("quad", "sqrt"):
        tests.append(lambda g=_gamma: test_zdot_unbiased_for_a_linear_field(g))
    for _gamma in ("quad", "sqrt", "sin2"):
        tests.append(lambda g=_gamma: test_zdot_finite_at_the_time_endpoints(g))
    for _path in ("linear", "trig", "encdec"):
        for _gamma in ("quad", "sqrt", "sin2"):
            tests.append(
                lambda p=_path, g=_gamma: test_zdot_finite_across_paths_and_gammas(p, g)
            )
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
