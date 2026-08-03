"""Tests for `eesi.interpolant.LJ13EESI` and the rank-agnostic EESI refactor.

Runs as either pytest or a plain script:

    pytest tests/lj13/test_lj13_eesi.py
    python tests/lj13/test_lj13_eesi.py

The point of the subclass is that every Gaussian it draws stays on the mean-zero
(COM-free) subspace, so the whole interpolant path does too. The nets are kept
tiny (hidden_nf=8, n_layers=1) -- these tests exercise geometry and plumbing, not
model quality. `N` is deliberately not 13, to pin the N-agnostic shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from torch.func import jvp

from eesi.systems.lj13.data import subspace_dirs
from eesi.interpolant import EESI, _div_hutchinson
from eesi.systems.lj13.interpolant import LJ13EESI
from eesi.systems.lj13.dynamics import LJ13Dynamics


N, D = 6, 3  # not 13: the subclass must be chain/cluster-size agnostic


# ---- helpers ---------------------------------------------------------------


def _nets(seed: int = 0):
    torch.manual_seed(seed)
    kw = dict(n_particles=N, n_dims=D, hidden_nf=8, n_layers=1)
    return LJ13Dynamics(**kw).double(), LJ13Dynamics(**kw).double()


def _make(path: str = "linear", gamma: str = "quad", seed: int = 0, **kw) -> LJ13EESI:
    net_b, net_s = _nets(seed)
    return LJ13EESI(net_b, net_s, d=N, path=path, gamma=gamma, **kw)


def _centered(B: int, seed: int = 0) -> torch.Tensor:
    """A COM-free (B, N, D) tensor."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, D, generator=g, dtype=torch.float64)
    return x - x.mean(dim=-2, keepdim=True)


def _is_com_free(x: torch.Tensor, atol: float = 1e-10) -> bool:
    return x.mean(dim=-2).abs().max().item() < atol


# ---- the noise hook is COM-free -------------------------------------------


def test_noise_like_is_com_free():
    """Every Gaussian the subclass draws lives on the mean-zero subspace."""
    model = _make()
    ref = torch.empty(5, N, D, dtype=torch.float64)
    for _ in range(20):
        assert _is_com_free(model._noise_like(ref))


def test_base_noise_like_is_unconstrained():
    """The base EESI hook is a plain standard normal (regression guard)."""
    net_b, net_s = _nets()
    base = EESI(net_b, net_s, d=N)
    torch.manual_seed(0)
    v = base._noise_like(torch.empty(4, N, D, dtype=torch.float64))
    assert not _is_com_free(v, atol=1e-3)  # a free normal has a nonzero COM


# ---- interpolant geometry stays on the subspace ----------------------------


def test_interpolant_and_targets_com_free():
    """With centered x0, x1, z the position and both targets are COM-free."""
    model = _make(path="trig", gamma="quad", seed=1)
    x0, x1 = _centered(4, seed=0), _centered(4, seed=1)
    z = model._noise_like(x0)
    t = torch.rand(4, 1, 1, dtype=torch.float64) * 0.8 + 0.1

    x_t, b_target, s_target = model._interpolant_sample(t, x0, x1, z)
    assert _is_com_free(x_t)
    assert _is_com_free(b_target)
    assert _is_com_free(s_target)


def test_interpolant_closed_form():
    """x_t, b_target, s_target match the Euclidean closed forms (inherited)."""
    model = _make(path="linear", gamma="quad", seed=2)
    x0, x1 = _centered(3, seed=2), _centered(3, seed=3)
    z = model._noise_like(x0)
    t = torch.rand(3, 1, 1, dtype=torch.float64) * 0.8 + 0.1

    x_t, b_target, s_target = model._interpolant_sample(t, x0, x1, z)
    alpha, beta, alpha_dot, beta_dot = model._path(t)
    g, g_dot = model._scaled_gamma(t)
    assert torch.allclose(x_t, alpha * x0 + beta * x1 + g * z, atol=1e-10)
    assert torch.allclose(b_target, alpha_dot * x0 + beta_dot * x1 + g_dot * z, atol=1e-10)
    assert torch.allclose(s_target, -z / g.clamp_min(1e-12), atol=1e-10)


def test_endpoints_recovered():
    """At t=0 the position is x0; at t=1 it is x1 (gamma vanishes at both)."""
    model = _make(path="linear", gamma="quad")
    x0, x1 = _centered(3, seed=0), _centered(3, seed=1)
    z = model._noise_like(x0)
    x0_t, _, _ = model._interpolant_sample(torch.zeros(3, 1, 1, dtype=torch.float64), x0, x1, z)
    x1_t, _, _ = model._interpolant_sample(torch.ones(3, 1, 1, dtype=torch.float64), x0, x1, z)
    assert torch.allclose(x0_t, x0, atol=1e-10)
    assert torch.allclose(x1_t, x1, atol=1e-10)


def test_antithetic_sign_relations():
    """The +z / -z branches average to the deterministic drift; score flips sign."""
    model = _make(path="linear", gamma="quad", seed=2)
    x0, x1 = _centered(3, seed=2), _centered(3, seed=3)
    z = model._noise_like(x0)
    t = torch.rand(3, 1, 1, dtype=torch.float64) * 0.8 + 0.1

    _, b_p, s_p = model._interpolant_sample(t, x0, x1, z)
    _, b_m, s_m = model._interpolant_sample(t, x0, x1, -z)
    _, _, alpha_dot, beta_dot = model._path(t)
    assert torch.allclose(0.5 * (b_p + b_m), alpha_dot * x0 + beta_dot * x1, atol=1e-10)
    assert torch.allclose(s_p, -s_m, atol=1e-10)


# ---- loss + samplers keep everything on the subspace -----------------------


@pytest.mark.parametrize("path", ["linear", "trig"])
@pytest.mark.parametrize("gamma", ["none", "quad", "sqrt"])
def test_loss_finite_and_backprops(path, gamma):
    """Every (path, gamma) gives finite b/s losses; the drift always reaches net_b.

    LJ13EESI forces `learn_score` off, so the score is learned only by the
    denoising objective (gamma != "none"). Under gamma="none" the ISM path is
    disabled -- its divergence would come from the no_grad subspace estimator --
    so loss_s is a detached zero and net_s receives no gradient.
    """
    model = _make(path=path, gamma=gamma, n_hutchinson_probes=2)
    x1, x0 = _centered(4, seed=10), _centered(4, seed=11)

    losses = model.loss(x1, x0)
    assert losses["b"].isfinite(), f"loss_b not finite for {path}/{gamma}"
    assert losses["s"].isfinite(), f"loss_s not finite for {path}/{gamma}"
    (losses["b"] + losses["s"]).backward()
    assert any(p.grad is not None for p in model.net_b.parameters())
    if gamma == "none":
        assert all(p.grad is None for p in model.net_s.parameters())
    else:
        assert any(p.grad is not None for p in model.net_s.parameters())


def test_sde_sampler_stays_com_free():
    """SDE integration (diffusion noise via the hook) never leaves the subspace."""
    model = _make(gamma="quad")
    x0 = _centered(4, seed=7)
    x1 = model.sample(x0, n_steps=8, eps=0.3)
    assert _is_com_free(x1, atol=1e-8)


# ---- divergence / entropy on the subspace ----------------------------------


def _exact_subspace_divergence(net, t: float, x: torch.Tensor) -> torch.Tensor:
    """tr(J) over the (N-1)*3 mean-zero directions via forward-mode jvp.

    The N-agnostic form of `lj13_dynamics.divergence` (whose module-level basis is
    fixed at 13 particles), used here as ground truth for the Hutchinson estimator.
    """
    dirs = subspace_dirs(x.shape[1], x.shape[2]).to(x.dtype)
    B = x.shape[0]
    s = torch.zeros(B, dtype=x.dtype)
    for b in dirs:
        bb = b.unsqueeze(0).expand(B, -1, -1).contiguous()
        _, jv = jvp(lambda z: net(torch.full((B,), t, dtype=x.dtype), z), (x,), (bb,))
        s = s + (bb * jv).sum(dim=(1, 2))
    return s


def test_hutchinson_matches_exact_subspace_divergence():
    """Centered-probe Hutchinson estimates the DOF=(N-1)*3 subspace divergence.

    Compares _div_hutchinson (with the subclass's COM-free probes v ~ N(0, P),
    E[v^T J v] = tr(PJ)) against the exact trace over the mean-zero basis. They
    must agree up to Hutchinson variance.
    """
    model = _make(gamma="quad", seed=3)
    net = model.net_b
    x = _centered(2, seed=5).requires_grad_(True)
    t = 0.4

    with torch.enable_grad():
        s = net(torch.full((2,), t, dtype=torch.float64), x)
        est = _div_hutchinson(s, x, n_probes=4000, create_graph=False,
                              noise_fn=model._noise_like)
    exact = _exact_subspace_divergence(net, t, x.detach())
    assert torch.allclose(est, exact, atol=0.5, rtol=0.05), (
        f"hutchinson {est.tolist()} vs exact {exact.tolist()}"
    )


def test_entropy_estimators_finite_and_com_free_inputs():
    """Every entropy estimator runs and the interpolant they build stays COM-free.

    "zdot" contracts net_b against the conditional score -z/gamma; the latent comes
    from `_noise_like`, so it is COM-free like everything else this class draws.
    """
    model = _make(gamma="quad", seed=4, n_hutchinson_probes=2)
    x1, x0 = _centered(4, seed=20), _centered(4, seed=21)
    for method in ("div", "dot", "zdot"):
        ent = model.entropy_estimate(x1, x0, method=method)
        assert ent.shape == (4,), f"{method}: {ent.shape}"
        assert ent.isfinite().all(), f"{method}: {ent}"


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_noise_like_is_com_free,
        test_base_noise_like_is_unconstrained,
        test_interpolant_and_targets_com_free,
        test_interpolant_closed_form,
        test_endpoints_recovered,
        test_antithetic_sign_relations,
        test_sde_sampler_stays_com_free,
        test_hutchinson_matches_exact_subspace_divergence,
        test_entropy_estimators_finite_and_com_free_inputs,
    ]
    for _path in ("linear", "trig"):
        for _gamma in ("none", "quad", "sqrt"):
            tests.append(lambda p=_path, g=_gamma: test_loss_finite_and_backprops(p, g))
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
