"""Tests for `eesi.systems.tap.interpolant.TAPEESI` and `...tap.dynamics`.

Runs as either pytest or a plain script:

    pytest tests/tap/test_tap_eesi.py
    python tests/tap/test_tap_eesi.py

The point of the subclass is that every Gaussian it draws stays on the tail-anchored
subspace, so the whole interpolant path does too. The nets are kept tiny (hidden_nf=8,
n_layers=1) -- these tests exercise geometry and plumbing, not model quality. `N` is
deliberately not 20, to pin the chain-length-agnostic shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
from scipy.spatial.transform import Rotation
from torch.func import jvp

from eesi.interpolant import EESI, _div_hutchinson
from eesi.systems.tap.data import sample_prior, subspace_dirs
from eesi.systems.tap.dynamics import TAPDynamics, divergence, rk4_sample
from eesi.systems.tap.interpolant import TAPEESI

N, D, RE_SQR = 6, 3, 4.0


# ---- helpers ---------------------------------------------------------------


def _nets(seed: int = 0, index_feature: bool = True):
    torch.manual_seed(seed)
    kw = dict(n_particles=N, n_dims=D, hidden_nf=8, n_layers=1,
              index_feature=index_feature)
    return TAPDynamics(**kw).double(), TAPDynamics(**kw).double()


def _make(path: str = "linear", gamma: str = "quad", seed: int = 0, **kw) -> TAPEESI:
    net_b, net_s = _nets(seed)
    return TAPEESI(net_b, net_s, d=N, path=path, gamma=gamma, **kw)


def _anchored(B: int, seed: int = 0) -> torch.Tensor:
    return sample_prior(B, RE_SQR, n_particles=N, n_dims=D,
                        generator=torch.Generator().manual_seed(seed))


def _is_anchored(x: torch.Tensor, atol: float = 1e-10) -> bool:
    return x[..., 0, :].abs().max().item() < atol


# ---- the noise hook --------------------------------------------------------


def test_noise_like_is_anchored():
    """Every Gaussian the subclass draws has a zero tail row."""
    model = _make()
    ref = torch.empty(5, N, D, dtype=torch.float64)
    for _ in range(20):
        assert _is_anchored(model._noise_like(ref))


def test_noise_like_is_isotropic_not_a_chain():
    """The latent is an isotropic subspace normal, NOT an ideal-chain draw.

    Guards the docstring's warning against "fixing" `_noise_like` to call
    `sample_prior`: the denoising score target -z/gamma assumes a standard normal on
    the subspace, so a correlated chain-shaped latent would quietly bias it. An ideal
    chain has strongly growing variance along the backbone; this must not.
    """
    model = _make()
    z = model._noise_like(torch.empty(20_000, N, D, dtype=torch.float64))
    var = (z ** 2).sum(-1).mean(0)                 # per-particle
    assert var[0] == 0                             # the pinned tail
    rest = var[1:]
    assert torch.allclose(rest, torch.full_like(rest, float(D)), rtol=0.05), rest.tolist()


def test_base_noise_like_is_unconstrained():
    """The base EESI hook is a plain standard normal (regression guard)."""
    net_b, net_s = _nets()
    base = EESI(net_b, net_s, d=N)
    torch.manual_seed(0)
    v = base._noise_like(torch.empty(4, N, D, dtype=torch.float64))
    assert not _is_anchored(v, atol=1e-3)


# ---- interpolant geometry --------------------------------------------------


def test_interpolant_and_targets_anchored():
    model = _make(path="trig", gamma="quad", seed=1)
    x0, x1 = _anchored(4, seed=0), _anchored(4, seed=1)
    z = model._noise_like(x0)
    t = torch.rand(4, 1, 1, dtype=torch.float64) * 0.8 + 0.1
    x_t, b_target, s_target = model._interpolant_sample(t, x0, x1, z)
    assert _is_anchored(x_t) and _is_anchored(b_target) and _is_anchored(s_target)


def test_interpolant_closed_form():
    model = _make(path="linear", gamma="quad", seed=2)
    x0, x1 = _anchored(3, seed=2), _anchored(3, seed=3)
    z = model._noise_like(x0)
    t = torch.rand(3, 1, 1, dtype=torch.float64) * 0.8 + 0.1
    x_t, b_target, s_target = model._interpolant_sample(t, x0, x1, z)
    alpha, beta, alpha_dot, beta_dot = model._path(t)
    g, g_dot = model._scaled_gamma(t)
    assert torch.allclose(x_t, alpha * x0 + beta * x1 + g * z, atol=1e-10)
    assert torch.allclose(b_target, alpha_dot * x0 + beta_dot * x1 + g_dot * z, atol=1e-10)
    assert torch.allclose(s_target, -z / g.clamp_min(1e-12), atol=1e-10)


def test_endpoints_recovered():
    model = _make(path="linear", gamma="quad")
    x0, x1 = _anchored(3, seed=0), _anchored(3, seed=1)
    z = model._noise_like(x0)
    zero = torch.zeros(3, 1, 1, dtype=torch.float64)
    x0_t, _, _ = model._interpolant_sample(zero, x0, x1, z)
    x1_t, _, _ = model._interpolant_sample(zero + 1.0, x0, x1, z)
    assert torch.allclose(x0_t, x0, atol=1e-10)
    assert torch.allclose(x1_t, x1, atol=1e-10)


def test_antithetic_sign_relations():
    model = _make(path="linear", gamma="quad", seed=2)
    x0, x1 = _anchored(3, seed=2), _anchored(3, seed=3)
    z = model._noise_like(x0)
    t = torch.rand(3, 1, 1, dtype=torch.float64) * 0.8 + 0.1
    _, b_p, s_p = model._interpolant_sample(t, x0, x1, z)
    _, b_m, s_m = model._interpolant_sample(t, x0, x1, -z)
    _, _, alpha_dot, beta_dot = model._path(t)
    assert torch.allclose(0.5 * (b_p + b_m), alpha_dot * x0 + beta_dot * x1, atol=1e-10)
    assert torch.allclose(s_p, -s_m, atol=1e-10)


# ---- the network -----------------------------------------------------------


def _raw_velocity(net, t, x):
    """The unprojected EGNN velocity, mirroring `TAPDynamics.forward` without `anchor`.

    White-box on purpose: the projection CONVENTION is the thing under test, and it is
    invisible from the outside -- both candidate projections give a zero tail row, are
    O(3)-equivariant, and keep the chain anchored under integration.
    """
    B = x.shape[0]
    row, col = net._batch_edges(B)
    xf = x.reshape(B * net.n_particles, net.n_dims)
    t = torch.as_tensor(t, dtype=xf.dtype, device=xf.device)
    h = net._node_features(t, B, xf.dtype, xf.device)
    edge_attr = ((xf[row] - xf[col]) ** 2).sum(1, keepdim=True)
    _, x_final = net.egnn(h, xf, (row, col), edge_attr)
    return (x_final - xf).view(B, net.n_particles, net.n_dims)


def test_velocity_tail_row_is_zero():
    """TAPDynamics projects onto {v : v_0 = 0} -- exactly, not approximately."""
    net, _ = _nets(seed=5)
    x = _anchored(4, seed=6)
    assert (net(0.3, x)[:, 0] == 0).all()


def test_projection_subtracts_the_tail_velocity():
    """v_i - v_0, the gauge-covariant projection -- NOT "discard row 0, keep the rest".

    Both conventions produce a zero tail row and keep the chain anchored, so nothing
    else in this file separates them. The difference is physical: relative coordinates
    x_i - x_0 evolve as v_i - v_0, so subtracting the tail velocity is what makes the
    integrated relative motion match what the raw field predicts. Zeroing row 0 instead
    would offset every relative velocity by the tail's own.
    """
    net, _ = _nets(seed=5)
    x = _anchored(4, seed=6)
    raw = _raw_velocity(net, 0.3, x)
    got = net(0.3, x)

    assert torch.allclose(got, raw - raw[:, :1], atol=1e-12)

    # and the two conventions really do differ on this input, so the check above has
    # teeth: the discarded row-0 velocity is not incidentally zero
    zeroed = torch.cat([torch.zeros_like(raw[:, :1]), raw[:, 1:]], dim=1)
    assert raw[:, 0].abs().max() > 1e-6, "degenerate input: the two conventions coincide"
    assert not torch.allclose(got, zeroed, atol=1e-6)


def test_projection_preserves_relative_velocities():
    """Differences v_i - v_j between non-tail monomers survive the projection untouched."""
    net, _ = _nets(seed=5)
    x = _anchored(4, seed=6)
    raw, got = _raw_velocity(net, 0.3, x), net(0.3, x)
    assert torch.allclose(got[:, 2] - got[:, 1], raw[:, 2] - raw[:, 1], atol=1e-12)


def test_velocity_is_o3_equivariant():
    """v(t, Rx) = R v(t, x) for R in O(3), including reflections.

    Zeroing a fixed row commutes with a global rotation, so the projection does not
    break the backbone's equivariance.
    """
    net, _ = _nets(seed=7)
    x = _anchored(4, seed=8)
    for proper in (True, False):
        R = torch.tensor(Rotation.random(random_state=1).as_matrix(), dtype=x.dtype)
        if not proper:
            R = R * torch.tensor([1.0, 1.0, -1.0], dtype=x.dtype)
        lhs = net(0.3, x @ R.T)
        rhs = net(0.3, x) @ R.T
        assert torch.allclose(lhs, rhs, atol=1e-9), (proper, (lhs - rhs).abs().max().item())


def test_index_feature_breaks_permutation_equivariance():
    """With the chain index the net distinguishes monomers; without it, it cannot.

    The whole reason `index_feature` exists. The bare LJ13 architecture is
    S(N)-equivariant, so for it a permuted input gives the permuted output exactly --
    which means it could never represent a directed chain's velocity field.

    The permutation must FIX particle 0. The projection subtracts whatever velocity
    sits at index 0, so moving a different monomer into the tail slot changes the
    subtracted vector and breaks the comparison for reasons that have nothing to do
    with the backbone's symmetry. Holding the pinned particle in place isolates the
    property actually under test -- and it is the physically meaningful question
    anyway, since the gauge already singles out particle 0.
    """
    x = _anchored(4, seed=9)
    g = torch.Generator().manual_seed(3)
    perm = torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(N - 1, generator=g)])
    assert perm[0] == 0 and not bool((perm == torch.arange(N)).all())

    plain, _ = _nets(seed=10, index_feature=False)
    lhs, rhs = plain(0.3, x[:, perm]), plain(0.3, x)[:, perm]
    assert torch.allclose(lhs, rhs, atol=1e-9), "bare net should be S(N)-equivariant"

    indexed, _ = _nets(seed=10, index_feature=True)
    lhs, rhs = indexed(0.3, x[:, perm]), indexed(0.3, x)[:, perm]
    assert not torch.allclose(lhs, rhs, atol=1e-6), \
        "index feature must break S(N) equivariance"


def test_rk4_keeps_the_chain_anchored():
    net, _ = _nets(seed=11)
    x1 = rk4_sample(net, _anchored(3, seed=12), n_steps=5)
    assert _is_anchored(x1, atol=1e-12)


def test_per_sample_time_is_accepted():
    net, _ = _nets(seed=13)
    x = _anchored(4, seed=14)
    t = torch.rand(4, dtype=torch.float64)
    assert net(t, x).shape == x.shape
    with pytest.raises(ValueError, match="t must be a scalar"):
        net(torch.rand(3, dtype=torch.float64), x)


# ---- loss + samplers -------------------------------------------------------


@pytest.mark.parametrize("path", ["linear", "trig"])
@pytest.mark.parametrize("gamma", ["none", "quad", "sqrt"])
def test_loss_finite_and_backprops(path, gamma):
    """Every (path, gamma) gives finite b/s losses; the drift always reaches net_b.

    TAPEESI forces `learn_score` off, so the score is learned only by the denoising
    objective (gamma != "none"), exactly as in LJ13EESI.
    """
    model = _make(path=path, gamma=gamma, n_hutchinson_probes=2)
    x1, x0 = _anchored(4, seed=10), _anchored(4, seed=11)
    losses = model.loss(x1, x0)
    assert losses["b"].isfinite() and losses["s"].isfinite()
    (losses["b"] + losses["s"]).backward()
    assert any(p.grad is not None for p in model.net_b.parameters())
    if gamma == "none":
        assert all(p.grad is None for p in model.net_s.parameters())
    else:
        assert any(p.grad is not None for p in model.net_s.parameters())


def test_learn_score_is_forced_off():
    model = _make(gamma="none", learn_score=True)
    assert model.learn_score is False


def test_sde_sampler_stays_anchored():
    model = _make(gamma="quad")
    x1 = model.sample(_anchored(4, seed=7), n_steps=8, eps=0.3)
    assert _is_anchored(x1, atol=1e-10)


# ---- divergence / entropy --------------------------------------------------


def _exact_subspace_divergence(net, t: float, x: torch.Tensor) -> torch.Tensor:
    """tr(J) over the 3(N-1) anchored directions, computed inline as ground truth."""
    dirs = subspace_dirs(x.shape[1], x.shape[2]).to(x.dtype)
    B = x.shape[0]
    s = torch.zeros(B, dtype=x.dtype)
    for b in dirs:
        bb = b.unsqueeze(0).expand(B, -1, -1).contiguous()
        _, jv = jvp(lambda z: net(torch.full((B,), t, dtype=x.dtype), z), (x,), (bb,))
        s = s + (bb * jv).sum(dim=(1, 2))
    return s


def test_divergence_matches_the_inline_trace():
    """`dynamics.divergence` agrees with a direct loop over the basis."""
    net, _ = _nets(seed=15)
    x = _anchored(3, seed=16)
    got = divergence(net, 0.4, x)
    want = _exact_subspace_divergence(net, 0.4, x)
    assert torch.allclose(got, want, atol=1e-10)


def test_divergence_accepts_per_sample_time():
    net, _ = _nets(seed=17)
    x = _anchored(5, seed=18)
    t = torch.full((5,), 0.4, dtype=torch.float64)
    assert torch.allclose(divergence(net, t, x), divergence(net, 0.4, x), atol=1e-10)


def test_hutchinson_matches_exact_subspace_divergence():
    """Anchored-probe Hutchinson estimates the DOF = 3(N-1) subspace divergence."""
    model = _make(gamma="quad", seed=3)
    x = _anchored(2, seed=5).requires_grad_(True)
    with torch.enable_grad():
        s = model.net_b(torch.full((2,), 0.4, dtype=torch.float64), x)
        est = _div_hutchinson(s, x, n_probes=4000, create_graph=False,
                              noise_fn=model._noise_like)
    exact = _exact_subspace_divergence(model.net_b, 0.4, x.detach())
    assert torch.allclose(est, exact, atol=0.5, rtol=0.05), (est.tolist(), exact.tolist())


def test_entropy_estimators_finite():
    model = _make(gamma="quad", seed=4, n_hutchinson_probes=2)
    x1, x0 = _anchored(4, seed=20), _anchored(4, seed=21)
    for method in ("div", "dot", "zdot"):
        ent = model.entropy_estimate(x1, x0, method=method)
        assert ent.shape == (4,) and ent.isfinite().all(), method


# ---- runner ---------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_noise_like_is_anchored,
        test_noise_like_is_isotropic_not_a_chain,
        test_base_noise_like_is_unconstrained,
        test_interpolant_and_targets_anchored,
        test_interpolant_closed_form,
        test_endpoints_recovered,
        test_antithetic_sign_relations,
        test_velocity_tail_row_is_zero,
        test_projection_subtracts_the_tail_velocity,
        test_projection_preserves_relative_velocities,
        test_velocity_is_o3_equivariant,
        test_index_feature_breaks_permutation_equivariance,
        test_rk4_keeps_the_chain_anchored,
        test_per_sample_time_is_accepted,
        test_learn_score_is_forced_off,
        test_sde_sampler_stays_anchored,
        test_divergence_matches_the_inline_trace,
        test_divergence_accepts_per_sample_time,
        test_hutchinson_matches_exact_subspace_divergence,
        test_entropy_estimators_finite,
    ]
    for _path in ("linear", "trig"):
        for _gamma in ("none", "quad", "sqrt"):
            tests.append(lambda p=_path, g=_gamma: test_loss_finite_and_backprops(p, g))
    failed = 0
    for t in tests:
        name = getattr(t, "__name__", "lambda")
        try:
            t()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
