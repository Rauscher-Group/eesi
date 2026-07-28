"""Tests for `eesi.systems.toy.mlp.TimeMLP`: shapes and configurable activations.

Runs as either pytest or a plain script:

    pytest tests/test_mlp.py
    python tests/test_mlp.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
from torch import nn

from eesi.systems.toy.mlp import TimeMLP, timestep_embedding, _ACTIVATIONS


def _inputs(B: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, d, generator=g)
    t = torch.rand(B, generator=g)
    return t, x


def test_forward_shape_and_finite():
    """(t, x[B, d]) -> [B, d], finite."""
    B, d = 4, 8
    net = TimeMLP(d=d, hidden=16, n_layers=2)
    t, x = _inputs(B, d)
    out = net(t, x)
    assert out.shape == (B, d)
    assert out.isfinite().all()


def test_scalar_time():
    """A 0-dim t is broadcast across the batch."""
    B, d = 4, 8
    net = TimeMLP(d=d, hidden=16, n_layers=2)
    _, x = _inputs(B, d)
    out = net(torch.tensor(0.3), x)
    assert out.shape == (B, d)


@pytest.mark.parametrize("name", sorted(_ACTIVATIONS))
def test_named_activations(name):
    """Every registered activation name builds a working network."""
    B, d = 4, 8
    net = TimeMLP(d=d, hidden=16, n_layers=3, activation=name)
    # Hidden nonlinearities are instances of the requested activation class.
    acts = [m for m in net.net if not isinstance(m, nn.Linear)]
    assert len(acts) == 3
    assert all(isinstance(a, _ACTIVATIONS[name]) for a in acts)
    # Distinct module instances per layer (no shared state).
    assert len({id(a) for a in acts}) == len(acts)
    t, x = _inputs(B, d)
    assert net(t, x).shape == (B, d)


def test_activation_name_case_insensitive():
    """Activation names are resolved case-insensitively."""
    net = TimeMLP(d=8, hidden=16, n_layers=2, activation="ReLU")
    acts = [m for m in net.net if not isinstance(m, nn.Linear)]
    assert all(isinstance(a, nn.ReLU) for a in acts)


def test_activation_callable_factory():
    """A callable/class factory is accepted and instantiated per layer."""
    net = TimeMLP(d=8, hidden=16, n_layers=2, activation=lambda: nn.LeakyReLU(0.1))
    acts = [m for m in net.net if not isinstance(m, nn.Linear)]
    assert acts and all(isinstance(a, nn.LeakyReLU) for a in acts)

    net2 = TimeMLP(d=8, hidden=16, n_layers=2, activation=nn.Tanh)
    acts2 = [m for m in net2.net if not isinstance(m, nn.Linear)]
    assert all(isinstance(a, nn.Tanh) for a in acts2)


def test_invalid_activation():
    """Unknown name -> ValueError; non-str/non-callable -> TypeError."""
    with pytest.raises(ValueError, match="activation"):
        TimeMLP(d=8, activation="swish")
    with pytest.raises(TypeError):
        TimeMLP(d=8, activation=123)


def test_d_mismatch_raises():
    """Feeding the wrong feature dimension is rejected."""
    net = TimeMLP(d=8, hidden=16, n_layers=2)
    t, x = _inputs(4, 5)
    with pytest.raises(ValueError, match="d mismatch"):
        net(t, x)


# ---- time embedding --------------------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 7, 16])
def test_timestep_embedding_shape(dim):
    """[B, 1] and [B] times both give [B, dim], finite and bounded."""
    B = 5
    t = torch.rand(B, 1)
    emb = timestep_embedding(t, dim)
    assert emb.shape == (B, dim)
    assert emb.isfinite().all()
    assert emb.abs().max() <= 1.0
    assert torch.equal(timestep_embedding(t.squeeze(-1), dim), emb)


def test_timestep_embedding_separates_endpoints():
    """t=0 and t=1 must not collide (the 2*pi-harmonic failure mode)."""
    t = torch.tensor([[0.0], [1.0]])
    emb = timestep_embedding(t, 16)
    assert (emb[0] - emb[1]).abs().max() > 1e-3
    # Distinct interior times stay distinct too.
    ts = torch.linspace(0, 1, 32).unsqueeze(-1)
    e = timestep_embedding(ts, 16)
    dists = torch.cdist(e, e) + torch.eye(32) * 10.0
    assert dists.min() > 1e-4


def test_timestep_embedding_invalid():
    """Bad dim or a non-[B, 1] time tensor is rejected."""
    with pytest.raises(ValueError, match="dim"):
        timestep_embedding(torch.rand(4, 1), 0)
    with pytest.raises(ValueError, match=r"t must be"):
        timestep_embedding(torch.rand(4, 3), 8)


def test_time_head_and_trunk_widths():
    """Time head is 2 layers of `hidden`; only the first trunk layer widens."""
    d, hidden = 5, 16
    net = TimeMLP(d=d, hidden=hidden, n_layers=3)

    time_linears = [m for m in net.time_mlp if isinstance(m, nn.Linear)]
    assert len(time_linears) == 2
    assert all(m.in_features == hidden and m.out_features == hidden for m in time_linears)

    trunk = [m for m in net.net if isinstance(m, nn.Linear)]
    assert trunk[0].in_features == hidden + d       # [time_embedding, x]
    assert all(m.in_features == hidden for m in trunk[1:])
    assert all(m.out_features == hidden for m in trunk[:-1])
    assert trunk[-1].out_features == d


def test_time_conditioning_is_used():
    """The output actually depends on t (the embedding is not dead)."""
    B, d = 4, 8
    net = TimeMLP(d=d, hidden=16, n_layers=2)
    _, x = _inputs(B, d)
    out0 = net(torch.zeros(B), x)
    out1 = net(torch.ones(B), x)
    assert (out0 - out1).abs().max() > 1e-6


# ---- runner ----------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_forward_shape_and_finite,
        test_scalar_time,
        test_activation_name_case_insensitive,
        test_activation_callable_factory,
        test_invalid_activation,
        test_d_mismatch_raises,
        test_timestep_embedding_separates_endpoints,
        test_timestep_embedding_invalid,
        test_time_head_and_trunk_widths,
        test_time_conditioning_is_used,
    ]
    tests += [lambda n=n: test_named_activations(n) for n in sorted(_ACTIVATIONS)]
    tests += [lambda k=k: test_timestep_embedding_shape(k) for k in (1, 2, 7, 16)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {getattr(t, '__name__', 'test_named_activations')}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {getattr(t, '__name__', 'lambda')}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {getattr(t, '__name__', 'lambda')}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
