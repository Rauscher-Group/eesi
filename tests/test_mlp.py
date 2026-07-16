"""Tests for `eesi.models.mlp.TimeMLP`: shapes and configurable activations.

Runs as either pytest or a plain script:

    pytest tests/test_mlp.py
    python tests/test_mlp.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch import nn

from eesi.models.mlp import TimeMLP, _ACTIVATIONS


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


# ---- runner ----------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_forward_shape_and_finite,
        test_scalar_time,
        test_activation_name_case_insensitive,
        test_activation_callable_factory,
        test_invalid_activation,
        test_d_mismatch_raises,
    ]
    tests += [lambda n=n: test_named_activations(n) for n in sorted(_ACTIVATIONS)]
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
