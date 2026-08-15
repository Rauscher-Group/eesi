"""Tests for `eesi.averaging`: the `torch.optim.swa_utils` wrapper shared by every
training loop in the package.

Runs as either pytest or a plain script:

    pytest tests/core/test_averaging.py
    python tests/core/test_averaging.py

`make_averager` and `avg_start_step` are thin, but the arithmetic they hand off to
`AveragedModel` is easy to get backwards (equal-weight vs. exponential, which end of
the run a window counts from), and that arithmetic is invisible from the training
loops that call them. Pin it here instead of by eye in a training log.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from eesi.averaging import avg_start_step, make_averager
from eesi.config import AveragingConfig


def _model() -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(2, 2).to(torch.float64)


def _step(model: torch.nn.Linear, seed: int) -> None:
    """Nudges every parameter by a seeded random amount, standing in for an
    optimizer step -- the exact values don't matter, only that each call moves the
    weights somewhere new and reproducible."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype))


def _flat(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.flatten() for p in model.parameters()])


# --- make_averager ------------------------------------------------------------


def test_off_by_default():
    assert make_averager(_model(), AveragingConfig()) is None


def test_linear_is_the_equal_weight_running_mean():
    model = _model()
    averager = make_averager(model, AveragingConfig(kind="linear"))
    snapshots = []
    for seed in range(5):
        _step(model, seed)
        averager.update_parameters(model)
        snapshots.append(_flat(model))

    expected = torch.stack(snapshots).mean(dim=0)
    assert torch.allclose(_flat(averager.module), expected)


def test_ema_matches_the_closed_form_recursion():
    model = _model()
    decay = 0.8
    averager = make_averager(model, AveragingConfig(kind="ema", decay=decay))

    _step(model, 0)
    averager.update_parameters(model)
    expected = _flat(model).clone()               # first update seeds the average
    for seed in range(1, 5):
        _step(model, seed)
        averager.update_parameters(model)
        expected = decay * expected + (1 - decay) * _flat(model)

    assert torch.allclose(_flat(averager.module), expected)


def test_bad_kind_is_rejected():
    with pytest.raises(ValueError, match="linear.*ema"):
        make_averager(_model(), AveragingConfig(kind="polyak"))


# --- avg_start_step -------------------------------------------------------------


def test_off_and_ema_and_windowless_linear_all_start_at_step_zero():
    assert avg_start_step(AveragingConfig(), 1000) == 0
    assert avg_start_step(AveragingConfig(kind="ema", decay=0.9), 1000) == 0
    assert avg_start_step(AveragingConfig(kind="linear", window=None), 1000) == 0


def test_a_linear_window_delays_the_start_to_the_tail_of_the_run():
    assert avg_start_step(AveragingConfig(kind="linear", window=200), 1000) == 800


def test_a_window_longer_than_the_run_clamps_to_step_zero():
    assert avg_start_step(AveragingConfig(kind="linear", window=5000), 1000) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
