"""Moving-average shadow weights, built on `torch.optim.swa_utils`.

The parsing/validation half of an `averaging:` config block lives in `eesi.config`
(`AveragingConfig`, no torch import); this module is the torch-touching mechanics half,
mirroring the `eesi.rundir` / `eesi.config` split for checkpoints. Every training loop
in the package calls only `make_averager` and `avg_start_step`, plus
`AveragedModel.update_parameters` after each optimizer step.
"""
from __future__ import annotations

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .config import AveragingConfig

__all__ = ["make_averager", "avg_start_step"]


def make_averager(model: torch.nn.Module, cfg: AveragingConfig) -> AveragedModel | None:
    """A shadow `AveragedModel` of `model`, or `None` if averaging is off.

    "ema" uses swa_utils' own EMA averaging function at `cfg.decay`. "linear" uses
    swa_utils' default averaging function, an equal-weight running mean of every
    iterate it is shown -- which is why the windowed case (see `avg_start_step`) works
    by delaying when updates start, rather than maintaining a ring buffer.
    """
    if cfg.kind is None:
        return None
    if cfg.kind == "ema":
        return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.decay))
    if cfg.kind == "linear":
        return AveragedModel(model)
    raise ValueError(f"averaging.kind must be 'linear' or 'ema', got {cfg.kind!r}")


def avg_start_step(cfg: AveragingConfig, total_steps: int) -> int:
    """First absolute step (0-indexed) at which `update_parameters` should be called.

    Only "linear" with a `window` delays the start: averaging the equal-weight running
    mean from `total_steps - window` onward is exactly the uniform average of the last
    `window` iterates, with no history buffer needed. Everything else averages from 0.
    """
    if cfg.kind == "linear" and cfg.window is not None:
        return max(0, total_steps - cfg.window)
    return 0
