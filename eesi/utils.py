"""Small training-time helpers: EMA, checkpointing, seeding, device selection,
and a periodic-eval hook.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn


# ---- seeding & device ------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (incl. CUDA / XPU if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def auto_device() -> str:
    """Pick the best available device: cuda > xpu > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


# ---- EMA -------------------------------------------------------------------


class EMA:
    """Thin wrapper around `torch.optim.swa_utils.AveragedModel` for EMA tracking.

    Usage:
        ema = EMA(model, decay=0.999)
        loss.backward(); opt.step(); opt.zero_grad()
        ema.update(model)
        # for evaluation / sampling:
        ema.model.sample_ode(...)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.avg = AveragedModel(model, avg_fn=get_ema_avg_fn(decay))

    def update(self, model: torch.nn.Module) -> None:
        self.avg.update_parameters(model)

    @property
    def model(self) -> torch.nn.Module:
        return self.avg.module

    def state_dict(self) -> dict:
        return self.avg.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.avg.load_state_dict(sd)


# ---- checkpointing ---------------------------------------------------------


def save_ckpt(
    path: str | Path,
    model: torch.nn.Module,
    ema: Optional[EMA],
    opt_b: Optional[torch.optim.Optimizer],
    opt_s: Optional[torch.optim.Optimizer],
    epoch: int,
    args: Any,
) -> None:
    """Save a single training-state checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "opt_b": opt_b.state_dict() if opt_b is not None else None,
        "opt_s": opt_s.state_dict() if opt_s is not None else None,
        "epoch": int(epoch),
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
    }
    torch.save(payload, path)


def load_ckpt(
    path: str | Path,
    model: torch.nn.Module,
    ema: Optional[EMA] = None,
    opt_b: Optional[torch.optim.Optimizer] = None,
    opt_s: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> dict:
    """Restore a checkpoint in place. Returns the payload dict for inspection."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    if opt_b is not None and payload.get("opt_b") is not None:
        opt_b.load_state_dict(payload["opt_b"])
    if opt_s is not None and payload.get("opt_s") is not None:
        opt_s.load_state_dict(payload["opt_s"])
    return payload


# ---- eval hook -------------------------------------------------------------


def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Euclidean pairwise distances: [B, N, N], diagonal is 0."""
    diff = x.unsqueeze(2) - x.unsqueeze(1)   # [B, N, N, d]
    return diff.norm(dim=-1)                  # [B, N, N]


def _pairwise_overlaps(x: torch.Tensor, sigma: float):
    """
    x: [B, N, d]
    Returns (overlaps_per_particle, overlap_fraction).
    """
    B, N, _ = x.shape
    dists = _pairwise_distances(x)
    eye = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
    dists = dists.masked_fill(eye, float("inf"))

    overlap_mask = dists < sigma
    overlaps_per_particle = float(
        overlap_mask.float().sum(dim=(1, 2)).mean().item() / N
    )

    overlap_amounts = torch.clamp(sigma - dists, min=0.0)
    total_overlap = overlap_amounts.sum(dim=(1, 2)) / 2.0
    overlap_fraction = float((total_overlap / (N * sigma)).mean().item())

    return overlaps_per_particle, overlap_fraction


@torch.no_grad()
def eval_hook(
    model: torch.nn.Module,
    held_out_loader: Iterable,
    base: torch.nn.Module,
    *,
    device: str = "cpu",
    n_batches: int = 4,
    do_sample: bool = True,
    n_sample_steps: int = 64,
    sigma: Optional[float] = None,
) -> dict:
    """Compute held-out loss on a few batches and run one ODE sample for sanity.

    Returns a dict:
        held_out_loss   mean of `model.loss` over up to `n_batches` batches
        sample_min,     observed range of the sample
        sample_max
    """
    model.eval()
    losses = []
    for i, (x1, _a1) in enumerate(held_out_loader):
        if i >= n_batches:
            break
        x1 = x1.to(device)
        B = x1.shape[0]
        _a0, x0 = base.sample((B,))
        x0 = x0.to(device)
        loss_dict = model.loss(x1, x0)
        losses.append(float(loss_dict["b"].item()))

    out = {"held_out_loss": float(np.mean(losses)) if losses else float("nan")}

    if do_sample:
        _a0, x0 = base.sample((1,))
        x0 = x0.to(device)
        x_s = model.sample_ode(x0, n_steps=n_sample_steps)
        out["sample_min"] = float(x_s.min().item())
        out["sample_max"] = float(x_s.max().item())
        if sigma is not None:
            opp, ofrac = _pairwise_overlaps(x_s, sigma)
            out["sample_overlaps_per_particle"] = opp
            out["sample_overlap_fraction"] = ofrac

    model.train()
    return out
