"""Group-agnostic pieces of the equivariant-OT couplings.

The per-system couplings live beside their systems:

    eesi.systems.lj13.ot   LJ13 point clouds over S(N) x SO(3)   (Klein et al. 2023)
    eesi.systems.xy.ot     XY chains over O(2) x Z2^site

This module holds only what both need: centering, the minibatch assignment, the
batched Hungarian, and the plain Euclidean transport cost. See the two system
modules for the theory; both are validated against `tests/ot_reference.py`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:                                    # optional: batched CUDA Hungarian
    from torch_linear_assignment import batch_linear_assignment
    _HAS_BLA = True
except ImportError:                     # pragma: no cover
    _HAS_BLA = False


# ---- helpers ---------------------------------------------------------------


def center(x: torch.Tensor) -> torch.Tensor:
    """Remove the center of mass. x: (..., N, d)."""
    return x - x.mean(dim=-2, keepdim=True)


def _outer_assignment(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minibatch OT: linear_sum_assignment on a (B, B) cost matrix.

    Just one scipy call, only about 1 ms at B=256, not worth moving to GPU. 
    Returns (rows, cols) as long tensors on M's device.
    """
    r, c = linear_sum_assignment(M.detach().cpu().numpy())
    return (torch.as_tensor(r, device=M.device, dtype=torch.long),
            torch.as_tensor(c, device=M.device, dtype=torch.long))


def _hungarian_nd(D: torch.Tensor) -> torch.Tensor:
    """Batched Hungarian on (K, N, N) costs -> (K, N) column index per row.

    Uses torch-linear-assignment on CUDA when available, else loops scipy.
    """
    if _HAS_BLA and D.is_cuda:
        # batch_linear_assignment returns (K, N): for each row, the assigned column.
        return batch_linear_assignment(D.contiguous())
    
    # CPU fallback
    out = np.empty(D.shape[:2], dtype=np.int64)
    Dn = D.detach().cpu().numpy()
    for k in range(Dn.shape[0]):
        _, c = linear_sum_assignment(Dn[k])
        out[k] = c
    return torch.as_tensor(out, device=D.device)


# ---- transport cost --------------------------------------------------------


def transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Mean per-sample ||x0 - x1||^2 of an already-coupled pair. (B, N, d)."""
    return ((x0 - x1) ** 2).sum(dim=(-1, -2)).mean()
