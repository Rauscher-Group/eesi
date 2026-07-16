"""Static-graph message-passing network for the 1D classical XY model.

A non-equivariant analogue of an EGNN in which the *angles* play the role of the
coordinates. It produces a per-node tangent-space scalar suitable as the velocity
field `b(t, x)` or the score field `s(t, x)` of an `EESI` stochastic interpolant
(see `eesi.interpolant`), trained on configurations of the 1D XY chain
(`eesi.datasets.xy`).

Components, in dependency order:

    scatter_add, angle_wrap, fourier_expand    tiny helpers
    chain_edge_index                           static open-chain graph builder
    XYChainConv                                one coordinate-update layer (single MLP)
    XYChainGNN                                 full backbone returning a scalar field

Design (mapping onto EGNN):

    node features h  ->  none. There is no per-node hidden state; time enters only
                         as a GLOBAL Fourier embedding concatenated to every edge's
                         attributes (identical for all nodes of a configuration).
    coordinates      ->  angles theta on S^1, updated every layer.
    rel = x[dst]-x[src]  ->  d_theta = wrap(theta[dst] - theta[src]) in (-pi, pi].
    radius graph     ->  a static open chain: each node connects to `n_neighbors`
                         nodes on each side, edge weight = 1 / (chain distance).

Because there is no node state, a message is only ever consumed by the coordinate
update, so each layer is a single MLP mapping [edge_attr, time_embedding] directly
to the scalar that weights the (wrapped) angle-difference direction. The MLP width
(`hidden`) is independent of the time-embedding size (`2*time_order`).

Manifold vs. tangent space:
    Edge geometry lives on S^1, so pairwise angle differences are wrapped. The
    coordinate accumulation and the output are tangent-space quantities in R^1, so
    they are plain additions / subtractions and are NEVER re-wrapped. As a result
    the output velocity is invariant to shifting any input angle by a multiple of
    2*pi (all wrapped differences are unchanged) and, because the time embedding is
    a Fourier expansion of 2*pi*t, invariant to shifting t by an integer.

Conventions (shared with `egnn.py`):
    edge_index = [src, dst], shape [2, E]; messages src -> dst; aggregate by dst.
    Node tensors are flat [N_tot = B*N, F]; the batch is a block-diagonal replica
    of the single-chain topology.

Forward:
    t [B] or scalar, x [B, N] (or [B, N, 1]) angles -> field of the same shape.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


# ---- helpers ---------------------------------------------------------------


def scatter_add(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """src [E, F], idx [E] long -> [dim_size, F]."""
    out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    return out.index_add_(0, idx, src)


def angle_wrap(d: torch.Tensor) -> torch.Tensor:
    """Wrap angle differences into (-pi, pi] via d - 2*pi*round(d / 2*pi)."""
    two_pi = 2.0 * math.pi
    return d - two_pi * torch.round(d / two_pi)


def fourier_expand(x: torch.Tensor, order: int) -> torch.Tensor:
    """Fourier features of a scalar field.

    Args:
        x: [...] scalar values (no trailing feature axis).
        order: number of harmonics n >= 1.

    Returns:
        [..., 2*order] = [cos(x), ..., cos(order*x), sin(x), ..., sin(order*x)].
    """
    k = torch.arange(1, order + 1, device=x.device, dtype=x.dtype)
    ang = x.unsqueeze(-1) * k          # [..., order]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


# ---- static chain graph ----------------------------------------------------


def chain_edge_index(
    N: int, n_neighbors: int, device: torch.device | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static open-chain graph: each node links to `n_neighbors` nodes per side.

    Args:
        N: number of spins in the chain.
        n_neighbors: neighbours per side (k = 1 .. n_neighbors in each direction).
        device: target device for the returned tensors.

    Returns:
        edge_index: [2, E] long, row 0 = src, row 1 = dst (single-chain indexing).
        inv_dist:   [E, 1] float, 1 / k for a bond spanning k sites.

    Open chain: bonds that would run past either end are simply omitted, so the
    two ends have fewer neighbours (matching the `np.diff` energy in `eesi.datasets.xy`).
    """
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
    src, dst, inv = [], [], []
    for i in range(N):                      # i = dst node
        for k in range(1, n_neighbors + 1):
            for j in (i - k, i + k):        # j = src node, k sites away
                if 0 <= j < N:
                    src.append(j)
                    dst.append(i)
                    inv.append(1.0 / k)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    inv_dist = torch.tensor(inv, dtype=torch.get_default_dtype(), device=device).unsqueeze(-1)
    return edge_index, inv_dist


# ---- message-passing / coordinate-update layer -----------------------------


class XYChainConv(nn.Module):
    """One coordinate-update layer.

    A single MLP maps each edge's [edge_attr, time_embedding] to a scalar weight,
    which multiplies the wrapped angle-difference direction; the result is
    scattered onto the destination nodes and added to the angles.

    Args:
        in_dim: input width = edge_attr_dim + 2*time_order.
        hidden: MLP width (independent of the time embedding size).
        mlp_layers: number of hidden layers of width `hidden` (>= 1).
        act_fn: hidden-layer activation module (default SiLU).

    Forward:
        theta [N_tot, 1], d_theta [E, 1] (wrapped), edge_attr [E, A],
        time_edges [E, 2*time_order], edge_index [2, E] -> theta_new [N_tot, 1]
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 16,
        mlp_layers: int = 2,
        act_fn: nn.Module | None = None,
    ):
        super().__init__()
        if mlp_layers < 1:
            raise ValueError(f"mlp_layers must be >= 1, got {mlp_layers}")
        act_fn = act_fn if act_fn is not None else nn.SiLU()

        # single MLP: [edge_attr, time_embedding] -> scalar update weight
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), act_fn]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(hidden, hidden), act_fn]
        last = nn.Linear(hidden, 1, bias=False)
        nn.init.xavier_uniform_(last.weight, gain=1e-3)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        theta: torch.Tensor,
        d_theta: torch.Tensor,
        edge_attr: torch.Tensor,
        time_edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        dst = edge_index[1]
        N_tot = theta.shape[0]

        scalar = self.net(torch.cat([edge_attr, time_edges], dim=-1))   # [E, 1]
        trans = d_theta * scalar                         # wrapped direction * weight
        theta_delta = scatter_add(trans, dst, dim_size=N_tot)
        return theta + theta_delta                       # plain R^1 update (not wrapped)


# ---- backbone --------------------------------------------------------------


class XYChainGNN(nn.Module):
    """Static-graph angle-flow network for the 1D XY model.

    The chain length `N` is inferred from each input and the graph is rebuilt on
    every forward call, so a single instance handles configurations of any length.

    Args:
        n_neighbors: neighbours per side for the static graph.
        edge_order: Fourier order for the (wrapped) edge angle differences.
        time_order: Fourier order for the global time embedding (base freq 2*pi,
            i.e. periodic over the interval (0, 1)).
        hidden: per-layer MLP width (default 64). Independent of `time_order`.
        n_layers: number of coordinate-update layers (default 6).
        mlp_layers: hidden layers of width `hidden` inside each layer's MLP
            (default 2).
        tanh: bound each layer's per-edge scalar with tanh * per-layer range.
        coords_range: total displacement budget spread across the layers.
        act_fn: hidden-layer activation module (default SiLU).

    Forward:
        t [B] or scalar, x [B, N] (or [B, N, 1]) -> field of the same shape.
    """

    def __init__(
        self,
        n_neighbors: int,
        edge_order: int = 4,
        time_order: int = 4,
        hidden: int = 64,
        n_layers: int = 6,
        mlp_layers: int = 2,
        act_fn: nn.Module | None = None,
    ):
        super().__init__()
        if edge_order < 1 or time_order < 1:
            raise ValueError("edge_order and time_order must be >= 1")
        self.n_neighbors = n_neighbors
        self.edge_order = edge_order
        self.time_order = time_order
        self.hidden = hidden
        self.n_layers = n_layers
        act_fn = act_fn if act_fn is not None else nn.SiLU()

        # Each layer's MLP consumes [edge_attr, raw Fourier time embedding].
        edge_attr_dim = 1 + 2 * edge_order            # inv_dist + Fourier(d_theta)
        in_dim = edge_attr_dim + 2 * time_order
        self.layers = nn.ModuleList([
            XYChainConv(
                in_dim=in_dim, hidden=hidden, mlp_layers=mlp_layers,
                act_fn=act_fn,
            )
            for _ in range(n_layers)
        ])

    def _batch_graph(
        self, B: int, N: int, device: torch.device | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the single-chain graph for length `N` and replicate it B times.

        The open-chain topology is reconstructed on every call (block-diagonal
        node ids), so the network is independent of any fixed chain length.
        """
        edge_index, inv_dist = chain_edge_index(N, self.n_neighbors, device)
        E1 = edge_index.shape[1]

        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
        ei = edge_index.unsqueeze(0) + offsets            # [B, 2, E1]
        ei = ei.permute(1, 0, 2).reshape(2, B * E1)       # [2, B*E1]
        edge_batch = torch.arange(B, device=device).repeat_interleave(E1)
        inv_b = inv_dist.repeat(B, 1)                     # [B*E1, 1]
        return ei, inv_b, edge_batch

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] == 1:
            squeeze_out = True
            x2 = x.squeeze(-1)
        elif x.dim() == 2:
            squeeze_out = False
            x2 = x
        else:
            raise ValueError(f"x must be [B, N] or [B, N, 1]; got {tuple(x.shape)}")
        B, N = x2.shape

        edge_index, inv_dist, edge_batch = self._batch_graph(B, N, device=x2.device)
        src, dst = edge_index[0], edge_index[1]
        N_tot = B * N

        # global Fourier time embedding, broadcast per edge (no projection)
        t = t if t.is_floating_point() else t.float()
        t_b = t.expand(B) if t.dim() == 0 else t.reshape(B) # [B]
        g_t = fourier_expand(2.0 * math.pi * t_b, self.time_order)    # [B, 2*time_order]
        time_edges = g_t[edge_batch]                       # [E, 2*time_order]

        theta_in = x2.reshape(N_tot, 1)
        theta = theta_in.clone()
        for layer in self.layers:
            d_theta = angle_wrap(theta[dst] - theta[src])                # [E, 1]
            edge_attr = torch.cat(
                [inv_dist, fourier_expand(d_theta.squeeze(-1), self.edge_order)], dim=-1
            ) # [E, 1 + 2*edge_order]
            theta = layer(theta, d_theta, edge_attr, time_edges, edge_index)

        v = (theta - theta_in).view(B, N)                  # tangent-space field
        return v.unsqueeze(-1) if squeeze_out else v
