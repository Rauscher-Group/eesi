"""Static-graph message-passing network for the 1D classical XY model.

A non-equivariant analogue of an EGNN in which the *angles* play the role of the
coordinates. It produces a per-node tangent-space scalar suitable as the velocity
field `b(t, x)` or the score field `s(t, x)` of an `EESI` stochastic interpolant
(see `eesi.interpolant`), trained on configurations of the 1D XY chain
(`eesi.systems.xy.data`).

Components, in dependency order:

    scatter_add, angle_wrap                    tiny helpers
    fourier_expand, cos_expand                 angle features (all / even-only)
    time_features                              non-periodic features of t
    chain_edge_index                           static open-chain graph builder

The three feature expansions now live in `eesi.features`, which is where the core
keeps pieces shared by more than one system (TAP's node features use them too). They
are re-imported here, so `from eesi.systems.xy.gnn import fourier_expand` still works
and this module remains the place to read about how the XY net uses them.
    XYChainConv                                one coordinate-update layer (single MLP)
    XYChainGNN                                 full backbone returning a scalar field

Design (mapping onto EGNN):

    node features h  ->  none. There is no per-node hidden state; time enters only
                         as a GLOBAL learned embedding concatenated to every edge's
                         attributes (identical for all nodes of a configuration).
                         The LJ13 EGNN in `lj13_dynamics` does the same thing one
                         step differently: h = ones * t, projected by a Linear.
    coordinates      ->  angles theta on S^1, updated every layer.
    rel = x[dst]-x[src]  ->  d_theta = wrap(theta[dst] - theta[src]) in (-pi, pi].
    radius graph     ->  a static open chain: each node connects to `n_neighbors`
                         nodes on each side, edge weight = 1 / (chain distance).

Because there is no node state, a message is only ever consumed by the coordinate
update, so each layer is a single MLP mapping [edge_attr, time_embedding] directly
to the scalar that weights the (wrapped) angle-difference direction. The MLP width
(`hidden`) is independent of the time-embedding size (`time_dim`).

Manifold vs. tangent space:
    Edge geometry lives on S^1, so pairwise angle differences are wrapped. The
    coordinate accumulation and the output are tangent-space quantities in R^1, so
    they are plain additions / subtractions and are NEVER re-wrapped. As a result
    the output velocity is invariant to shifting any input angle by a multiple of
    2*pi (all wrapped differences are unchanged).

Symmetry (the group both marginals share, review Sec. 6):
    G = O(2) x Z2^site = {theta -> +-theta + phi} x {theta_i -> theta_{N+1-i}}.
    The field must be rotation-INVARIANT, ODD under negation, and permuted under
    site reversal; the same holds of the regression targets, so imposing it is a
    free variance reduction rather than a restriction.

    Parity discipline that buys the oddness, and that later changes must keep:
    every scalar channel in the network is EVEN, and the only odd quantity is
    d_theta, appearing exactly once as the direction carried by `trans`. Hence
    `edge_attr` uses `cos_expand`, not `fourier_expand` -- a sin(k*d_theta)
    channel would be odd and would break it. Oddness then follows inductively:
    d_theta odd x phi even => trans odd => theta^(l) odd at every layer => the
    output theta^(L) - theta^(0) is odd. This is EGNN's invariant-scalar times
    equivariant-direction structure. The time embedding is invariant under
    negation, so it needs nothing. Tested in tests/test_xygnn.py.

Conventions (shared with `egnn.py`):
    edge_index = [src, dst], shape [2, E]; messages src -> dst; aggregate by dst.
    Node tensors are flat [N_tot = B*N, F]; the batch is a block-diagonal replica
    of the single-chain topology.

Forward:
    t [B] or scalar, x [B, N] (or [B, N, 1]) angles -> field of the same shape.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import torch
from torch import nn

# re-exported: the expansions are core, but this is their documented home of use
from ...features import cos_expand, fourier_expand, time_features


# ---- helpers ---------------------------------------------------------------


def scatter_add(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """src [E, F], idx [E] long -> [dim_size, F]."""
    out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    return out.index_add_(0, idx, src)


def angle_wrap(d: torch.Tensor) -> torch.Tensor:
    """Wrap angle differences into (-pi, pi] via d - 2*pi*round(d / 2*pi)."""
    two_pi = 2.0 * math.pi
    return d - two_pi * torch.round(d / two_pi)


# ---- static chain graph ----------------------------------------------------


@lru_cache(maxsize=None)
def _chain_edge_index_cached(
    N: int, n_neighbors: int, device: torch.device | None, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uncached builder behind `chain_edge_index`; see there for the semantics."""
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
    inv_dist = torch.tensor(inv, dtype=dtype, device=device).unsqueeze(-1)
    return edge_index, inv_dist


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
    two ends have fewer neighbours (matching the `np.diff` energy in `eesi.systems.xy.data`).

    Cached on (N, n_neighbors, device, default dtype): repeated calls return the
    SAME tensor objects, which matters because the graph is otherwise rebuilt by
    this Python loop on every forward. All downstream use is read-only (`inv_dist`
    is only ever `cat`-ed, `edge_index` only indexed), so the sharing is safe --
    but do not mutate the returned tensors in place.
    """
    dev = torch.device(device) if device is not None else None
    return _chain_edge_index_cached(N, n_neighbors, dev, torch.get_default_dtype())


@lru_cache(maxsize=32)
def _batch_graph_cached(
    B: int, N: int, n_neighbors: int, device: torch.device | None, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`B` block-diagonal replicas of the single-chain graph. Read-only, as above.

    Bounded cache: `B` is constant during training but differs at sampling time,
    so an unbounded one would pin a tensor per batch size ever used.
    """
    edge_index, inv_dist = _chain_edge_index_cached(N, n_neighbors, device, dtype)
    E1 = edge_index.shape[1]

    offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
    ei = edge_index.unsqueeze(0) + offsets            # [B, 2, E1]
    ei = ei.permute(1, 0, 2).reshape(2, B * E1)       # [2, B*E1]
    edge_batch = torch.arange(B, device=device).repeat_interleave(E1)
    inv_b = inv_dist.repeat(B, 1)                     # [B*E1, 1]
    return ei, inv_b, edge_batch


# ---- message-passing / coordinate-update layer -----------------------------


class XYChainConv(nn.Module):
    """One coordinate-update layer.

    A single MLP maps each edge's [edge_attr, time_embedding] to a scalar weight,
    which multiplies the wrapped angle-difference direction; the result is
    scattered onto the destination nodes and added to the angles.

    Args:
        in_dim: input width = edge_attr_dim + time_dim.
        hidden: MLP width (independent of the time embedding size).
        mlp_layers: number of hidden layers of width `hidden` (>= 1).
        act_fn: hidden-layer activation module (default SiLU).

    Forward:
        theta [N_tot, 1], d_theta [E, 1] (wrapped), edge_attr [E, A],
        time_edges [E, time_dim], edge_index [2, E] -> theta_new [N_tot, 1]
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
        src = edge_index[0]
        N_tot = theta.shape[0]

        scalar = self.net(torch.cat([edge_attr, time_edges], dim=-1))   # [E, 1]
        trans = d_theta * scalar                         # wrapped direction * weight
        theta_delta = scatter_add(trans, src, dim_size=N_tot)
        return theta + theta_delta                       # plain R^1 update (not wrapped)


# ---- backbone --------------------------------------------------------------


class XYChainGNN(nn.Module):
    """Static-graph angle-flow network for the 1D XY model.

    The chain length `N` is inferred from each input and the graph is rebuilt on
    every forward call, so a single instance handles configurations of any length.

    Args:
        n_neighbors: neighbours per side for the static graph.
        edge_order: number of cosine harmonics of the (wrapped) edge angle
            differences; cosines only, so the field stays odd under theta -> -theta.
        time_order: log-spaced frequency pairs appended to raw `t` before the
            learned projection; 0 (default) feeds raw `t` alone. See
            `time_features` -- the embedding is non-periodic either way.
        time_dim: width of the learned global time embedding (default 32).
        hidden: per-layer MLP width (default 64). Independent of `time_dim`.
        n_layers: number of coordinate-update layers (default 6).
        mlp_layers: hidden layers of width `hidden` inside each layer's MLP
            (default 2).
        act_fn: hidden-layer activation module (default SiLU).

    Forward:
        t [B] or scalar, x [B, N] (or [B, N, 1]) -> field of the same shape.
    """

    def __init__(
        self,
        n_neighbors: int,
        edge_order: int = 4,
        time_order: int = 0,
        time_dim: int = 32,
        hidden: int = 64,
        n_layers: int = 6,
        mlp_layers: int = 2,
        act_fn: nn.Module | None = None,
    ):
        super().__init__()
        if edge_order < 1:
            raise ValueError(f"edge_order must be >= 1, got {edge_order}")
        if time_order < 0:
            raise ValueError(f"time_order must be >= 0, got {time_order}")
        self.n_neighbors = n_neighbors
        self.edge_order = edge_order
        self.time_order = time_order
        self.time_dim = time_dim
        self.hidden = hidden
        self.n_layers = n_layers
        act_fn = act_fn if act_fn is not None else nn.SiLU()

        # Learned projection of the (non-periodic) time features, computed once
        # per forward and broadcast to every edge.
        self.time_mlp = nn.Sequential(
            nn.Linear(1 + 2 * time_order, time_dim), act_fn,
            nn.Linear(time_dim, time_dim),
        )

        # Each layer's MLP consumes [edge_attr, time embedding].
        edge_attr_dim = 1 + edge_order                # inv_dist + cos(k*d_theta)
        in_dim = edge_attr_dim + time_dim
        self.layers = nn.ModuleList([
            XYChainConv(
                in_dim=in_dim, hidden=hidden, mlp_layers=mlp_layers,
                act_fn=act_fn,
            )
            for _ in range(n_layers)
        ])

    def _batch_graph(
        self, B: int, N: int, device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The single-chain graph for length `N`, replicated B times.

        `N` comes from the input rather than the constructor, so the network is
        independent of any fixed chain length; the topology itself is cached (see
        `_batch_graph_cached`) and the returned tensors must not be mutated.
        """
        dev = torch.device(device) if device is not None else None
        return _batch_graph_cached(
            B, N, self.n_neighbors, dev, dtype or torch.get_default_dtype()
        )

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

        edge_index, inv_dist, edge_batch = self._batch_graph(
            B, N, device=x2.device, dtype=x2.dtype
        )
        src, dst = edge_index[0], edge_index[1]
        N_tot = B * N

        # global learned time embedding, computed once and broadcast per edge
        t = t if t.is_floating_point() else t.to(x2.dtype)
        t_b = t.expand(B) if t.dim() == 0 else t.reshape(B) # [B]
        g_t = self.time_mlp(time_features(t_b, self.time_order))      # [B, time_dim]
        time_edges = g_t[edge_batch]                       # [E, time_dim]

        theta_in = x2.reshape(N_tot, 1)
        theta = theta_in.clone()
        for layer in self.layers:
            d_theta = angle_wrap(theta[dst] - theta[src])                # [E, 1]
            edge_attr = torch.cat(
                [inv_dist, cos_expand(d_theta.squeeze(-1), self.edge_order)], dim=-1
            ) # [E, 1 + edge_order], every channel EVEN in d_theta
            theta = layer(theta, d_theta, edge_attr, time_edges, edge_index)

        v = (theta - theta_in).view(B, N)                  # tangent-space field
        return v.unsqueeze(-1) if squeeze_out else v
