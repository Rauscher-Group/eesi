"""Cutoff EGNN with global linear attention for Euclidean particle systems.

Components, in dependency order:

    scatter_add, _rel_dist           tiny helpers
    radius_graph                     batched Euclidean cutoff graph builder
    EGCL                             one equivariant message-passing layer
    GlobalLinearAttention            K-token cross-attention block
    EGNN                             full backbone returning a velocity field

Time conditioning: a scalar `t` is broadcast per node and linearly projected up
to `hidden`.

Conventions:
    edge_index = [src, dst], shape [2, E].
    Messages flow src -> dst; aggregation is scattered by dst.
    rel[e] = coord[dst[e]] - coord[src[e]]  (Euclidean displacement).
    All node-level tensors are flat [N_tot=B*N, F]; spatial reshape to [B, N, F]
    is only done before the global attention block, then flattened again.

Equivariance:
    Particle permutations and orthogonal coordinate transforms (rotations,
    reflections) leave the output invariant/equivariant by construction: rel is
    a pairwise displacement (translation-invariant), the coord update is
    rel * scalar (preserves equivariance), the velocity readout is
    x_out - x_in (a displacement, equivariant).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


# ---- helpers ---------------------------------------------------------------


def scatter_add(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """src [E, F], idx [E] long -> [dim_size, F]."""
    out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    return out.index_add_(0, idx, src)


def _rel_dist(
    coords: torch.Tensor, edge_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-edge Euclidean displacement vectors and squared distances.

    Args:
        coords: [N_tot, d]
        edge_index: [2, E] long

    Returns:
        rel: [E, d] = coord[dst] - coord[src]
        dist_sq: [E, 1]
    """
    src, dst = edge_index[0], edge_index[1]
    rel = coords[dst] - coords[src]
    dist_sq = (rel * rel).sum(dim=-1, keepdim=True)
    return rel, dist_sq


def polynomial_cutoff(r, r_c, p=6):
    x = r / r_c
    mask = (r < r_c).float()
    fc = 1.0 - 0.5*(p+1)*(p+2) * x.pow(p) \
             + p*(p+2)          * x.pow(p+1) \
             - 0.5*p*(p+1)      * x.pow(p+2)
    return fc * mask


# ---- radius graph ----------------------------------------------------------


def radius_graph(
    x: torch.Tensor, r_cut: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched naive O(B*N^2) Euclidean cutoff graph.

    Args:
        x: [B, N, d] coordinates.
        r_cut: cutoff radius.

    Returns:
        edge_index: [2, E] long, row 0 = src, row 1 = dst (flat B*N indexing).
        rel: [E, d] = coord[dst] - coord[src].
        dist_sq: [E, 1] squared Euclidean distance.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be [B, N, d]; got {tuple(x.shape)}")
    B, N, d = x.shape
    device = x.device

    diff = x.unsqueeze(2) - x.unsqueeze(1)   # [B, N, N, d]; diff[b,i,j] = x[b,i] - x[b,j]
    d2 = (diff * diff).sum(dim=-1)            # [B, N, N]

    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    mask = (d2 < r_cut * r_cut) & ~eye

    b_idx, i_idx, j_idx = mask.nonzero(as_tuple=True)

    # Edge convention: src=j -> dst=i; rel = x[dst] - x[src] = x[i] - x[j] = diff[b,i,j]
    # Note: diff[b,i,j] = x[b,i] - x[b,j], which equals x[dst] - x[src]. Correct.
    src = b_idx * N + j_idx
    dst = b_idx * N + i_idx
    edge_index = torch.stack([src, dst], dim=0)
    rel = diff[b_idx, i_idx, j_idx]
    dist_sq = d2[b_idx, i_idx, j_idx].unsqueeze(-1)
    return edge_index, rel, dist_sq


# ---- EGCL (equivariant graph conv layer) -----------------------------------


class EGCL(nn.Module):
    """One equivariant message-passing layer.

    Forward:
        h [N_tot, F], coords [N_tot, d], edge_index [2, E], rel [E, d],
        dist_sq [E, 1] -> (h_new, coords_new, edge_feat)
    """

    def __init__(
        self,
        hidden: int,
        r_cut: float,
        edge_extra_in: int = 0,
        attention: bool = True,
        tanh: bool = True,
        coords_range: float = 15.0 / 6,
        act_fn: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.attention = attention
        self.tanh = tanh
        self.coords_range = coords_range
        self.r_cut = r_cut

        edge_in = 2 * hidden + 1 + edge_extra_in
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden), act_fn,
            nn.Linear(hidden, hidden), act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden + hidden, hidden), act_fn,
            nn.Linear(hidden, hidden),
        )

        coord_last = nn.Linear(hidden, 1, bias=False)
        nn.init.xavier_uniform_(coord_last.weight, gain=1e-3)
        coord_layers = [nn.Linear(hidden, hidden), act_fn, coord_last]
        if tanh:
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)

        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        rel: torch.Tensor,
        dist_sq: torch.Tensor,
        edge_extra: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        N_tot = h.shape[0]

        edge_in = [h[src], h[dst], dist_sq]
        if edge_extra is not None:
            edge_in.append(edge_extra)
        edge_feat = self.edge_mlp(torch.cat(edge_in, dim=-1))
        if self.attention:
            edge_feat = edge_feat * self.att_mlp(edge_feat)

        norm = torch.sqrt(dist_sq + 1e-8)
        fc = polynomial_cutoff(norm, self.r_cut)
        rel_normed = rel / (norm + 1.0)
        scalar = self.coord_mlp(edge_feat)
        trans = rel_normed * scalar * fc
        if self.tanh:
            trans = trans * self.coords_range
        coord_delta = scatter_add(trans, dst, dim_size=N_tot)
        coords_new = coords + coord_delta

        agg = scatter_add(edge_feat * fc, dst, dim_size=N_tot)
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_new, coords_new, edge_feat


# ---- global linear attention ----------------------------------------------


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, 2 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = (
            t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
            for t in (q, k, v)
        )
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).flatten(2)
        return self.to_out(out)


class GlobalLinearAttention(nn.Module):
    """K learnable global tokens, two cross-attentions, residual + FF.

    Linear in N: attention matrices are K x N and N x K, never N x N.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.attn1 = _Attention(dim, heads, dim_head)
        self.attn2 = _Attention(dim, heads, dim_head)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self, x: torch.Tensor, queries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, N, F]; queries: [B, K, F]. Returns updated x, queries."""
        res_x, res_q = x, queries
        x_n, q_n = self.norm_x(x), self.norm_q(queries)
        induced = self.attn1(q_n, x_n)
        out = self.attn2(x_n, induced)
        x = out + res_x
        queries = induced + res_q
        x = self.ff(x) + x
        return x, queries


# ---- backbone --------------------------------------------------------------


class EGNN(nn.Module):
    """Cutoff-EGNN for Euclidean particle systems with global linear attention.

    Args:
        d: spatial dimension.
        r_cut: cutoff radius for the radius graph.
        hidden: hidden width (default 64).
        n_layers: number of EGCL layers (default 6).
        k_attn: insert a GlobalLinearAttention block every `k_attn` EGCLs
            (default 2). Set to 0 to disable global attention.
        n_global_tokens: K, number of learnable global tokens.
        attn_heads: heads in each cross-attention.
        attn_dim_head: per-head dim in each cross-attention.
        tanh: apply tanh to the coord-MLP output (with `coords_range` scaling).
        coords_range: total displacement budget across layers.

    Forward:
        t [B] or scalar, x [B, N, d] -> [B, N, d]
    """

    def __init__(
        self,
        d: int,
        r_cut: float,
        hidden: int = 64,
        n_layers: int = 6,
        k_attn: int = 2,
        n_global_tokens: int = 4,
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        tanh: bool = True,
        coords_range: float = 15.0,
    ):
        super().__init__()
        self.d = d
        self.r_cut = float(r_cut)
        self.hidden = hidden
        self.n_layers = n_layers
        self.k_attn = k_attn

        self.input_proj = nn.Linear(1, hidden)

        per_layer_range = coords_range / max(n_layers, 1)
        self.layers = nn.ModuleList([
            EGCL(hidden=hidden, r_cut=r_cut, attention=True, tanh=tanh, coords_range=per_layer_range)
            for _ in range(n_layers)
        ])

        if k_attn > 0:
            self.global_attns = nn.ModuleList([
                GlobalLinearAttention(dim=hidden, heads=attn_heads, dim_head=attn_dim_head)
                for _ in range((n_layers - 1) // k_attn + 1)
            ])
            self.global_tokens = nn.Parameter(torch.randn(n_global_tokens, hidden) * 0.02)
        else:
            self.global_attns = nn.ModuleList()
            self.global_tokens = None

    def _embed(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        if t.dim() == 0:
            t_node = t.expand(B, N, 1).to(dtype=torch.get_default_dtype())
        else:
            t_node = t.view(B, 1, 1).expand(B, N, 1).to(dtype=torch.get_default_dtype())
        return self.input_proj(t_node)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B, N, d]; got {tuple(x.shape)}")
        B, N, d = x.shape
        if d != self.d:
            raise ValueError(f"d mismatch: model d={self.d}, input d={d}")

        edge_index, _, _ = radius_graph(x, self.r_cut)

        t_f = t if t.is_floating_point() else t.float()
        h = self._embed(t_f, B, N)
        h = h.reshape(B * N, self.hidden)
        coords = x.reshape(B * N, d).clone()
        x_in_flat = coords.clone()

        attn_idx = 0
        for i, layer in enumerate(self.layers):
            rel, dist_sq = _rel_dist(coords, edge_index)
            h, coords, _ = layer(h, coords, edge_index, rel, dist_sq)
            if self.k_attn > 0 and (i + 1) % self.k_attn == 0:
                h_dense = h.view(B, N, self.hidden)
                tokens = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
                h_dense, _ = self.global_attns[attn_idx](h_dense, tokens)
                h = h_dense.reshape(B * N, self.hidden)
                attn_idx += 1

        v_flat = coords - x_in_flat
        return v_flat.view(B, N, d)
