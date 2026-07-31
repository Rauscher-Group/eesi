"""The Satorras E(n)-GNN backbone, shared by every point-cloud system.

    unsorted_segment_sum    scatter-add aggregation over edges
    E_GCL                   one equivariant graph-conv layer
    EGNN                    the stacked backbone

This is a reimplementation of the net used in `vgsatorras/en_flows`
(`egnn/{models,gcl}.py`), kept bit-compatible with the released checkpoint
`LJ13_eq_OT_flow_matching` (Klein, Kraemer & Noe 2023; OSF https://osf.io/srqg7/),
which ships as a bare ``state_dict``. It is the architecture the surrounding
literature builds on, so it lives here in full rather than behind a dependency on
`en_flows` or `hollowflow`.

It sits in the core rather than in a system subpackage because more than one system
uses it -- LJ13 and the tangentially active polymer -- and `eesi.systems` forbids one
system importing from another. What is genuinely core about it: the net is a pure
E(n)-equivariant map on point clouds, with no notion of which subspace a particular
system lives on. That part -- the time conditioning, the edge list, and the projection
of the output velocity onto the system's tangent space -- belongs to the per-system
`Dynamics` wrapper:

    eesi.systems.lj13.dynamics.LJ13Dynamics   projects onto the COM-free subspace
    eesi.systems.tap.dynamics.TAPDynamics     projects onto {v : v_0 = 0}

Neither class is a subclass of the other, and this module knows about neither.

Do not rename `E_GCL`, `EGNN`, or their submodules: `LJ13Dynamics.from_checkpoint`
loads the released state_dict with `strict=True`, and those names are the keys. See
tests/lj13/test_checkpoint_keys.py, which pins them.
"""
from __future__ import annotations

import torch
from torch import nn


def unsorted_segment_sum(data: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    out = data.new_zeros((n, data.size(1)))
    out.scatter_add_(0, seg.unsqueeze(-1).expand(-1, data.size(1)), data)
    return out


class E_GCL(nn.Module):
    """One equivariant graph-conv layer (en_flows LJ13 path: attention + tanh + sum agg)."""

    def __init__(self, hidden_nf: int, edges_in_d: int = 1, coords_range: float = 5.0,
                 act_fn: nn.Module = nn.SiLU()):
        super().__init__()
        self.coords_range = coords_range
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + 1 + edges_in_d, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf))
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, 1, bias=False), nn.Tanh())
        self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def forward(self, h, edge_index, coord, edge_attr):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = (coord_diff ** 2).sum(1, keepdim=True)
        coord_diff = coord_diff / (torch.sqrt(radial + 1e-8) + 1)

        edge_feat = self.edge_mlp(torch.cat([h[row], h[col], radial, edge_attr], dim=1))
        edge_feat = edge_feat * self.att_mlp(edge_feat)

        trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        coord = coord + unsorted_segment_sum(trans, row, coord.size(0))

        agg = unsorted_segment_sum(edge_feat, row, h.size(0))
        h = h + self.node_mlp(torch.cat([h, agg], dim=1))
        return h, coord


class EGNN(nn.Module):
    """Satorras E(n)-GNN backbone; returns updated node features and coordinates."""

    def __init__(self, hidden_nf: int = 32, n_layers: int = 3, in_node_nf: int = 1,
                 in_edge_nf: int = 1, coords_range: float = 15.0):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, in_node_nf)
        for i in range(n_layers):
            self.add_module(f"gcl_{i}", E_GCL(hidden_nf, in_edge_nf,
                                              coords_range=coords_range / n_layers))

    def forward(self, h, x, edge_index, edge_attr):
        h = self.embedding(h)
        for i in range(self.n_layers):
            h, x = self._modules[f"gcl_{i}"](h, edge_index, x, edge_attr)
        return self.embedding_out(h), x
