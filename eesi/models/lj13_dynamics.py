"""The LJ13 generative model: the Satorras E(n)-GNN velocity field, and what you do with it.

    E_GCL, EGNN, LJ13Dynamics       Satorras E(n)-GNN velocity field v(t, x)
    rk4_sample                      integrate the prior forward to the target
    divergence, integrate_with_logdet   exact log-density along the flow
    free_energy                     dF and diagnostics from importance weights

Everything after the class needs a velocity field to mean anything, which is why it
lives here rather than in `eesi.datasets.lj13` -- that module holds the system's
closed-form facts (energies, the prior, subspace geometry), and this one builds on
them. The dependency runs one way: models -> datasets.

The net is a reimplementation of the Satorras E(n)-GNN used in `vgsatorras/en_flows`
(`egnn/{models,gcl}.py`), kept bit-compatible with the released checkpoint
`LJ13_eq_OT_flow_matching` (Klein, Kraemer & Noe 2023; OSF https://osf.io/srqg7/),
which ships as a bare ``state_dict``. It is the architecture the surrounding
literature builds on, so it lives here in full rather than behind a dependency on
`en_flows` or `hollowflow`.

`EGNN` here is an implementation detail of `LJ13Dynamics` and is deliberately not
exported from `eesi.models`; construct `LJ13Dynamics` instead.

Resolved architecture (from the checkpoint key shapes + the en_flows LJ13 config):
    n_particles = 13, n_dims = 3            -> 39 ambient dims
    hidden_nf   = 32, n_layers = 3
    attention   = True, tanh = True, agg = 'sum'  (coords_range_layer = 15/3 = 5)
    node feature h = ones * t   (time conditioning)
    edge feature   = squared interatomic distance
    velocity       = COM-removed coordinate update  (maps the mean-zero subspace to itself)

Time convention (validated against the shipped dataset): t=0 -> prior (COM-free
standard normal), t=1 -> LJ13 target. Integrate dx/dt = v(t, x) from 0 to 1.

Dead weight (faithful to en_flows, and present in the released checkpoint, so it must
stay for strict loading -- but do not go looking for a bug here):
    egnn.embedding_out.*    LJ13Dynamics does `_, x_final = self.egnn(...)`, discarding
                            the node-feature output entirely.
    egnn.gcl_{n-1}.node_mlp.*  the LAST layer's node update, whose h_new only ever feeds
                            embedding_out, which is itself discarded.
Together 3169 of 22468 parameters (14%) receive no gradient under any objective that
reads only the velocity. See tests/test_training.py::test_lj13_unused_parameters.

Equivariance: v is equivariant to S(13) x O(3) and invariant to translation, and
both the prior and the LJ13 target are invariant under the same group -- which is
what makes the equivariant-OT coupling in `eesi.ot` marginal-preserving. See that
module's docstring for why that condition matters.

The energies, the prior, and the reference data for this system live in
`eesi.datasets.lj13`.
"""
from __future__ import annotations

import math
import pathlib

import torch
from torch import nn
from torch.func import jvp

from ..datasets.lj13 import DOF, subspace_dirs

CKPT_PREFIX = "_flow._dynamics._dynamics._dynamics_function."

# The released checkpoint sits next to this module, so it resolves from `__file__`
# rather than the caller's cwd -- notebooks and scripts find it from anywhere.
# Gitignored: it is an OSF download, not a repo artifact. See `from_checkpoint`.
CKPT_PATH = pathlib.Path(__file__).resolve().parent / "LJ13_eq_OT_flow_matching"


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


class LJ13Dynamics(nn.Module):
    """Velocity field v(t, x) for LJ13. x: (B, 13, 3) mean-free -> v: (B, 13, 3) mean-free."""

    def __init__(self, n_particles: int = 13, n_dims: int = 3, **egnn_kwargs):
        super().__init__()
        self.n_particles, self.n_dims = n_particles, n_dims
        self.egnn = EGNN(**egnn_kwargs)
        rows, cols = [], []  # fully connected, both directions (self-loops excluded)
        for i in range(n_particles):
            for j in range(n_particles):
                if i != j:
                    rows.append(i); cols.append(j)
        self.register_buffer("_row", torch.tensor(rows), persistent=False)
        self.register_buffer("_col", torch.tensor(cols), persistent=False)

    def _batch_edges(self, n_batch: int):
        n = self.n_particles
        off = (torch.arange(n_batch, device=self._row.device) * n).view(-1, 1)
        return (self._row + off).reshape(-1), (self._col + off).reshape(-1)

    def forward(self, t, x):
        """t: scalar (shared by the batch) or [B] (per-sample). x: (B, 13, 3)."""
        B = x.shape[0]
        row, col = self._batch_edges(B)
        xf = x.reshape(B * self.n_particles, self.n_dims)
        t = torch.as_tensor(t, dtype=xf.dtype, device=xf.device)
        # Node feature h = t. A scalar t broadcasts; a per-sample t must be repeated
        # per particle, or `ones(B*n, 1) * t` silently broadcasts to (B*n, B).
        # Sampling passes a scalar; flow-matching training passes [B]. Both are used.
        if t.dim() == 0:
            h = torch.ones(B * self.n_particles, 1, dtype=xf.dtype, device=xf.device) * t
        elif t.dim() == 1 and t.shape[0] == B:
            h = t.reshape(B, 1).repeat_interleave(self.n_particles, dim=0)
        else:
            raise ValueError(f"t must be a scalar or [B={B}]; got {tuple(t.shape)}")
        edge_attr = ((xf[row] - xf[col]) ** 2).sum(1, keepdim=True)
        _, x_final = self.egnn(h, xf, (row, col), edge_attr)
        vel = (x_final - xf).view(B, self.n_particles, self.n_dims)
        return vel - vel.mean(1, keepdim=True)  # project onto mean-zero subspace

    @classmethod
    def from_checkpoint(cls, path=None, map_location="cpu", dtype=torch.float64):
        """Load the released OSF checkpoint. Defaults to `CKPT_PATH`, beside this module."""
        path = pathlib.Path(path) if path is not None else CKPT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"LJ13 checkpoint not found at {path}. It is a third-party download, "
                f"not part of the repo: fetch `LJ13_eq_OT_flow_matching` from "
                f"OSF https://osf.io/srqg7/ and put it there."
            )
        sd = torch.load(path, map_location=map_location, weights_only=False)
        sub = {k[len(CKPT_PREFIX):]: v for k, v in sd.items() if k.startswith(CKPT_PREFIX)}
        model = cls()
        model.load_state_dict(sub, strict=True)  # errors loudly if the mapping drifts
        return model.to(dtype).eval()


# --- sampling ---------------------------------------------------------------


@torch.no_grad()
def rk4_sample(dynamics: LJ13Dynamics, x0: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
    """Fixed-step RK4 integration of dx/dt = v(t, x) from t=0 (prior) to t=1 (target)."""
    x, dt = x0.clone(), 1.0 / n_steps
    for i in range(n_steps):
        t = i * dt
        k1 = dynamics(t, x)
        k2 = dynamics(t + dt / 2, x + dt / 2 * k1)
        k3 = dynamics(t + dt / 2, x + dt / 2 * k2)
        k4 = dynamics(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# --- log-density / free energy ----------------------------------------------
#
# The flow gives an exact sampling density via the instantaneous change of
# variables. Integrating dx/dt = v(t,x) from the prior (t=0) to the target
# (t=1) while accumulating A = int_0^1 (div v) dt gives
#
#     log q(x_1) = log p_prior(x_0) - A.
#
# All densities live on the 36-dim COM-free subspace (DOF = (N-1)*d), so both the
# prior normalizer and the divergence trace are taken there. Both come from
# `eesi.datasets.lj13`: the subspace is a property of the system, not of the flow.

_DIRS = subspace_dirs()


@torch.no_grad()
def divergence(dynamics: LJ13Dynamics, t: float, x: torch.Tensor,
               chunk: int = 64) -> torch.Tensor:
    """Exact subspace divergence tr(J) = sum_m b_m^T J b_m via forward-mode jvp.

    Uses the 36 orthonormal subspace directions (the 3 translational directions
    are outside the model's support and are correctly excluded). Batched in
    chunks to cap the peak memory of the dual-number forward passes.
    """
    dirs = _DIRS.to(x.dtype)
    out = []
    for xc in x.split(chunk):
        s = torch.zeros(xc.shape[0], dtype=x.dtype, device=x.device)
        for b in dirs:
            bb = b.unsqueeze(0).expand(xc.shape[0], -1, -1).contiguous()
            _, jv = jvp(lambda z: dynamics(t, z), (xc,), (bb,))
            s = s + (bb * jv).sum(dim=(1, 2))
        out.append(s)
    return torch.cat(out)


@torch.no_grad()
def integrate_with_logdet(dynamics: LJ13Dynamics, x0: torch.Tensor, n_steps: int = 60,
                          chunk: int = 64, backward: bool = False):
    """RK4 for x with trapezoid accumulation of the divergence on the time grid.

    Forward (backward=False): x0 ~ prior at t=0 -> x1 ~ target at t=1.
    Returns (x_final, A, div_traj) where A = int (div v) dt along the path and
    div_traj is (n_steps+1, B). Then log q(x1) = log_prior(x0) - A, with
    `log_prior` from `eesi.datasets.lj13`.
    """
    dt = (-1.0 if backward else 1.0) / n_steps
    t0 = 1.0 if backward else 0.0
    x = x0.clone()
    d_prev = divergence(dynamics, t0, x, chunk)
    div_traj = [d_prev]
    A = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(n_steps):
        t = t0 + i * dt
        k1 = dynamics(t, x)
        k2 = dynamics(t + dt / 2, x + dt / 2 * k1)
        k3 = dynamics(t + dt / 2, x + dt / 2 * k2)
        k4 = dynamics(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        d_now = divergence(dynamics, t + dt, x, chunk)
        A = A + 0.5 * dt * (d_prev + d_now)
        d_prev = d_now
        div_traj.append(d_now)
    return x, A, torch.stack(div_traj)


def free_energy(log_w: torch.Tensor):
    """Free-energy difference F_target - F_prior and diagnostics from log-weights.

    log_w_i = -U_target(x_i) - log q(x_i),  x_i ~ q.  Because the prior normalizer
    cancels in the difference, dF = -(logsumexp(log_w) - log N) + log Z_prior with
    log Z_prior = 0.5*DOF*log(2 pi); equivalently dF = F_target - F_prior directly.
    Returns a dict with dF, log_Z_target, ESS, ESS_frac, max_weight.
    """
    n = log_w.numel()
    log_Z_target = torch.logsumexp(log_w, 0) - math.log(n)
    log_Z_prior = 0.5 * DOF * math.log(2 * math.pi)
    dF = -(log_Z_target - log_Z_prior)                      # F_target - F_prior
    ess = torch.exp(2 * torch.logsumexp(log_w, 0) - torch.logsumexp(2 * log_w, 0))
    w_norm = torch.softmax(log_w, 0)
    return {
        "dF": dF.item(),
        "F_target": -log_Z_target.item(),
        "F_prior": -log_Z_prior,
        "log_Z_target": log_Z_target.item(),
        "ESS": ess.item(),
        "ESS_frac": (ess / n).item(),
        "max_weight": w_norm.max().item(),
    }
