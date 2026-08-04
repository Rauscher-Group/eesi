"""The TAP generative model: the velocity field v(t, x), and what you do with it.

    TAPDynamics                     velocity field on the tail-anchored subspace
    rk4_sample                      integrate the prior forward to the target
    divergence, integrate_with_logdet   exact log-density along the flow

The net is `eesi.egnn.EGNN`, the same backbone LJ13 uses. What is TAP-specific is
this wrapper: the edge list, the node features, and the projection of the output
velocity onto the tangent space {v : v_0 = 0}.

There is no `free_energy` here, unlike `eesi.systems.lj13.dynamics`. Importance
reweighting needs a target density to weight against, and an active polymer has no
Boltzmann target -- see the `eesi.systems.tap.data` docstring. `integrate_with_logdet`
still applies: the flow's own log-density is well defined regardless of whether the
distribution it transports is an equilibrium one.


Projection: subtract the tail velocity
--------------------------------------
LJ13 removes the mean velocity, keeping the flow on the COM-free subspace. TAP pins
particle 0 instead, so the projection is `v_i -> v_i - v_0` -- the same `anchor` used
for points, applied to tangents. Integrating from an anchored x0 then keeps x_0 = 0 for
all t, since row 0 is identically zero by construction.

This is the GAUGE-COVARIANT choice, and it is why it is preferred over the other way of
holding the tail fixed (discarding the net's row-0 output and leaving the rest alone).
The physically meaningful coordinates are the relative vectors x_i - x_0, and under this
projection they evolve as d(x_i - x_0)/dt = v_i - v_0, which is exactly the relative
velocity the raw field predicts. Simply zeroing row 0 would instead evolve them as v_i,
silently distorting relative motion by the tail's own velocity.

It stays O(3)-equivariant: (R v)_i - (R v)_0 = R (v_i - v_0).


Why the node index feature
--------------------------
The bare LJ13 architecture (node feature h = t, identical for every particle; fully
connected edges carrying only squared distances) is S(N)-EQUIVARIANT by construction.
That is exactly right for a homogeneous cluster and exactly wrong here: a tangentially
active polymer is a DIRECTED chain, its head and tail are not interchangeable, and such
a net cannot represent a velocity field that distinguishes monomer 1 from monomer 7.
That argument stands on the directedness alone and does not depend on the prior.

A weaker supporting point does depend on it, so read it with care: the particle ordering
is not reliably recoverable from the point cloud either. The finite equilibrium length
in the prior's bond law pins bonded pairs near |b|, which makes them much more often the
nearest neighbours than the old ideal chain did -- but there is still no excluded volume
anywhere in the prior, so non-bonded monomers may sit closer than a bond length, and the
flow's inputs at intermediate t are interpolations that need not look like either
endpoint. Recovery is not guaranteed; do not let the stiffer bond law tempt you into
dropping the feature.

`index_feature=True` (the default) therefore appends a normalized position along the
chain, i / (N-1), to the node features, giving in_node_nf = 2. This is the minimal
change that breaks S(N) while leaving the O(3) equivariance and translation invariance
of the backbone untouched -- node scalars never enter the coordinate update's direction,
only its magnitude.

`index_feature=False` restores the exact LJ13 net (in_node_nf = 1). Keep it for the
ablation that demonstrates the point: it should train visibly worse.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.func import jvp

from ...egnn import EGNN
from .data import N_DEFAULT, N_DIMS, anchor, subspace_dirs


class TAPDynamics(nn.Module):
    """Velocity field v(t, x) for TAP. x: (B, N, 3) anchored -> v: (B, N, 3), v[:, 0] = 0."""

    def __init__(self, n_particles: int = N_DEFAULT, n_dims: int = N_DIMS,
                 index_feature: bool = True, **egnn_kwargs):
        super().__init__()
        self.n_particles, self.n_dims = n_particles, n_dims
        self.index_feature = bool(index_feature)
        egnn_kwargs.setdefault("in_node_nf", 2 if self.index_feature else 1)
        self.egnn = EGNN(**egnn_kwargs)

        rows, cols = [], []  # fully connected, both directions (self-loops excluded)
        for i in range(n_particles):
            for j in range(n_particles):
                if i != j:
                    rows.append(i); cols.append(j)
        self.register_buffer("_row", torch.tensor(rows), persistent=False)
        self.register_buffer("_col", torch.tensor(cols), persistent=False)
        # normalized position along the chain, 0 at the tail and 1 at the head
        idx = torch.arange(n_particles, dtype=torch.get_default_dtype())
        self.register_buffer("_idx", (idx / max(n_particles - 1, 1)).unsqueeze(-1),
                             persistent=False)

    def _batch_edges(self, n_batch: int):
        n = self.n_particles
        off = (torch.arange(n_batch, device=self._row.device) * n).view(-1, 1)
        return (self._row + off).reshape(-1), (self._col + off).reshape(-1)

    def _node_features(self, t: torch.Tensor, B: int, dtype, device) -> torch.Tensor:
        """Node features (B*N, in_node_nf): the time, optionally plus the chain index.

        A scalar t broadcasts; a per-sample t must be repeated per particle, or
        `ones(B*n, 1) * t` silently broadcasts to (B*n, B). Sampling passes a scalar,
        training passes [B]. Both are used.
        """
        n = self.n_particles
        if t.dim() == 0:
            h = torch.ones(B * n, 1, dtype=dtype, device=device) * t
        elif t.dim() == 1 and t.shape[0] == B:
            h = t.reshape(B, 1).repeat_interleave(n, dim=0)
        else:
            raise ValueError(f"t must be a scalar or [B={B}]; got {tuple(t.shape)}")
        if not self.index_feature:
            return h
        idx = self._idx.to(dtype=dtype, device=device).repeat(B, 1)   # (B*N, 1)
        return torch.cat([h, idx], dim=1)

    def forward(self, t, x):
        """t: scalar (shared by the batch) or [B] (per-sample). x: (B, N, 3)."""
        B = x.shape[0]
        row, col = self._batch_edges(B)
        xf = x.reshape(B * self.n_particles, self.n_dims)
        t = torch.as_tensor(t, dtype=xf.dtype, device=xf.device)
        h = self._node_features(t, B, xf.dtype, xf.device)
        edge_attr = ((xf[row] - xf[col]) ** 2).sum(1, keepdim=True)
        _, x_final = self.egnn(h, xf, (row, col), edge_attr)
        vel = (x_final - xf).view(B, self.n_particles, self.n_dims)
        # project onto the tangent space {v : v_0 = 0} by subtracting the tail's own
        # velocity, so relative motion is preserved. Row 0 comes out exactly zero.
        return anchor(vel)


# --- sampling ---------------------------------------------------------------


@torch.no_grad()
def rk4_sample(dynamics: TAPDynamics, x0: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
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


# --- log-density along the flow ---------------------------------------------
#
# Integrating dx/dt = v(t,x) from the prior (t=0) to the target (t=1) while
# accumulating A = int_0^1 (div v) dt gives log q(x_1) = log p_prior(x_0) - A. All
# densities live on the DOF = 3(N-1) tail-anchored subspace, so both the prior
# normalizer and the divergence trace are taken there. The basis comes from
# `eesi.systems.tap.data`: the subspace is a property of the system, not of the flow.

_DIRS = subspace_dirs()


@torch.no_grad()
def divergence(dynamics: TAPDynamics, t, x: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Exact subspace divergence tr(J) = sum_m b_m^T J b_m via forward-mode jvp.

    Traces over the 3(N-1) tangent directions of {v : v_0 = 0}; the 3 directions that
    would move the pinned tail are outside the model's support and correctly excluded.
    Batched in chunks to cap the peak memory of the dual-number forward passes.

    `t` is a scalar shared by the whole batch (ODE integration) or a per-sample `[B]`
    tensor (entropy estimation, where every sample sits at its own time). A per-sample
    `t` is split alongside `x` so each chunk carries its own times.

    The basis is read from `x`'s shape, so the estimator is chain-length agnostic. The
    20-particle basis is precomputed at import (`_DIRS`); other sizes build on demand.
    """
    n, d = x.shape[1], x.shape[2]
    dirs = _DIRS if (n, d) == tuple(_DIRS.shape[1:]) else subspace_dirs(n, d)
    dirs = dirs.to(device=x.device, dtype=x.dtype)  # _DIRS is built on CPU at import
    batched_t = torch.is_tensor(t) and t.dim() >= 1
    x_chunks = x.split(chunk)
    t_chunks = t.to(x.dtype).split(chunk) if batched_t else [t] * len(x_chunks)
    out = []
    for xc, tc in zip(x_chunks, t_chunks):
        s = torch.zeros(xc.shape[0], dtype=x.dtype, device=x.device)
        for b in dirs:
            bb = b.unsqueeze(0).expand(xc.shape[0], -1, -1).contiguous()
            _, jv = jvp(lambda z: dynamics(tc, z), (xc,), (bb,))
            s = s + (bb * jv).sum(dim=(1, 2))
        out.append(s)
    return torch.cat(out)


@torch.no_grad()
def integrate_with_logdet(dynamics: TAPDynamics, x0: torch.Tensor, n_steps: int = 60,
                          chunk: int = 64, backward: bool = False):
    """RK4 for x with trapezoid accumulation of the divergence on the time grid.

    Forward (backward=False): x0 ~ prior at t=0 -> x1 ~ target at t=1. Returns
    (x_final, A, div_traj) where A = int (div v) dt along the path and div_traj is
    (n_steps+1, B). Then log q(x1) = log_prior(x0, k, b) - A, with `log_prior` from
    `eesi.systems.tap.data` and (k, b) the prior's bond parameters.
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
