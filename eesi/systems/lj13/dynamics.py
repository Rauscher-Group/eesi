"""The LJ13 generative model: the Satorras E(n)-GNN velocity field, and what you do with it.

    LJ13Dynamics                    velocity field v(t, x) on the COM-free subspace
    rk4_sample                      integrate the prior forward to the target
    divergence, integrate_with_logdet   exact log-density along the flow
    free_energy                     dF and diagnostics from importance weights

Everything after the class needs a velocity field to mean anything, which is why it
lives here rather than in `eesi.systems.lj13.data` -- that module holds the system's
closed-form facts (energies, the prior, subspace geometry), and this one builds on
them. The dependency runs one way: models -> datasets.

The net itself is `eesi.egnn.EGNN`, in the core because more than one system uses it
(see that module for the en_flows provenance and the checkpoint-compatibility note).
What is LJ13-specific, and therefore here, is the wrapper: the fully-connected edge
list, the time conditioning, and the projection of the output velocity onto the
mean-zero subspace. `EGNN` is an implementation detail of `LJ13Dynamics` and is
deliberately not re-exported from `eesi.systems.lj13`; construct `LJ13Dynamics`.

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
what makes the equivariant-OT coupling in `eesi.systems.lj13.ot` marginal-preserving. See that
module's docstring for why that condition matters.

The energies, the prior, and the reference data for this system live in
`eesi.systems.lj13.data`.
"""
from __future__ import annotations

import math
import pathlib

import torch
from torch import nn
from torch.func import jvp

from ...egnn import EGNN
from .data import DOF, subspace_dirs

CKPT_PREFIX = "_flow._dynamics._dynamics._dynamics_function."

# The released checkpoint sits next to this module, so it resolves from `__file__`
# rather than the caller's cwd -- notebooks and scripts find it from anywhere.
# Gitignored: it is an OSF download, not a repo artifact. See `from_checkpoint`.
CKPT_PATH = pathlib.Path(__file__).resolve().parent / "LJ13_eq_OT_flow_matching"
#CKPT_PATH = pathlib.Path(__file__).resolve().parent / "LJ55_eq_OT_flow_matching"

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
# `eesi.systems.lj13.data`: the subspace is a property of the system, not of the flow.

_DIRS = subspace_dirs()


@torch.no_grad()
def divergence(dynamics: LJ13Dynamics, t, x: torch.Tensor,
               chunk: int = 64) -> torch.Tensor:
    """Exact subspace divergence tr(J) = sum_m b_m^T J b_m via forward-mode jvp.

    Uses the 36 orthonormal subspace directions (the 3 translational directions
    are outside the model's support and are correctly excluded). Batched in
    chunks to cap the peak memory of the dual-number forward passes.

    `t` is a scalar shared by the whole batch (ODE integration) or a per-sample
    `[B]` tensor (entropy estimation, where every sample sits at its own time).
    A per-sample `t` is split alongside `x` so each chunk carries its own times.

    The subspace basis is read from `x`'s shape, so the estimator is
    cluster-size agnostic (as is `LJ13EESI`). The 13-particle basis is
    precomputed at import (`_DIRS`); other sizes build it on demand.
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
def integrate_with_logdet(dynamics: LJ13Dynamics, x0: torch.Tensor, n_steps: int = 60,
                          chunk: int = 64, backward: bool = False):
    """RK4 for x with trapezoid accumulation of the divergence on the time grid.

    Forward (backward=False): x0 ~ prior at t=0 -> x1 ~ target at t=1.
    Returns (x_final, A, div_traj) where A = int (div v) dt along the path and
    div_traj is (n_steps+1, B). Then log q(x1) = log_prior(x0) - A, with
    `log_prior` from `eesi.systems.lj13.data`.
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
