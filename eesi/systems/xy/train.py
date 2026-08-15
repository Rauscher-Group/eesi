"""Interpolant training for the 1D XY chain with O(2) x Z2 OT coupling (plans/EQOT_PLAN.md Phase C4).

Unlike the LJ13 side (`eesi.systems.lj13.train`), which is plain flow matching, this trains an
`xyEESI` stochastic interpolant with both a drift and a score network -- the coupling is
orthogonal to the model, so wiring it in is one line before `model.loss`.

Usage:
    python -m eesi.systems.xy.train --steps 2000 --batch 256 --J 1.0
    python -m eesi.systems.xy.train --no-align            # ablation arms
    python -m eesi.systems.xy.train --no-reflect          # drop the site reversal
    python -m eesi.systems.xy.train --no-negate           # drop the spin flip

On entropy: `xyEESI` can estimate dS, and the chain has an exact answer
(dS/N = (N-1)/N * (-J*I1(J)/I0(J))). That is the POINT of the project, not a test of the
coupling -- it runs through the trained networks, so it mixes model quality with coupling
error, and trainability degrades exactly where the physics is interesting (large J).
The model-free check that the coupling preserves the prior marginal is
tests/test_ot.py::test_xy_marginal_preserved_over_z2_u1. Do the entropy comparison in a
notebook, as a result.

`--entropy` / `train(entropy=...)` logs the b.s and b.z estimators per batch, reusing the
interpolant draw `model.loss` already made (see `EESI.loss`), so it costs no extra network
evaluation. It is there to watch dS converge DURING a run -- it is not a pass/fail
criterion for a coupling or architecture change, for exactly the reason above. The
arbiters stay model-free: `xy_transport_cost` for the coupling, bond moments of generated
samples against the analytic von Mises for the network.

Measured (real mcxy data, N=32, J=2, % vs random pairing): `align` alone -19.7%, `batch`
alone -21.8% at B=32; at B=256, -20.2% vs -31.6%. The batch layer dominates -- the
opposite of LJ13. Z2 x U(1) is a tiny group, so most of the win here is ordinary
minibatch OT wearing a symmetry-aware cost.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ...averaging import avg_start_step, make_averager
from ...config import AveragingConfig
from .data import mcxy, sample_p1_exact
from .interpolant import xyEESI
from .gnn import XYChainGNN
from .ot import xy_ot_couple, xy_transport_cost


def sample_base(B: int, N: int, device="cpu", dtype=torch.float64, generator=None):
    """Prior p0: i.i.d. uniform on (-pi, pi]. Invariant under Z2 x U(1) -- and under
    S(N) too, but S(N) is NOT a symmetry of the chain's energy, so it is not in the
    coupling group. See eesi/ot.py."""
    u = torch.rand(B, N, device=device, dtype=dtype, generator=generator)
    return u * 2.0 * np.pi - np.pi


def make_model(n_neighbors: int = 2, hidden: int = 32, n_layers: int = 3, mlp_layers = 4,
               edge_order: int = 16, time_order: int = 0, time_dim: int = 16,
               path: str = "trig", gamma: str = "sqrt", gamma_scale: float = 0.5,
               **kw) -> xyEESI:
    """An xyEESI with two independent XYChainGNNs. The nets are chain-length
    independent: N is inferred per forward, so it is not a constructor argument.

    `time_order=0` is raw `t` into the learned time MLP; > 0 appends log-spaced
    frequency pairs. Either way the embedding is non-periodic in `t`.
    """
    net_kw = dict(n_neighbors=n_neighbors, hidden=hidden, n_layers=n_layers,
                  mlp_layers=mlp_layers, edge_order=edge_order,
                  time_order=time_order, time_dim=time_dim)
    return xyEESI(XYChainGNN(**net_kw), XYChainGNN(**net_kw),
                  path=path, gamma=gamma, gamma_scale=gamma_scale, **kw)


def xy_step(model: xyEESI, x1: torch.Tensor, align: bool = True, batch: bool = True,
            reflect: bool = True, negate: bool = True, generator=None,
            entropy: str | None = None):
    """One coupled training step's losses. Returns (losses, x0, x1).

    The coupling runs under no_grad inside `xy_ot_couple`; `model.loss` already takes
    both endpoints, so no API change is needed to insert it.

    `entropy` ("dot", "zdot", "both") is passed straight to `model.loss`, which adds
    the matching detached "ent_dot"/"ent_zdot" keys to the returned dict.
    """
    B, N = x1.shape
    x0 = sample_base(B, N, device=x1.device, dtype=x1.dtype, generator=generator)
    x0, x1 = xy_ot_couple(x0, x1, align=align, batch=batch, reflect=reflect,
                          negate=negate)
    return model.loss(x1, x0, entropy=entropy), x0, x1


def load_exact_data(N: int, J: float, n_data: int = 10_000, seed: int | None = None,
                    dtype=torch.float64) -> torch.Tensor:
    """Exact Boltzmann samples for the open XY chain, wrapped to (-pi, pi].

    The default target sampler: the chain's bonds are independent von Mises, so
    this is exact and O(n_data * N). See `eesi.systems.xy.data.sample_p1_exact`.
    """
    confs = sample_p1_exact(n_data, N, J, np.random.default_rng(seed))
    return torch.as_tensor(confs, dtype=dtype)


def load_mc_data(N: int, J: float, n_save: int = 4000, n_eq: int = 50_000,
                 n_prod: int = 400_000, dtype=torch.float64) -> torch.Tensor:
    """Boltzmann samples from the classical-XY Monte Carlo sampler, wrapped to (-pi, pi].

    Kept as an independent check on `load_exact_data`, which draws from the same
    distribution exactly and far faster; train on that one.
    """
    confs, _ = mcxy(N=N, J=J, n_eq=n_eq, n_prod=n_prod, n_save=n_save)
    return torch.as_tensor((confs + np.pi) % (2 * np.pi) - np.pi, dtype=dtype)


#: History columns contributed by each `entropy` setting, and how they are labelled
#: in the log line. "b"/"s" always come first, so the default history stays the
#: 2-tuple (loss_b, loss_s) that the notebooks and tests unpack.
_ENTROPY_KEYS = {None: (), "dot": ("ent_dot",), "zdot": ("ent_zdot",),
                 "both": ("ent_dot", "ent_zdot")}
_HIST_LABELS = {"b": "loss_b", "s": "loss_s", "ent_dot": "S_dot", "ent_zdot": "S_zdot"}


def _hist_keys(entropy: str | None) -> tuple[str, ...]:
    """The ordered `model.loss` keys recorded per step, given the `entropy` setting."""
    if entropy not in _ENTROPY_KEYS:
        raise ValueError(f"entropy must be one of {sorted(map(str, _ENTROPY_KEYS))}, got {entropy!r}")
    return ("b", "s") + _ENTROPY_KEYS[entropy]


def train(data: torch.Tensor, steps: int = 2000, batch: int = 256, lr: float = 1e-3,
          align: bool = True, batch_ot: bool = True, reflect: bool = True,
          negate: bool = True, device: str = "cpu", seed: int = 0, log_every: int = 200,
          dtype=torch.float64, model: xyEESI | None = None, entropy: str | None = None,
          averaging: AveragingConfig | None = None):
    """Train an xyEESI on XY-chain data. Returns (model, history).

    `history` is a list of per-step tuples, `(loss_b, loss_s)` by default. `entropy`
    ("dot", "zdot" or "both") appends the matching per-batch entropy estimates as
    extra columns -- `(loss_b, loss_s, S_dot, S_zdot)` for "both" -- and prints them
    in the log line. They are computed inside `model.loss` from the draw it already
    made, so they add no network evaluations; see `EESI.loss`.

    The estimates inherit the model's `eps`, which floors the 1/gamma in "zdot". If
    that channel looks noisy, build the model with a looser floor --
    `make_model(..., eps=1e-3)` -- as `EESI.entropy_estimate` documents.

    `averaging` turns on a moving-average shadow of the weights (see `eesi.averaging`);
    off by default. When on, the averaged copy is left on `model.avg` -- `None`
    otherwise -- so the return signature stays `(model, hist)` either way.
    """
    torch.manual_seed(seed)
    model = (model or make_model()).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    averager = make_averager(model, averaging) if averaging else None
    avg_start = avg_start_step(averaging, steps) if averaging else 0
    data = data.to(device=device, dtype=dtype)
    keys = _hist_keys(entropy)

    hist = []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch,), device=device)
        losses, a, b = xy_step(model, data[idx], align=align, batch=batch_ot,
                               reflect=reflect, negate=negate, entropy=entropy)
        loss = losses["b"] + losses["s"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if averager is not None and step >= avg_start:
            averager.update_parameters(model)
        hist.append(tuple(losses[k].item() for k in keys))

        if log_every and (step % log_every == 0 or step == steps - 1):
            means = np.mean(hist[-log_every:], axis=0)
            cols = "  ".join(f"{_HIST_LABELS[k]} {m:9.4f}" for k, m in zip(keys, means))
            print(f"  step {step:5d}  {cols}  "
                  f"transport {xy_transport_cost(a, b).item():7.3f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
    # object.__setattr__, not plain assignment: nn.Module.__setattr__ would register
    # an nn.Module value as a submodule, which would put "avg.*" keys into
    # model.state_dict() and break torch.save(model.state_dict(), ...) downstream.
    object.__setattr__(model, "avg", averager.module if averager is not None else None)
    return model, hist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32, help="chain length")
    p.add_argument("--J", type=float, default=1.0,
                   help="coupling. Larger J = colder = harder to train; J->0 makes p1 "
                        "the prior, so dS->0 and the flow is trivial.")
    p.add_argument("--n-data", type=int, default=4000)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-align", action="store_true", help="disable the group-OT layer")
    p.add_argument("--no-batch", action="store_true", help="disable the minibatch-OT layer")
    p.add_argument("--no-reflect", action="store_true",
                   help="drop the site reversal from the coupling group")
    p.add_argument("--no-negate", action="store_true",
                   help="drop the spin flip; with --no-reflect too, SO(2) only")
    p.add_argument("--entropy", choices=("dot", "zdot", "both"), default=None,
                   help="log the per-batch entropy estimators alongside the losses: "
                        "'dot' is -b.s with the learned score, 'zdot' is -b.z with the "
                        "exact conditional score. Free (reuses the loss's own draw). A "
                        "progress diagnostic, not a validation metric. 'zdot' inherits "
                        "the model's eps, which floors its 1/gamma; if it looks noisy, "
                        "build the model with eps~1e-3.")
    p.add_argument("--mc-data", action="store_true",
                   help="draw the target with the Metropolis sampler instead of the "
                        "exact von-Mises-bond one (slow; a cross-check, not a default)")
    p.add_argument("--avg-kind", choices=("linear", "ema"), default=None,
                   help="keep a moving-average shadow of the weights alongside "
                        "training; 'linear' is an equal-weight running average, "
                        "'ema' an exponential one. Off by default.")
    p.add_argument("--avg-decay", type=float, default=0.999, help="--avg-kind ema only")
    p.add_argument("--avg-window", type=int, default=None,
                   help="--avg-kind linear only: average only the last N steps "
                        "(default: the whole run)")
    p.add_argument("--out", default=None, help="path to save the state_dict")
    a = p.parse_args()

    print(f"sampling XY chain: N={a.N} J={a.J}  ({'mc' if a.mc_data else 'exact'})")
    data = (load_mc_data(a.N, a.J, n_save=a.n_data) if a.mc_data else
            load_exact_data(a.N, a.J, n_data=a.n_data, seed=a.seed))
    print(f"data {tuple(data.shape)}  align={not a.no_align}  batch={not a.no_batch}  "
          f"reflect={not a.no_reflect}  negate={not a.no_negate}  device={a.device}")
    averaging = (AveragingConfig(kind=a.avg_kind, decay=a.avg_decay, window=a.avg_window)
                 if a.avg_kind else None)
    model, hist = train(data, steps=a.steps, batch=a.batch, lr=a.lr,
                        align=not a.no_align, batch_ot=not a.no_batch,
                        reflect=not a.no_reflect, negate=not a.no_negate,
                        device=a.device, seed=a.seed, entropy=a.entropy,
                        averaging=averaging)
    means = np.mean(hist[-100:], axis=0)
    cols = "  ".join(f"{_HIST_LABELS[k]} {m:.4f}"
                     for k, m in zip(_hist_keys(a.entropy), means))
    print(f"final (last 100): {cols}")
    if a.out:
        torch.save(model.state_dict(), a.out)
        print(f"saved -> {a.out}")
        if model.avg is not None:
            torch.save(model.avg.state_dict(), a.out + ".avg")
            print(f"saved averaged weights -> {a.out}.avg")


if __name__ == "__main__":
    main()
