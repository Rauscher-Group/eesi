"""Flow-matching training for LJ13 with equivariant-OT coupling (plans/EQOT_PLAN.md Phase B4).

Linear interpolant, t=0 -> prior, t=1 -> data. This is HollowFlow's convention
(mu_t = x0*(1-t) + x1*t) and it matches the checkpoint conventions in
`eesi.systems.lj13.dynamics`, so a model trained here is directly comparable to the
released one.

Data-driven: trained on the OSF MCMC samples, no energy function anywhere.

Usage:
    python -m eesi.systems.lj13.train --steps 2000 --batch 64
    python -m eesi.systems.lj13.train --no-align --no-batch     # ablation arms

The `--no-align` / `--no-batch` flags expose the 2x2 of plans/EQOT_PLAN.md's "Attributing
the win": `align` is OT over the group S(13) x SO(3), `batch` is OT over the minibatch.
Measured on real data, `align` dominates for LJ13 (-74.6% vs -29.0% of random pairing
at B=32) -- the opposite of the XY chain, where the group is tiny and `batch` wins.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .data import REF_DATA_PATH, load_ref_data, sample_prior
from .interpolant import LJ13EESI
from .dynamics import LJ13Dynamics
from ...averaging import avg_start_step, make_averager
from ...config import AveragingConfig
from ...ot import transport_cost
from .ot import equivariant_ot_couple


def flow_matching_loss(net, x0: torch.Tensor, x1: torch.Tensor, sigma: float = 0.01,
                       align: bool = True, batch: bool = True, generator=None):
    """One flow-matching loss on an already-drawn (x0, x1). x0, x1: (B, 13, 3).

    The coupling runs under no_grad inside `equivariant_ot_couple` -- it is a data
    pairing step, and the regression sees the aligned pair as fixed targets. Never
    backprop through it.

    Returns (loss, x0_coupled, x1_coupled); the latter two are returned so callers can
    assert mean-freeness and report transport cost without recomputing the coupling.
    """
    x0, x1 = equivariant_ot_couple(x0, x1, align=align, batch=batch)

    B = x1.shape[0]
    t = torch.rand(B, 1, 1, device=x1.device, dtype=x1.dtype, generator=generator)
    mu_t = x0 * (1.0 - t) + x1 * t
    if sigma > 0:
        noise = sample_prior(B, dtype=x1.dtype, device=x1.device, generator=generator)
        x_t = mu_t + sigma * noise
    else:
        x_t = mu_t
    u_t = x1 - x0                                   # the target velocity, constant in t

    v = net(t.view(B), x_t)
    return ((v - u_t) ** 2).mean(), x0, x1


def train(data: torch.Tensor, steps: int = 2000, batch: int = 64, lr: float = 1e-3,
          sigma: float = 0.01, align: bool = True, batch_ot: bool = True,
          device: str = "cpu", seed: int = 0, log_every: int = 200, dtype=torch.float64,
          averaging: AveragingConfig | None = None):
    """Train an LJ13Dynamics velocity field. Returns (net, history).

    `averaging` turns on a moving-average shadow of the weights (see `eesi.averaging`);
    off by default. When on, the averaged copy is left on `net.avg` -- `None`
    otherwise -- so the return signature stays `(net, hist)` either way.
    """
    torch.manual_seed(seed)
    net = LJ13Dynamics().to(device=device, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    averager = make_averager(net, averaging) if averaging else None
    avg_start = avg_start_step(averaging, steps) if averaging else 0
    data = data.to(device=device, dtype=dtype)

    hist = []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch,), device=device)
        x1 = data[idx]
        x0 = sample_prior(batch, dtype=dtype, device=device)

        loss, a, b = flow_matching_loss(net, x0, x1, sigma=sigma,
                                        align=align, batch=batch_ot)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if averager is not None and step >= avg_start:
            averager.update_parameters(net)
        hist.append(loss.item())

        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {np.mean(hist[-log_every:]):9.4f}  "
                  f"transport {transport_cost(a, b).item():7.2f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
    # object.__setattr__, not plain assignment: nn.Module.__setattr__ would register
    # an nn.Module value as a submodule, which would put "avg.*" keys into
    # net.state_dict() and break torch.save(net.state_dict(), ...) downstream.
    object.__setattr__(net, "avg", averager.module if averager is not None else None)
    return net, hist


# ---- stochastic-interpolant path (LJ13EESI) --------------------------------
#
# The plain flow matching above is the released-checkpoint convention. The path
# below trains a proper stochastic interpolant (LJ13EESI): a drift AND a score
# field, a latent noise schedule, and interpolant-based entropy. The equivariant
# OT coupling is identical -- only the loss changes. See plans/LJ13_SI_PLAN.md.
#
# On entropy: `--entropy` / `train_si(entropy=...)` logs the b.s and b.z estimators per
# batch, reusing the interpolant draw `model.loss` already made (see `EESI.loss`), so it
# costs no extra network evaluation. As on the XY side, it is there to watch dS converge
# DURING a run -- not a pass/fail criterion for a coupling or architecture change, since
# it runs through the trained networks. The model-free arbiter for the coupling stays
# `transport_cost`.


def make_si_model(n_particles: int = 13, n_dims: int = 3, hidden_nf: int = 32,
                    n_layers: int = 3, path: str = "linear", gamma: str = "quad", 
                    gamma_scale: float = 1.0, **kw) -> LJ13EESI:
    """An LJ13EESI wrapping two independent LJ13Dynamics fields (drift + score)."""
    egnn_kw = {"hidden_nf":hidden_nf, "n_layers": n_layers}
    net_b = LJ13Dynamics(n_particles=n_particles, n_dims=n_dims, **egnn_kw)
    net_s = LJ13Dynamics(n_particles=n_particles, n_dims=n_dims, **egnn_kw)
    return LJ13EESI(net_b, net_s, d=n_particles, path=path, gamma=gamma,
                    gamma_scale=gamma_scale, **kw)


def si_step(model: LJ13EESI, x1: torch.Tensor, align: bool = True, batch: bool = True,
            generator=None, entropy: str | None = None):
    """One coupled SI training step's losses. Returns (losses, x0, x1).

    Mirrors eesi.systems.xy.train.xy_step: sample a COM-free base, OT-couple it to the data,
    then hand both endpoints to model.loss. The coupling runs under no_grad.

    `entropy` ("dot", "zdot", "both") is passed straight to `model.loss`, which adds
    the matching detached "ent_dot"/"ent_zdot" keys to the returned dict.
    """
    B, N, D = x1.shape
    x0 = sample_prior(B, n_particles=N, n_dims=D, dtype=x1.dtype, device=x1.device,
                      generator=generator)
    x0, x1 = equivariant_ot_couple(x0, x1, align=align, batch=batch)
    return model.loss(x1, x0, entropy=entropy), x0, x1


#: History columns contributed by each `entropy` setting, and how they are labelled
#: in the log line. "b"/"s" always come first, so the default history stays the
#: 2-tuple (loss_b, loss_s) that the notebooks and tests unpack. Mirrors
#: eesi.systems.xy.train.
_ENTROPY_KEYS = {None: (), "dot": ("ent_dot",), "zdot": ("ent_zdot",),
                 "both": ("ent_dot", "ent_zdot")}
_HIST_LABELS = {"b": "loss_b", "s": "loss_s", "ent_dot": "S_dot", "ent_zdot": "S_zdot"}


def _hist_keys(entropy: str | None) -> tuple[str, ...]:
    """The ordered `model.loss` keys recorded per step, given the `entropy` setting."""
    if entropy not in _ENTROPY_KEYS:
        raise ValueError(f"entropy must be one of {sorted(map(str, _ENTROPY_KEYS))}, got {entropy!r}")
    return ("b", "s") + _ENTROPY_KEYS[entropy]


def train_si(data: torch.Tensor, steps: int = 2000, batch: int = 64, lr: float = 1e-3,
             align: bool = True, batch_ot: bool = True, device: str = "cpu", seed: int = 0,
             log_every: int = 200, dtype=torch.float64, model: LJ13EESI | None = None,
             entropy: str | None = None, averaging: AveragingConfig | None = None):
    """Train an LJ13EESI stochastic interpolant. Returns (model, history).

    `history` is a list of per-step tuples, `(loss_b, loss_s)` by default. `entropy`
    ("dot", "zdot" or "both") appends the matching per-batch entropy estimates as
    extra columns -- `(loss_b, loss_s, S_dot, S_zdot)` for "both" -- and prints them
    in the log line. They are computed inside `model.loss` from the draw it already
    made, so they add no network evaluations; see `EESI.loss`.

    Same caveat as the XY side: this is a progress diagnostic, watched DURING a run,
    not a pass/fail criterion for a coupling or architecture change -- it runs through
    the trained networks. The model-free arbiters stay `transport_cost` for the
    coupling and generated-sample statistics for the network.

    The estimates inherit the model's `eps`, which floors the 1/gamma in "zdot". If
    that channel looks noisy, build the model with a looser floor --
    `make_si_model(..., eps=1e-3)` -- as `EESI.entropy_estimate` documents.

    `averaging` turns on a moving-average shadow of the weights (see `eesi.averaging`);
    off by default. When on, the averaged copy is left on `model.avg` -- `None`
    otherwise -- so the return signature stays `(model, hist)` either way.
    """
    torch.manual_seed(seed)
    model = (model or make_si_model()).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    averager = make_averager(model, averaging) if averaging else None
    avg_start = avg_start_step(averaging, steps) if averaging else 0
    data = data.to(device=device, dtype=dtype)
    keys = _hist_keys(entropy)

    hist = []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch,), device=device)
        losses, a, b = si_step(model, data[idx], align=align, batch=batch_ot,
                               entropy=entropy)
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
                  f"transport {transport_cost(a, b).item():7.2f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
    # object.__setattr__, not plain assignment: nn.Module.__setattr__ would register
    # an nn.Module value as a submodule, which would put "avg.*" keys into
    # model.state_dict() and break torch.save(model.state_dict(), ...) downstream.
    object.__setattr__(model, "avg", averager.module if averager is not None else None)
    return model, hist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(REF_DATA_PATH))
    p.add_argument("--n-data", type=int, default=100_000)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=None,
                   help="seed for the random subset of the reference file "
                        "(default: follow --seed, so the whole run is reproducible "
                        "from one number; set it apart to resample the data while "
                        "holding the model init and minibatch order fixed)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-align", action="store_true", help="disable the group-OT layer")
    p.add_argument("--no-batch", action="store_true", help="disable the minibatch-OT layer")
    p.add_argument("--si", action="store_true",
                   help="train a stochastic interpolant (LJ13EESI, drift+score) "
                        "instead of plain flow matching")
    p.add_argument("--entropy", choices=("dot", "zdot", "both"), default=None,
                   help="--si only: log the per-batch entropy estimators alongside the "
                        "losses: 'dot' is -b.s with the learned score, 'zdot' is -b.z "
                        "with the exact conditional score. Free (reuses the loss's own "
                        "draw). A progress diagnostic, not a validation metric. 'zdot' "
                        "inherits the model's eps, which floors its 1/gamma; if it looks "
                        "noisy, build the model with eps~1e-3.")
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
    if a.entropy and not a.si:
        p.error("--entropy needs --si: plain flow matching trains no score field")

    data_seed = a.seed if a.data_seed is None else a.data_seed
    print(f"loading {a.n_data} random configs from {a.data} (data seed {data_seed})")
    data = load_ref_data(a.data, a.n_data, seed=data_seed)
    print(f"data {tuple(data.shape)}  align={not a.no_align}  batch={not a.no_batch}  "
          f"si={a.si}  device={a.device}")
    averaging = (AveragingConfig(kind=a.avg_kind, decay=a.avg_decay, window=a.avg_window)
                 if a.avg_kind else None)
    if a.si:
        model, hist = train_si(data, steps=a.steps, batch=a.batch, lr=a.lr,
                               align=not a.no_align, batch_ot=not a.no_batch,
                               device=a.device, seed=a.seed, entropy=a.entropy,
                               averaging=averaging)
        means = np.mean(hist[-100:], axis=0)
        cols = "  ".join(f"{_HIST_LABELS[k]} {m:.4f}"
                         for k, m in zip(_hist_keys(a.entropy), means))
        print(f"final (last 100): {cols}")
    else:
        model, hist = train(data, steps=a.steps, batch=a.batch, lr=a.lr, sigma=a.sigma,
                            align=not a.no_align, batch_ot=not a.no_batch,
                            device=a.device, seed=a.seed, averaging=averaging)
        print(f"final loss (last 100): {np.mean(hist[-100:]):.4f}")
    if a.out:
        torch.save(model.state_dict(), a.out)
        print(f"saved -> {a.out}")
        if model.avg is not None:
            torch.save(model.avg.state_dict(), a.out + ".avg")
            print(f"saved averaged weights -> {a.out}.avg")


if __name__ == "__main__":
    main()
