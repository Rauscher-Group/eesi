"""Stochastic-interpolant training for TAP with equivariant-OT coupling.

Trains a `TAPEESI` -- a drift AND a score field, a latent noise schedule, and
interpolant-based entropy -- on reference configurations of a tangentially active
polymer. Data-driven: no energy function anywhere, and unlike LJ13 there could not be
one (see the `eesi.systems.tap.data` docstring).

Usage:
    python -m eesi.systems.tap.train --data data/tap_N20.npy \
        --k 91.5 --b 1.024 --gamma 2.33 --cos-theta-0 0.50 --steps 2000
    python -m eesi.systems.tap.train --k 91.5 --b 1.024 --gamma 2.33 --cos-theta-0 0.50 \
        --no-align --no-batch                                  # ablation arms
    python -m eesi.systems.tap.train --k 91.5 --b 1.024 --gamma 0 --cos-theta-0 1 \
                                                               # freely-jointed prior

The four prior parameters are properties of the polymer, not of the chain's observed
size, and the recipe for them is:

    --k            1 / var(Q) from the reference data's bond lengths
    --b            E[Q] from the same
    --gamma        1 / (2 var(cos theta)) from its bond angles
    --cos-theta-0  `solve_cos_theta_0(k, b, gamma, target, N)`, i.e. tuned so the
                   prior's E[Re^2] matches the data's

Only the last is fitted rather than measured, and deliberately so: the reference chain's
angular correlations decay slower than the geometric law any nearest-neighbour bending
potential can produce, so matching the observed <cos theta> would leave the chain too
small. cos_theta_0 buys the correct global size instead. `--gamma 0` recovers the
freely-jointed harmonic-bond prior, and `--gamma 0 --b 0` the original ideal chain.
`end_to_end_mean_sq` reports what any four values imply before you train on them; the
startup line below prints it against the data.

There is no plain flow-matching path here, unlike `eesi.systems.lj13.train`. LJ13
carries one only because it must reproduce the convention of a released checkpoint;
TAP has no such constraint, so the interpolant is the only entry point.

`--no-align` / `--no-batch` are the 2x2 ablation: `align` is OT over O(3), `batch` is
OT over the minibatch. Expect the balance to sit closer to the XY chain than to LJ13 --
with no permutation to optimize over, the group is only 3-dimensional, so `batch`
should do most of the work.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ...ot import transport_cost
from .data import (N_DEFAULT, N_DIMS, REF_DATA_PATH, angle_moments, bond_cosines,
                   end_to_end_mean_sq, end_to_end_sq, load_ref_data, sample_prior)
from .dynamics import TAPDynamics
from .interpolant import TAPEESI
from .ot import tap_ot_couple


def make_si_model(n_particles: int = N_DEFAULT, n_dims: int = N_DIMS, hidden_nf: int = 32,
                  n_layers: int = 3, index_feature: bool = True, path: str = "linear",
                  gamma: str = "quad", gamma_scale: float = 1.0, **kw) -> TAPEESI:
    """A TAPEESI wrapping two independent TAPDynamics fields (drift + score)."""
    net_kw = dict(n_particles=n_particles, n_dims=n_dims, index_feature=index_feature,
                  hidden_nf=hidden_nf, n_layers=n_layers)
    net_b = TAPDynamics(**net_kw)
    net_s = TAPDynamics(**net_kw)
    return TAPEESI(net_b, net_s, d=n_particles, path=path, gamma=gamma,
                   gamma_scale=gamma_scale, **kw)


def tap_step(model: TAPEESI, x1: torch.Tensor, k: float, b: float, gamma: float,
             cos_theta_0: float, align: bool = True, batch: bool = True, generator=None):
    """One coupled SI training step's losses. Returns (losses, x0, x1).

    Mirrors `eesi.systems.lj13.train.si_step`: sample a semiflexible base, OT-couple it
    to the data, then hand both endpoints to `model.loss`. The coupling runs under
    no_grad, so no gradient reaches the pairing.

    `k`, `b`, `gamma`, `cos_theta_0` are the prior's four parameters, passed straight
    through to `eesi.systems.tap.data.sample_prior` -- see its module docstring.

    Note that `gamma` here is the prior's BENDING constant, unrelated to the `gamma`
    argument of `make_si_model` above, which names the interpolant's latent noise
    schedule ("quad", "sqrt", ...). The two never meet, but the collision is easy to
    misread.
    """
    B, N, D = x1.shape
    x0 = sample_prior(B, k, b, gamma, cos_theta_0, n_particles=N, n_dims=D,
                      dtype=x1.dtype, device=x1.device, generator=generator)
    x0, x1 = tap_ot_couple(x0, x1, align=align, batch=batch)
    return model.loss(x1, x0), x0, x1


def train_si(data: torch.Tensor, k: float, b: float, gamma: float, cos_theta_0: float,
             steps: int = 2000, batch: int = 64, lr: float = 1e-3, align: bool = True,
             batch_ot: bool = True, device: str = "cpu", seed: int = 0,
             log_every: int = 200, dtype=torch.float64, model: TAPEESI | None = None):
    """Train a TAPEESI stochastic interpolant. Returns (model, history).

    `k`, `b` are the prior's bond spring constant and equilibrium length; `gamma` and
    `cos_theta_0` its bending constant and equilibrium bond-angle cosine (NOT the
    interpolant's `gamma` schedule -- see `tap_step`).
    """
    torch.manual_seed(seed)
    if model is None:
        model = make_si_model(n_particles=data.shape[1], n_dims=data.shape[2])
    model = model.to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(device=device, dtype=dtype)

    hist = []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch,), device=device)
        # x0/x1, not a/b: `b` is the prior's equilibrium length in this scope, and
        # rebinding it here would feed a tensor back into the next step's sampler.
        losses, x0, x1 = tap_step(model, data[idx], k, b, gamma, cos_theta_0,
                                  align=align, batch=batch_ot)
        loss = losses["b"] + losses["s"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append((losses["b"].item(), losses["s"].item()))

        if log_every and (step % log_every == 0 or step == steps - 1):
            lb, ls = np.mean(hist[-log_every:], axis=0)
            print(f"  step {step:5d}  loss_b {lb:9.4f}  loss_s {ls:9.4f}  "
                  f"transport {transport_cost(x0, x1).item():7.3f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
    return model, hist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(REF_DATA_PATH))
    p.add_argument("--k", type=float, required=True,
                   help="the prior's bond spring constant")
    p.add_argument("--b", type=float, required=True,
                   help="the prior's bond equilibrium length; 0 gives the ideal chain")
    p.add_argument("--gamma", type=float, required=True,
                   help="the prior's bending constant; 0 gives a freely-jointed chain")
    p.add_argument("--cos-theta-0", type=float, required=True,
                   help="cosine of the prior's equilibrium bond angle, in [-1, 1]; "
                        "tune it to match the data's E[Re^2] (see solve_cos_theta_0)")
    p.add_argument("--n-particles", type=int, default=N_DEFAULT)
    p.add_argument("--n-data", type=int, default=100_000)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-align", action="store_true", help="disable the O(3) OT layer")
    p.add_argument("--no-batch", action="store_true", help="disable the minibatch-OT layer")
    p.add_argument("--no-index-feature", action="store_true",
                   help="drop the chain-index node feature, restoring the S(N)-equivariant "
                        "LJ13 net (ablation: it cannot represent a directed chain)")
    p.add_argument("--out", default=None, help="path to save the state_dict")
    a = p.parse_args()

    print(f"loading {a.n_data} configs from {a.data}")
    data = load_ref_data(a.data, a.n_data, n_particles=a.n_particles)
    # Both structural summaries the prior can get wrong, side by side with the data's
    # own values: a prior that misses these is one the flow has to transport further.
    print(f"data {tuple(data.shape)}  k={a.k} b={a.b} gamma={a.gamma} "
          f"cos_theta_0={a.cos_theta_0}\n"
          f"  prior E[Re^2]={end_to_end_mean_sq(a.k, a.b, a.gamma, a.cos_theta_0, data.shape[1]):.4f} "
          f"(data {end_to_end_sq(data).mean().item():.4f})   "
          f"prior <cos theta>={angle_moments(a.gamma, a.cos_theta_0)[1]:+.4f} "
          f"(data {bond_cosines(data).mean().item():+.4f})")
    print(f"  align={not a.no_align}  batch={not a.no_batch}  "
          f"index_feature={not a.no_index_feature}  device={a.device}")
    model = make_si_model(n_particles=data.shape[1], n_dims=data.shape[2],
                          index_feature=not a.no_index_feature)
    model, hist = train_si(data, a.k, a.b, a.gamma, a.cos_theta_0,
                           steps=a.steps, batch=a.batch, lr=a.lr,
                           align=not a.no_align, batch_ot=not a.no_batch,
                           device=a.device, seed=a.seed, model=model)
    lb, ls = np.mean(hist[-100:], axis=0)
    print(f"final (last 100): loss_b {lb:.4f}  loss_s {ls:.4f}")
    if a.out:
        torch.save(model.state_dict(), a.out)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
