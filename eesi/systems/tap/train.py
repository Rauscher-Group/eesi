"""Stochastic-interpolant training for TAP with equivariant-OT coupling.

Trains a `TAPEESI` -- a drift AND a score field, a latent noise schedule, and
interpolant-based entropy -- on reference configurations of a tangentially active
polymer. Data-driven: no energy function anywhere, and unlike LJ13 there could not be
one (see the `eesi.systems.tap.data` docstring).

Usage:
    python -m eesi.systems.tap.train --data data/tap_N20.npy --k 100 --b 1.0 --steps 2000
    python -m eesi.systems.tap.train --k 100 --b 1.0 --no-align --no-batch  # ablation arms
    python -m eesi.systems.tap.train --k 100 --b 1.0 --no-index-feature     # S(N) net ablation
    python -m eesi.systems.tap.train --k 1.287 --b 0.0                      # ideal-chain prior

`--k` and `--b` are the prior's bond spring constant and equilibrium length; set them
from the polymer being sampled, not from the chain's observed size (`--b 0` recovers the
old ideal chain, and `eesi.systems.tap.data.end_to_end_mean_sq` says what E[Re^2] a given
pair implies, if you want to check the prior against the data before training).

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
from .data import (N_DEFAULT, N_DIMS, REF_DATA_PATH, end_to_end_mean_sq, end_to_end_sq,
                   load_ref_data, sample_prior)
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


def tap_step(model: TAPEESI, x1: torch.Tensor, k: float, b: float, align: bool = True,
             batch: bool = True, generator=None):
    """One coupled SI training step's losses. Returns (losses, x0, x1).

    Mirrors `eesi.systems.lj13.train.si_step`: sample a harmonic-bond base, OT-couple it
    to the data, then hand both endpoints to `model.loss`. The coupling runs under
    no_grad, so no gradient reaches the pairing.

    `k` and `b` are the prior's bond parameters, passed straight through to
    `eesi.systems.tap.data.sample_prior` -- see its module docstring for the law.
    """
    B, N, D = x1.shape
    x0 = sample_prior(B, k, b, n_particles=N, n_dims=D, dtype=x1.dtype,
                      device=x1.device, generator=generator)
    x0, x1 = tap_ot_couple(x0, x1, align=align, batch=batch)
    return model.loss(x1, x0), x0, x1


def train_si(data: torch.Tensor, k: float, b: float, steps: int = 2000, batch: int = 64,
             lr: float = 1e-3, align: bool = True, batch_ot: bool = True,
             device: str = "cpu", seed: int = 0, log_every: int = 200,
             dtype=torch.float64, model: TAPEESI | None = None):
    """Train a TAPEESI stochastic interpolant. Returns (model, history).

    `k`, `b` are the prior's bond spring constant and equilibrium length.
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
        losses, x0, x1 = tap_step(model, data[idx], k, b, align=align, batch=batch_ot)
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
    print(f"data {tuple(data.shape)}  k={a.k} b={a.b}  "
          f"prior E[Re^2]={end_to_end_mean_sq(a.k, a.b, data.shape[1]):.4f} "
          f"(data {end_to_end_sq(data).mean().item():.4f})  align={not a.no_align}  "
          f"batch={not a.no_batch}  index_feature={not a.no_index_feature}  "
          f"device={a.device}")
    model = make_si_model(n_particles=data.shape[1], n_dims=data.shape[2],
                          index_feature=not a.no_index_feature)
    model, hist = train_si(data, a.k, a.b, steps=a.steps, batch=a.batch, lr=a.lr,
                           align=not a.no_align, batch_ot=not a.no_batch,
                           device=a.device, seed=a.seed, model=model)
    lb, ls = np.mean(hist[-100:], axis=0)
    print(f"final (last 100): loss_b {lb:.4f}  loss_s {ls:.4f}")
    if a.out:
        torch.save(model.state_dict(), a.out)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
