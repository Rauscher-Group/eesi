"""Interpolant training for the 1D XY chain with Z2 x U(1) OT coupling (plans/EQOT_PLAN.md Phase C4).

Unlike the LJ13 side (`eesi.train.lj13`), which is plain flow matching, this trains an
`xyEESI` stochastic interpolant with both a drift and a score network -- the coupling is
orthogonal to the model, so wiring it in is one line before `model.loss`.

Usage:
    python -m eesi.train.xy --steps 2000 --batch 256 --J 1.0
    python -m eesi.train.xy --no-align            # ablation arms
    python -m eesi.train.xy --no-reflect          # U(1) only, no Z2

On entropy: `xyEESI` can estimate dS, and the chain has an exact answer
(dS/N = (N-1)/N * (-J*I1(J)/I0(J))). That is the POINT of the project, not a test of the
coupling -- it runs through the trained networks, so it mixes model quality with coupling
error, and trainability degrades exactly where the physics is interesting (large J).
The model-free check that the coupling preserves the prior marginal is
tests/test_ot.py::test_xy_marginal_preserved_over_z2_u1. Do the entropy comparison in a
notebook, as a result.

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

from ..datasets.xy import mcxy
from ..interpolant import xyEESI
from ..models.xygnn import XYChainGNN
from ..ot import xy_ot_couple, xy_transport_cost


def sample_base(B: int, N: int, device="cpu", dtype=torch.float64, generator=None):
    """Prior p0: i.i.d. uniform on (-pi, pi]. Invariant under Z2 x U(1) -- and under
    S(N) too, but S(N) is NOT a symmetry of the chain's energy, so it is not in the
    coupling group. See eesi/ot.py."""
    u = torch.rand(B, N, device=device, dtype=dtype, generator=generator)
    return u * 2.0 * np.pi - np.pi


def make_model(n_neighbors: int = 2, hidden: int = 32, n_layers: int = 6,
               edge_order: int = 16, time_order: int = 16, path: str = "trig",
               gamma: str = "sqrt", gamma_scale: float = 0.5, **kw) -> xyEESI:
    """An xyEESI with two independent XYChainGNNs. The nets are chain-length
    independent: N is inferred per forward, so it is not a constructor argument."""
    net_kw = dict(n_neighbors=n_neighbors, hidden=hidden, n_layers=n_layers,
                  edge_order=edge_order, time_order=time_order)
    return xyEESI(XYChainGNN(**net_kw), XYChainGNN(**net_kw),
                  path=path, gamma=gamma, gamma_scale=gamma_scale, **kw)


def xy_step(model: xyEESI, x1: torch.Tensor, align: bool = True, batch: bool = True,
            reflect: bool = True, generator=None):
    """One coupled training step's losses. Returns (losses, x0, x1).

    The coupling runs under no_grad inside `xy_ot_couple`; `model.loss` already takes
    both endpoints, so no API change is needed to insert it.
    """
    B, N = x1.shape
    x0 = sample_base(B, N, device=x1.device, dtype=x1.dtype, generator=generator)
    x0, x1 = xy_ot_couple(x0, x1, align=align, batch=batch, reflect=reflect)
    return model.loss(x1, x0), x0, x1


def load_mc_data(N: int, J: float, n_save: int = 4000, n_eq: int = 50_000,
                 n_prod: int = 400_000, dtype=torch.float64) -> torch.Tensor:
    """Boltzmann samples from the classical-XY Monte Carlo sampler, wrapped to (-pi, pi]."""
    confs, _ = mcxy(N=N, J=J, n_eq=n_eq, n_prod=n_prod, n_save=n_save)
    return torch.as_tensor((confs + np.pi) % (2 * np.pi) - np.pi, dtype=dtype)


def train(data: torch.Tensor, steps: int = 2000, batch: int = 256, lr: float = 1e-3,
          align: bool = True, batch_ot: bool = True, reflect: bool = True,
          device: str = "cpu", seed: int = 0, log_every: int = 200,
          dtype=torch.float64, model: xyEESI | None = None):
    """Train an xyEESI on XY-chain data. Returns (model, history)."""
    torch.manual_seed(seed)
    model = (model or make_model()).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(device=device, dtype=dtype)

    hist = []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch,), device=device)
        losses, a, b = xy_step(model, data[idx], align=align, batch=batch_ot,
                               reflect=reflect)
        loss = losses["b"] + losses["s"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append((losses["b"].item(), losses["s"].item()))

        if log_every and (step % log_every == 0 or step == steps - 1):
            lb, ls = np.mean(hist[-log_every:], axis=0)
            print(f"  step {step:5d}  loss_b {lb:9.4f}  loss_s {ls:9.4f}  "
                  f"transport {xy_transport_cost(a, b).item():7.3f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
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
    p.add_argument("--no-reflect", action="store_true", help="U(1) only, drop the Z2")
    p.add_argument("--out", default=None, help="path to save the state_dict")
    a = p.parse_args()

    print(f"sampling XY chain: N={a.N} J={a.J}")
    data = load_mc_data(a.N, a.J, n_save=a.n_data)
    print(f"data {tuple(data.shape)}  align={not a.no_align}  batch={not a.no_batch}  "
          f"reflect={not a.no_reflect}  device={a.device}")
    model, hist = train(data, steps=a.steps, batch=a.batch, lr=a.lr,
                        align=not a.no_align, batch_ot=not a.no_batch,
                        reflect=not a.no_reflect, device=a.device, seed=a.seed)
    lb, ls = np.mean(hist[-100:], axis=0)
    print(f"final (last 100): loss_b {lb:.4f}  loss_s {ls:.4f}")
    if a.out:
        torch.save(model.state_dict(), a.out)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
