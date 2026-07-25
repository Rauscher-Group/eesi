"""Fixed benchmark for the XY-chain fix plan (plans/XY_FIX_PLAN.md).

One number set per phase, produced by an identical harness so the phases stay
comparable: N=10, J=2, B=256, exact-sampled target data, 700 steps at lr=1e-3,
seed 0. Both observables have closed forms at these settings, so the report is an
error, not a trend line:

    dU = -(N-1) J I1(J)/I0(J)                 = -12.5599   (from ODE samples)
    dS =  (N-1) [ln I0(J) - J I1(J)/I0(J)]    =  -5.1425   (from the interpolant)

Baselines recorded in the plan, for the same harness on `main`:

    n_layers=4, 700 steps      dU err 5.0 %   dS err 6.7 %
    n_layers=2, 700 steps      dU err 6.7 %   dS err 6.6 %

Usage:
    python benchmarks/xy_phase_bench.py                  # the fixed harness
    python benchmarks/xy_phase_bench.py --label phase-1  # tag the output line
    python benchmarks/xy_phase_bench.py --n-layers 4 --steps 3500

Bessel functions come from quadrature rather than scipy, which is not a declared
dependency of the package.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from eesi.ot import xy_ot_couple
from eesi.train.xy import load_exact_data, make_model, sample_base, train

# The harness. Model settings follow experiments/XY_chain_eqOT.ipynb, except
# n_layers, which the plan drops 4 -> 2 (Phase 0.4).
N, J, BATCH, STEPS, LR, SEED = 10, 2.0, 256, 700, 1e-3, 0
NET_KW = dict(n_neighbors=4, hidden=24, mlp_layers=4, edge_order=4,
              time_order=0, time_dim=32)      # Phase 1: raw t -> learned MLP
EESI_KW = dict(path="linear", gamma="sqrt", gamma_scale=0.2, score_div_method="exact")


def bessel_i(n: int, x: float) -> float:
    """I_n(x) = (1/pi) int_0^pi e^{x cos u} cos(n u) du."""
    u = np.linspace(0.0, np.pi, 20_001)
    return float(np.trapezoid(np.exp(x * np.cos(u)) * np.cos(n * u), u) / np.pi)


def exact_targets(N: int, J: float) -> tuple[float, float]:
    """(dU, dS) for the open chain: independent von Mises bonds (review Sec. 1)."""
    ratio = bessel_i(1, J) / bessel_i(0, J)
    return -(N - 1) * J * ratio, (N - 1) * (np.log(bessel_i(0, J)) - J * ratio)


def chain_energy(x: torch.Tensor, J: float) -> torch.Tensor:
    """Per-configuration energy of [B, N] angles: -J sum_i cos(theta_i+1 - theta_i)."""
    d = x[:, 1:] - x[:, :-1]
    return -J * torch.cos((d + np.pi) % (2 * np.pi) - np.pi).sum(-1)


def measure(model, data, *, J: float, batch: int, n_gen: int = 5_000,
            n_ent_batches: int = 100, device="cpu", seed: int = 1):
    """dU from ODE samples, dS from all three estimators. Returns a dict.

    The three dS routes disagree by more than their own scatter on an
    undertrained model, and they fail differently, so the benchmark reports all
    of them rather than picking one:

        dot   interpolant estimator, accumulator -(b.s)   -- needs net_b, net_s
        div   interpolant estimator, accumulator div(b)   -- needs net_b only
        ode   path integral of -(b.s) along the sampler   -- needs both, and a
              correct trajectory, so it compounds sampler error on top

    "dot" is the headline: it is what the notebook's N-sweep runs on, and it is
    the one that exercises net_s, where the plan expects the movement.
    """
    torch.manual_seed(seed)
    N = data.shape[1]

    x0 = sample_base(n_gen, N, device=device, dtype=data.dtype)
    gen = model.sample(x0, n_steps=50)
    dU = chain_energy(gen, J).mean().item()

    dot, div = [], []
    for _ in range(n_ent_batches):
        idx = torch.randint(0, data.shape[0], (batch,), device=device)
        b0 = sample_base(batch, N, device=device, dtype=data.dtype)
        b0, b1 = xy_ot_couple(b0, data[idx])
        dot.append(model.entropy_estimate(b1, b0, "dot").mean().item())
        div.append(model.entropy_estimate(b1, b0, "div").mean().item())

    x0 = sample_base(2_000, N, device=device, dtype=data.dtype)
    _, ent_traj, _ = model.sample(x0, n_steps=50, return_traj=True, entropy="dot")
    return {"dU": dU, "dS_dot": float(np.mean(dot)), "dS_div": float(np.mean(div)),
            "dS_ode": float(ent_traj[-1].mean().item()),
            "dS_scatter": float(np.std(dot))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="unlabelled", help="tag for the summary line")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--n-layers", type=int, default=2, help="Phase 0.4 drops 4 -> 2")
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--n-data", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    torch.set_default_dtype(torch.float64)
    dU_ex, dS_ex = exact_targets(N, J)
    print(f"harness: N={N} J={J} B={a.batch} steps={a.steps} lr={LR} seed={a.seed} "
          f"n_layers={a.n_layers} device={a.device}")
    print(f"exact:   dU = {dU_ex:+.4f}   dS = {dS_ex:+.4f}")

    data = load_exact_data(N, J, n_data=a.n_data, seed=a.seed).to(a.device)
    torch.manual_seed(a.seed)
    model = make_model(n_layers=a.n_layers, **NET_KW, **EESI_KW).to(a.device)
    n_par = sum(p_.numel() for p_ in model.net_b.parameters())

    t0 = time.perf_counter()
    model, hist = train(data, model=model, steps=a.steps, batch=a.batch, lr=LR,
                        device=a.device, seed=a.seed, log_every=a.steps // 4 or 1)
    wall = time.perf_counter() - t0

    m = measure(model, data, J=J, batch=a.batch, device=a.device)
    err = lambda got, ex: 100.0 * abs(got - ex) / abs(ex)
    lb, ls = np.mean(hist[-100:], axis=0)

    print(f"\n{'label':>12}  {'params/net':>10}  {'wall':>7}  {'loss_b':>8}  "
          f"{'loss_s':>8}  {'dU':>9}  {'err':>6}")
    print(f"{a.label:>12}  {n_par:>10}  {wall:6.1f}s  {lb:8.3f}  {ls:8.3f}  "
          f"{m['dU']:+9.4f}  {err(m['dU'], dU_ex):5.1f}%")
    print(f"\n{'dS estimator':>12}  {'value':>9}  {'err':>6}   (exact {dS_ex:+.4f}, "
          f"batch scatter {m['dS_scatter']:.3f})")
    for k in ("dot", "div", "ode"):
        print(f"{k:>12}  {m['dS_' + k]:+9.4f}  {err(m['dS_' + k], dS_ex):5.1f}%")


if __name__ == "__main__":
    main()
