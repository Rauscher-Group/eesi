"""Benchmark: what fraction of an LJ13 training step is the OT coupling? (plans/EQOT_PLAN.md Phase B5)

Three questions, three sections:

  step       coupling time vs EGNN forward+backward at B in {64, 128, 256, 512}.
  breakdown  where inside the coupling that time actually goes.
  overlap    inline coupling vs SemlaFlow's approach -- coupling on CPU in dataloader
             workers, overlapped with GPU compute. The plan calls this "a comparison
             point, not a contingency: keep whichever wins."

Usage:
    python benchmarks/bench_ot.py                      # all three, on cuda if present
    python benchmarks/bench_ot.py --sections step
    python benchmarks/bench_ot.py --device cpu --batches 64 128

Timing is data-independent to within noise (the Hungarian's iteration count depends
mildly on cost structure, nothing else does), so this runs on prior samples for both
endpoints unless --data points at the OSF MCMC file.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Callable

import numpy as np
import torch

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from eesi.datasets.lj13 import sample_prior
from eesi.models.lj13_dynamics import LJ13Dynamics
from eesi.ot import (_hungarian_nd, _outer_assignment, _svdvals_3x3, center,
                     equivariant_ot_couple)

BATCHES = (64, 128, 256, 512)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timeit(fn: Callable[[], object], device: str, n: int = 10, warmup: int = 3) -> float:
    """Median-free mean wall time per call, in ms. Warms up first (cuSOLVER, autotune)."""
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n * 1e3


def make_endpoints(B: int, data: torch.Tensor | None, device: str, dtype):
    """(x0, x1), both centred. x1 from real data when given, else from the prior."""
    x0 = sample_prior(B, dtype=dtype, device=device)
    if data is None:
        x1 = sample_prior(B, dtype=dtype, device=device)
    else:
        idx = torch.randint(0, data.shape[0], (B,))
        x1 = data[idx].to(device=device, dtype=dtype)
    return center(x0), center(x1)


# ---- section 1: coupling vs the network step -------------------------------


def bench_step(data, device: str, dtype, batches, n: int):
    """The headline number: coupling as a fraction of total step time."""
    print("\n=== step: coupling vs EGNN forward+backward ===")
    print(f"{'B':>5}  {'couple ms':>10}  {'net f+b ms':>11}  {'step ms':>9}  {'coupling':>9}")
    net = LJ13Dynamics().to(device=device, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rows = []
    for B in batches:
        x0, x1 = make_endpoints(B, data, device, dtype)
        t = torch.rand(B, device=device, dtype=dtype)

        def couple():
            return equivariant_ot_couple(x0, x1)

        def netstep():
            # forward + backward + optimizer, i.e. everything the coupling competes with
            v = net(t, x1)
            loss = ((v - x1) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        c_ms = timeit(couple, device, n)
        n_ms = timeit(netstep, device, n)
        frac = c_ms / (c_ms + n_ms)
        rows.append((B, c_ms, n_ms, frac))
        print(f"{B:>5}  {c_ms:>10.1f}  {n_ms:>11.2f}  {c_ms+n_ms:>9.1f}  {frac:>8.1%}")
    return rows


# ---- section 2: inside the coupling ----------------------------------------


def bench_breakdown(data, device: str, dtype, batches, n: int):
    """Where the coupling's time goes. The answer is not the Hungarian.

    `eigvals` is the live singular-value path (`ot._svdvals_3x3`); `svdvals` is the
    rejected alternative it replaced, timed alongside to keep the ~20x gap honest and
    to catch a future torch release closing it. See plans/EQOT_PLAN.md Phase B5.
    """
    print("\n=== breakdown: inside lj_cost_matrix ===")
    print(f"{'B':>5}  {'D build':>8}  {'hungarian':>10}  {'gather':>7}  "
          f"{'eigvals':>8}  {'det':>6}  {'outer LSA':>10}  | {'svdvals (rejected)':>18}")
    for B in batches:
        x0, x1 = make_endpoints(B, data, device, dtype)
        N = x0.shape[1]
        sq0, sq1 = (x0 ** 2).sum(-1), (x1 ** 2).sum(-1)

        def build_D():
            inner = torch.einsum('iad,jbd->ijab', x0, x1)
            return (sq0[:, None, :, None] + sq1[None, :, None, :] - 2 * inner).clamp_min(0)

        D = build_D()
        cols = _hungarian_nd(D.reshape(B * B, N, N))
        perm = torch.argsort(cols, dim=-1).reshape(B, B, N)
        idx = perm.reshape(B, B, N, 1).expand(B, B, N, 3)

        def gather():
            return x0[:, None].expand(B, B, N, 3).gather(2, idx)

        x0p = gather()
        H = torch.einsum('ijna,jnb->ijab', x0p, x1)
        M = sq0.sum(-1)[:, None] + sq1.sum(-1)[None, :]

        d_ms = timeit(build_D, device, n)
        h_ms = timeit(lambda: _hungarian_nd(D.reshape(B * B, N, N)), device, n)
        g_ms = timeit(gather, device, n)
        e_ms = timeit(lambda: _svdvals_3x3(H), device, n)
        det_ms = timeit(lambda: torch.linalg.det(H), device, n)
        o_ms = timeit(lambda: _outer_assignment(M), device, n)
        s_ms = timeit(lambda: torch.linalg.svdvals(H), device, n)
        print(f"{B:>5}  {d_ms:>8.2f}  {h_ms:>10.2f}  {g_ms:>7.2f}  {e_ms:>8.2f}  "
              f"{det_ms:>6.2f}  {o_ms:>10.2f}  | {s_ms:>18.1f}")


# ---- section 3: the dataloader-overlap comparison point ---------------------


class CoupledBatches(torch.utils.data.Dataset):
    """One item = one CPU-coupled batch. SemlaFlow's pattern: the OT runs in the worker.

    Map-style with index = step number; each __getitem__ draws its own batch, so with
    num_workers > 0 the coupling of step k+1 overlaps the GPU work of step k.
    """

    def __init__(self, data: torch.Tensor | None, B: int, steps: int, dtype):
        self.data, self.B, self.steps, self.dtype = data, B, steps, dtype

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        x0, x1 = make_endpoints(self.B, self.data, "cpu", self.dtype)
        return equivariant_ot_couple(x0, x1)


def bench_overlap(data, device: str, dtype, batches, steps: int, workers: int):
    """Inline GPU coupling vs CPU coupling hidden in dataloader workers.

    Reports end-to-end steps/sec for each. Overlap can only ever hide the coupling if
    one worker couples a batch faster than the GPU consumes `workers` of them.
    """
    print(f"\n=== overlap: inline GPU coupling vs {workers} CPU dataloader workers ===")
    print(f"{'B':>5}  {'inline s/step':>13}  {'workers s/step':>14}  {'winner':>8}")
    net = LJ13Dynamics().to(device=device, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def train_on(x0, x1):
        t = torch.rand(x1.shape[0], device=device, dtype=dtype)
        mu = x0 * (1 - t.view(-1, 1, 1)) + x1 * t.view(-1, 1, 1)
        loss = ((net(t, mu) - (x1 - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for B in batches:
        # warm up cuSOLVER, the autograd graph and Adam's state before either arm is
        # timed -- otherwise the first B measured absorbs all of it and reads slow.
        for _ in range(3):
            x0, x1 = make_endpoints(B, data, device, dtype)
            train_on(*equivariant_ot_couple(x0, x1))

        # inline: couple on the training device, in the training loop
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            x0, x1 = make_endpoints(B, data, device, dtype)
            train_on(*equivariant_ot_couple(x0, x1))
        _sync(device)
        inline = (time.perf_counter() - t0) / steps

        # workers: couple on CPU in parallel processes, overlapped with the GPU step
        dl = torch.utils.data.DataLoader(
            CoupledBatches(data, B, steps, dtype), batch_size=None,
            num_workers=workers, persistent_workers=False, prefetch_factor=2)
        it = iter(dl)
        x0, x1 = next(it)                       # prime: pay the worker spin-up once
        train_on(x0.to(device), x1.to(device))
        _sync(device)
        t0 = time.perf_counter()
        k = 0
        for x0, x1 in it:
            train_on(x0.to(device, non_blocking=True), x1.to(device, non_blocking=True))
            k += 1
        _sync(device)
        overlapped = (time.perf_counter() - t0) / max(k, 1)
        del it, dl

        win = "inline" if inline < overlapped else "workers"
        print(f"{B:>5}  {inline:>13.4f}  {overlapped:>14.4f}  {win:>8}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    p.add_argument("--repeats", type=int, default=10, help="timed calls per measurement")
    p.add_argument("--steps", type=int, default=20, help="steps per overlap arm")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--data", default=None,
                   help="OSF MCMC .npy; omit to use prior samples for both endpoints")
    p.add_argument("--sections", nargs="+", default=["step", "breakdown", "overlap"],
                   choices=["step", "breakdown", "overlap"])
    a = p.parse_args()

    dtype = getattr(torch, a.dtype)
    torch.manual_seed(0)

    data = None
    if a.data:
        raw = np.asarray(np.load(a.data, mmap_mode="r")[:20_000]).astype(np.float64)
        data = torch.from_numpy(raw).view(-1, 13, 3).to(dtype)
        data = data - data.mean(1, keepdim=True)

    print(f"device={a.device}  dtype={a.dtype}  "
          f"x1={'real MCMC' if data is not None else 'prior samples'}")
    if a.device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    if "step" in a.sections:
        bench_step(data, a.device, dtype, a.batches, a.repeats)
    if "breakdown" in a.sections:
        bench_breakdown(data, a.device, dtype, a.batches, a.repeats)
    if "overlap" in a.sections:
        bench_overlap(data, a.device, dtype, a.batches, a.steps, a.workers)


if __name__ == "__main__":
    main()
