"""Interpolant training for the toy systems, with minibatch-OT coupling.

The loop the two notebooks in `experiments/Toy/` used to define inline. Same shape as
`eesi.systems.xy.train` -- draw the base, couple it to the data, hand both endpoints to
`model.loss` -- with the simpler coupling of `eesi.systems.toy.ot`: no symmetry group,
so the only ablation flag is `batch_ot`.

Usage:
    python -m eesi.systems.toy.train --d 40 --n-mixes 16 --steps 20000
    python -m eesi.systems.toy.train --no-batch                # ablation: independent coupling
    python -m eesi.systems.toy.train --n-data 4000             # fixed dataset, "limited data"
    python -m eesi.systems.toy.train --no-vel                  # score-only, as in 40D_GMM.ipynb

`train` takes either form of target the notebooks use:

    a torch.Tensor (n_data, d)  a dataset drawn once and minibatched with replacement,
                                the regime where p1 is expensive to sample;
    anything with .sample((B,)) a `GaussianMixture` or a `torch.distributions`
                                object, redrawn every step.

Note the dtype default is float32 here, not the float64 of `eesi.systems.xy.train` and
`eesi.systems.lj13.train`: the toy target's buffers are float32 and `GaussianMixture.to`
ignores dtype entirely, so float64 would need casts at every draw for no benefit -- these
are 2-to-40-dimensional demos, not the numerically delicate physics runs.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ...interpolant import EESI
from .data import GaussianMixture
from .mlp import TimeMLP
from .ot import toy_ot_couple, toy_transport_cost


def sample_base(B: int, d: int, device="cpu", dtype=torch.float32, generator=None):
    """Prior p0: the standard normal N(0, I) in R^d. Returns (B, d).

    Rotationally invariant, but the target is not, so the coupling does not exploit it
    -- see `eesi.systems.toy.ot`.
    """
    return torch.randn(B, d, device=device, dtype=dtype, generator=generator)


def make_model(d: int, hidden: int = 256, hidden_s: int | None = None, n_layers: int = 5,
               activation: str = "gelu", path: str = "linear", gamma: str = "sqrt",
               gamma_scale: float = 0.2, **kw) -> EESI:
    """An EESI with two independent TimeMLPs (drift + score).

    No toy subclass is needed: the base `EESI` reads every shape off the incoming
    tensors, and R^d needs no geometry-aware interpolant. `hidden_s` defaults to
    `hidden`; the score field is the harder of the two to fit, so widening it alone is
    the usual first move (256/384 in `experiments/Toy/40D_GMM.ipynb`).
    """
    return EESI(TimeMLP(d=d, hidden=hidden, n_layers=n_layers, activation=activation),
                TimeMLP(d=d, hidden=hidden_s or hidden, n_layers=n_layers,
                        activation=activation),
                d=d, path=path, gamma=gamma, gamma_scale=gamma_scale, **kw)


def toy_step(model: EESI, x1: torch.Tensor, batch_ot: bool = True, generator=None):
    """One coupled training step's losses. Returns (losses, x0, x1).

    The coupling runs under no_grad inside `toy_ot_couple`; `model.loss` already takes
    both endpoints, so no API change is needed to insert it.
    """
    B, d = x1.shape
    x0 = sample_base(B, d, device=x1.device, dtype=x1.dtype, generator=generator)
    x0, x1 = toy_ot_couple(x0, x1, batch=batch_ot)
    return model.loss(x1, x0), x0, x1


def _target_sampler(target, batch: int, device, dtype):
    """Resolve either target form into a `draw() -> (batch, d)` closure and its d.

    A tensor is a fixed dataset, minibatched with replacement (as in
    `eesi.systems.xy.train.train`); anything else is asked for fresh samples each call.
    """
    if torch.is_tensor(target):
        data = target.to(device=device, dtype=dtype)
        if data.dim() != 2:
            raise ValueError(f"tensor target must be (n_data, d); got {tuple(data.shape)}")
        n = data.shape[0]

        def draw():
            return data[torch.randint(0, n, (batch,), device=device)]

        return draw, data.shape[1]

    if not hasattr(target, "sample"):
        raise TypeError("target must be a (n_data, d) tensor or expose .sample((B,)); "
                        f"got {type(target).__name__}")

    def draw():
        return target.sample((batch,)).to(device=device, dtype=dtype)

    # `GaussianMixture.dim` and a Distribution's `event_shape` both state d without
    # drawing anything; the throwaway sample is the last resort, and it perturbs both
    # the RNG stream and `GaussianMixture.call_time`.
    d = getattr(target, "dim", None)
    if not isinstance(d, int):
        shape = getattr(target, "event_shape", None)
        d = int(shape[0]) if shape else int(target.sample((1,)).shape[-1])
    return draw, d


def train(target, steps: int = 2000, batch: int = 1000, lr: float = 1e-4,
          batch_ot: bool = True, learn_vel: bool = True, learn_score: bool = True,
          device: str = "cpu", seed: int = 0, log_every: int = 200,
          dtype=torch.float32, model: EESI | None = None):
    """Train an EESI on a toy target. Returns (model, history).

    `target` is a (n_data, d) tensor or anything exposing `.sample((B,))`; see the
    module docstring. `history` is a list of (loss_b, loss_s) pairs, one per step.

    `learn_vel` / `learn_score` select which fields are trained -- dropping the drift
    is how `experiments/Toy/40D_GMM.ipynb` fits the score alone. At least one must be
    on; the untrained net still reports its loss in the history, it just gets no
    gradient.
    """
    if not (learn_vel or learn_score):
        raise ValueError("learn_vel and learn_score cannot both be False: no loss to train")
    torch.manual_seed(seed)
    draw_x1, d = _target_sampler(target, batch, device, dtype)
    model = (model or make_model(d)).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = []
    t0 = time.perf_counter()
    for step in range(steps):
        losses, a, b = toy_step(model, draw_x1(), batch_ot=batch_ot)
        loss = losses["b"].new_zeros(())
        if learn_vel:
            loss = loss + losses["b"]
        if learn_score:
            loss = loss + losses["s"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append((losses["b"].item(), losses["s"].item()))

        if log_every and (step % log_every == 0 or step == steps - 1):
            lb, ls = np.mean(hist[-log_every:], axis=0)
            print(f"  step {step:5d}  loss_b {lb:9.4f}  loss_s {ls:9.4f}  "
                  f"transport {toy_transport_cost(a, b).item():7.3f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
    return model, hist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=40, help="dimension of the toy problem")
    p.add_argument("--n-mixes", type=int, default=16, help="Gaussian-mixture components")
    p.add_argument("--loc-scaling", type=float, default=2.0,
                   help="spread of the component means; larger = more separated modes")
    p.add_argument("--log-var-scaling", type=float, default=-3.0,
                   help="component width (pre-softplus log variance)")
    p.add_argument("--n-data", type=int, default=0,
                   help="draw a fixed dataset of this size once instead of resampling "
                        "the target every step (0 = resample, the default)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--no-batch", action="store_true", help="disable the minibatch-OT layer")
    p.add_argument("--no-vel", action="store_true", help="do not train the drift field")
    p.add_argument("--no-score", action="store_true", help="do not train the score field")
    p.add_argument("--out", default=None, help="path to save the state_dict")
    a = p.parse_args()

    print(f"target: {a.n_mixes}-component mixture in R^{a.d}  (seed {a.seed})")
    target = GaussianMixture(dim=a.d, n_mixes=a.n_mixes, loc_scaling=a.loc_scaling,
                             log_var_scaling=a.log_var_scaling, seed=a.seed,
                             device=a.device)
    if a.n_data:
        target = target.sample((a.n_data,))
        print(f"data {tuple(target.shape)}  (fixed dataset)")
    print(f"batch_ot={not a.no_batch}  vel={not a.no_vel}  score={not a.no_score}  "
          f"device={a.device}")
    model, hist = train(target, steps=a.steps, batch=a.batch, lr=a.lr,
                        batch_ot=not a.no_batch, learn_vel=not a.no_vel,
                        learn_score=not a.no_score, device=a.device, seed=a.seed,
                        log_every=a.log_every)
    lb, ls = np.mean(hist[-100:], axis=0)
    print(f"final (last 100): loss_b {lb:.4f}  loss_s {ls:.4f}")
    if a.out:
        torch.save(model.state_dict(), a.out)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
