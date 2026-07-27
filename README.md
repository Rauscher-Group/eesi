# eesi

**Entropy Estimation by Stochastic Interpolants** — general stochastic interpolants
on Euclidean space R^d. The `EESI` model operates on `[B, d]` vectors with a
time-conditioned MLP backbone. Two physical systems are built out: the LJ13 cluster
(`[B, 13, 3]`, Satorras E(n)-GNN velocity field) and the 1D classical XY chain
(`xyEESI` on the periodic angle manifold), each with a symmetry-aware OT coupling.

## Layout

```
.
├── pyproject.toml
├── README.md
├── eesi/                       # source only
│   ├── __init__.py
│   ├── interpolant.py          # EESI: loss, samplers, entropy estimators
│   ├── ot.py                   # group-agnostic OT helpers (centering, Hungarian)
│   └── systems/                # one self-contained subpackage per system
│       ├── toy/                # the 1D/2D pedagogical examples
│       │   ├── data.py         # GaussianMixture base distribution
│       │   └── mlp.py          # TimeMLP: time-conditioned MLP field on R^d
│       ├── xy/                 # the 1D XY chain
│       │   ├── data.py         # 1D classical XY Monte Carlo (mcxy), exact sampler
│       │   ├── gnn.py          # XYChainGNN: static-graph net for the chain
│       │   ├── interpolant.py  # xyEESI: geodesic interpolant on (S^1)^N
│       │   ├── ot.py           # equivariant OT over O(2) x Z2^site
│       │   └── train.py        # python -m eesi.systems.xy.train
│       └── lj13/               # the LJ13 cluster
│           ├── data.py         # reference data, prior, energies, subspace geometry
│           ├── dynamics.py     # LJ13Dynamics + rk4_sample, divergence, free_energy
│           ├── interpolant.py  # LJ13EESI: interpolant on the COM-free subspace
│           ├── ot.py           # equivariant OT over S(N) x SO(3)
│           ├── train.py        # python -m eesi.systems.lj13.train
│           └── LJ13_eq_OT_flow_matching   # the released OSF checkpoint (gitignored)
├── experiments/                # notebooks -- where the physics gets done
├── benchmarks/                 # performance measurement (bench_ot.py)
├── tests/                      # correctness, mirroring the source layout
│   ├── core/                   # + ot_reference.py, the scipy OT oracle
│   ├── toy/  xy/  lj13/
├── data/                       # large third-party downloads (gitignored) -- see data/README.md
└── plans/                      # design docs
```

The package is grouped by **system**, not by layer: everything the XY chain needs
sits in `eesi/systems/xy/`, everything LJ13 needs in `eesi/systems/lj13/`, and the
two never import from each other. What is left at the top level is the part that is
genuinely general — the `EESI` interpolant and the group-agnostic OT helpers. The
dependency rule is that a system may import from the core and from its own siblings,
never the reverse.

Within a system, the line between `data.py` and the model modules is whether a symbol
needs a trained field to mean anything. `lj13/data.py` holds closed-form facts — the
reference data, the prior (`sample_prior`, `log_prior`), the energies, the geometry of
the COM-free subspace (`DOF`, `subspace_dirs`) — all true whether or not a model was
ever fit. `lj13/dynamics.py` holds the flow and everything you get by running it:
`rk4_sample`, `divergence`, `integrate_with_logdet`, `free_energy`. The dependency
runs one way, `dynamics → data`.

`tests/` checks correctness; `benchmarks/` measures speed. They are separate on
purpose. `tests/core/ot_reference.py` is the slow, obviously-correct scipy oracle that
both OT couplings are validated against — it is test scaffolding, not shipped code,
which is why it lives under `tests/`.

## Install

```bash
pip install -e .            # core
pip install -e .[dev,log]   # tests + tensorboard
```

## Quickstart

```python
from eesi import TimeMLP, EESI, GaussianMixture

d = 8
net_b = TimeMLP(d=d)
net_s = TimeMLP(d=d)
# path in {"linear", "trig", "encdec"}, gamma in {"none", "quad", "sqrt"}
model = EESI(net_b=net_b, net_s=net_s, d=d, path="linear", gamma="quad")

# training: x0, x1 are [B, d]
losses = model.loss(x1, x0)
(losses["b"] + losses["s"]).backward()

# sampling
x1_hat = model.sample_ode(x0, n_steps=100)
x1_hat, entropy = model.sample_ode_entropy(x0, n_steps=100)
```

## Training

Both systems train from the command line or from a notebook:

```bash
python -m eesi.systems.xy.train --steps 2000 --batch 256 --J 1.0
python -m eesi.systems.lj13.train --steps 2000 --batch 64
```

```python
from eesi.systems.xy.train import train, load_mc_data

data = load_mc_data(L=32, J=1.0)
model, hist = train(data, steps=2000)
```

`eesi.systems.lj13.train` needs the OSF reference data (`all_data_LJ13-1000.npy`, Klein,
Krämer & Noé 2023, [osf.io/srqg7](https://osf.io/srqg7/)); it is a 1.5 GB download,
not in the repo. See [data/README.md](data/README.md) for where it goes and how to
point elsewhere with `EESI_DATA_DIR`.

Both LJ13 artifacts resolve from the package rather than the cwd, so this works from
any directory:

```python
from eesi.systems.lj13.dynamics import LJ13Dynamics

dyn = LJ13Dynamics.from_checkpoint()   # defaults to the bundled CKPT_PATH
```
