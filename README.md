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
│   ├── interpolant.py          # EESI / xyEESI: loss, samplers, entropy estimators
│   ├── ot.py                   # equivariant-OT couplings (LJ13, XY chain)
│   ├── utils.py                # EMA, checkpointing, seeding, eval hook
│   ├── models/                 # the networks
│   │   ├── mlp.py              # TimeMLP: time-conditioned MLP field on R^d
│   │   ├── xygnn.py            # XYChainGNN: static-graph net for the 1D XY chain
│   │   ├── lj13_dynamics.py    # LJ13Dynamics + rk4_sample, divergence, free_energy
│   │   └── LJ13_eq_OT_flow_matching   # the released OSF checkpoint (gitignored)
│   ├── datasets/               # the systems themselves: data, priors, energies
│   │   ├── base.py             # GaussianMixture base distribution
│   │   ├── data.py             # ParticleDataset and make_loader
│   │   ├── xy.py               # 1D classical XY Monte Carlo (mcxy)
│   │   └── lj13.py             # LJ13 reference data, prior, energies, subspace geometry
│   └── train/                  # training entry points
│       ├── xy.py               # python -m eesi.train.xy
│       └── lj13.py             # python -m eesi.train.lj13
├── experiments/                # notebooks -- where the physics gets done
├── benchmarks/                 # performance measurement (bench_ot.py)
├── tests/                      # correctness (+ ot_reference.py, the scipy OT oracle)
├── data/                       # large third-party downloads (gitignored) -- see data/README.md
└── plans/                      # design docs
```

`tests/` checks correctness; `benchmarks/` measures speed. They are separate on
purpose. `tests/ot_reference.py` is the slow, obviously-correct scipy oracle that
`eesi/ot.py` is validated against — it is test scaffolding, not shipped code, which
is why it lives under `tests/`.

The line between `datasets/` and `models/` is whether a symbol needs a trained field
to mean anything. `datasets/lj13.py` holds closed-form facts about the system — the
reference data, the prior (`sample_prior`, `log_prior`), the energies, the geometry of
the COM-free subspace (`DOF`, `subspace_dirs`) — all true whether or not a model was
ever fit. `models/lj13_dynamics.py` holds the flow and everything you get by running
it: `rk4_sample`, `divergence`, `integrate_with_logdet`, `free_energy`. The dependency
runs one way, `models → datasets`.

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
python -m eesi.train.xy --steps 2000 --batch 256 --J 1.0
python -m eesi.train.lj13 --steps 2000 --batch 64
```

```python
from eesi.train.xy import train, load_mc_data

data = load_mc_data(L=32, J=1.0)
model, hist = train(data, steps=2000)
```

`eesi.train.lj13` needs the OSF reference data (`all_data_LJ13-1000.npy`, Klein,
Krämer & Noé 2023, [osf.io/srqg7](https://osf.io/srqg7/)); it is a 1.5 GB download,
not in the repo. See [data/README.md](data/README.md) for where it goes and how to
point elsewhere with `EESI_DATA_DIR`.

Both LJ13 artifacts resolve from the package rather than the cwd, so this works from
any directory:

```python
from eesi.models.lj13_dynamics import LJ13Dynamics

dyn = LJ13Dynamics.from_checkpoint()   # defaults to the bundled CKPT_PATH
```
