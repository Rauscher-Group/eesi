# eesi

**Entropy Estimation by Stochastic Interpolants (EESI)** — general stochastic interpolants
with entropy estimation. Four systems are built out: a Gaussian Mixture Model, 
a 1D classical XY chain, the 13-atom Lennard-Jones cluster (using E(n)-GNN equivariant 
graph neural networks) and a tangentially active Brownian polymer.

## Layout

```
.
├── pyproject.toml
├── README.md
├── eesi/                       # source only
│   ├── __init__.py
│   ├── interpolant.py          # EESI: loss, samplers, entropy estimators
│   ├── ot.py                   # group-agnostic optimal transport helpers
│   ├── egnn.py                 # Satorras E(n)-GNN backbone, shared by lj13 and tap
│   ├── config.py               # YAML run configs: schema vocabulary, stages, overrides
│   ├── rundir.py               # run directories: checkpoints, history, logging
│   └── systems/                # one self-contained subpackage per system
│       ├── gmm/                # 40-D Gaussian mixture model (toy system)
│       │   ├── data.py         # GaussianMixture base distribution
│       │   ├── mlp.py          # TimeMLP: time-conditioned MLP field on R^d
│       │   ├── ot.py           # plain minibatch OT, no symmetry group
│       │   └── train.py        # python -m eesi.systems.gmm.train
│       ├── xy/                 # the 1D XY chain
│       │   ├── data.py         # exact sampler using the von Mises distribution
│       │   ├── gnn.py          # XYChainGNN: static-graph net for the chain
│       │   ├── interpolant.py  # xyEESI: geodesic interpolant on (S^1)^N
│       │   ├── ot.py           # equivariant OT over O(2) x Z2 symmetry group
│       │   └── train.py        # python -m eesi.systems.xy.train
│       ├── lj13/               # the LJ 13-particle cluster
│       │   ├── data.py         # reference data, prior, energies, subspace geometry
│       │   ├── dynamics.py     # LJ13Dynamics + Runge-Kutta integrator, divergence, free_energy
│       │   ├── interpolant.py  # LJ13EESI: interpolant on the COM-free subspace
│       │   ├── ot.py           # equivariant OT over S(N) x SO(3)
│       │   ├── train.py        # python -m eesi.systems.lj13.train
│       │   └── LJ13_eq_OT_flow_matching   # the released OSF checkpoint of Klein et al. 2023 (gitignored)
│       └── tap/                # the tangentially active polymer
│           ├── data.py         # semiflexible prior, subspace geometry, observables
│           ├── dynamics.py     # TAPDynamics + Runge-Kutta integrator, divergence
│           ├── interpolant.py  # TAPEESI: interpolant on the tail-anchored subspace
│           ├── ot.py           # equivariant OT over O(3) - no permutation layer
│           ├── train.py        # python -m eesi.systems.tap.train  (single-shot)
│           ├── config.py       # the TAP run schema and prior resolution
│           └── run.py          # python -m eesi.systems.tap.run    (config-driven)
├── experiments/                # notebooks -- where the physics gets done
└── data/                       # large third-party downloads (gitignored) -- see data/README.md
    └── checkpoints/            # trained weights, mirrored by system (gitignored except
                                 # its own README) -- see data/checkpoints/README.md
```

The package is grouped by system, not by layer: everything the XY chain needs
sits in `eesi/systems/xy/`, everything LJ13 needs in `eesi/systems/lj13/`, etc. and 
the systems never import from each other. What is left at the top level is the generic 
parts: the interpolant, the group-agnostic OT helpers, the E(n)-GNN backbone, and tools
for config file parsing and run directory setup.

Within a system, the line between `data.py` files hold precise artifacts, like the
reference data, the prior sampler (`sample_prior`, `log_prior`), energy functions, 
geometry operations, etc. Everything else is the

## Install

```bash
pip install -e .            # core
pip install -e .[dev,log]   # tests + tensorboard
```

### Data & checkpoints

Two things are downloaded rather than shipped in the repo: the OSF reference data +
released checkpoint for LJ13 (Klein, Krämer & Noé 2023), and the mirror of every
system's trained weights under `data/checkpoints/` (see
[data/checkpoints/README.md](data/checkpoints/README.md)). After installing, you can 
pull both with:

```bash
eesi-fetch-data              # both --osf and --checkpoints
eesi-fetch-data --osf        # just the OSF LJ13 data + checkpoint
eesi-fetch-data --checkpoints  # just data/checkpoints/, from the companion repo
```

`--checkpoints` needs `EESI_CHECKPOINTS_REPO` set to that repo's URL, which does not
exist yet. Until then it exits with an explanation rather than a stack trace.

If you want to avoid installation, just do what the notebooks themselves do: download 
`all_data_LJ13-1000.npy` and `LJ13_eq_OT_flow_matching` by hand from [osf.io/srqg7](https://osf.io/srqg7/) 
into the paths [data/README.md](data/README.md) documents, and populate `data/checkpoints/`
per-system as described in its own README.

Both locations resolve from `eesi` itself rather than a bare relative path, so a
notebook can find them from anywhere:

```python
from eesi import CHECKPOINTS_DIR   # data/checkpoints/, or $EESI_DATA_DIR/checkpoints

model.net_b.load_state_dict(torch.load(CHECKPOINTS_DIR / "gmm" / "gmm_b.pth"))
```

`CHECKPOINTS_DIR` and `DATA_DIR` both live in `eesi.paths` and both accept
`EESI_DATA_DIR` as an override.

## Quickstart

```python
from eesi import EESI, GaussianMixture, TimeMLP

d, B = 8, 64
base = GaussianMixture(d=d, n=3, sigma=4.0, seed=0)   # the t=0 distribution

net_b = TimeMLP(d=d)   # velocity field b(t, x)
net_s = TimeMLP(d=d)   # score field s(t, x)
# path in {"linear", "trig", "trig2", "encdec"}, gamma in {"none", "quad", "sqrt", "sin2"}
model = EESI(net_b=net_b, net_s=net_s, d=d, path="linear", gamma="sqrt")

# training: x0 (base) and x1 (data) are both [B, d]
x0 = base.sample(B)
losses = model.loss(x1, x0)                # {"b": ..., "s": ...}
(losses["b"] + losses["s"]).backward()
```

Sampling is a single call for the whole family — `eps` picks ODE vs SDE (0 -> ODE, otherwise SDE),
`entropy` adds the entropy calculation, `return_traj` keeps the whole path:

```python
x1_hat           = model.sample(x0, n_steps=100)                      # probability-flow ODE
x1_hat           = model.sample(x0, n_steps=100, eps=0.1)             # reverse-time SDE
x1_hat, ent      = model.sample(x0, n_steps=100, entropy="dot")       # ent is [B]
traj, ts         = model.sample(x0, n_steps=100, return_traj=True)    # [n_steps+1, B, d]
```

`entropy="dot"` accumulates `-b·s` and needs `net_s`; `entropy="div"`
accumulates `div(b)` and needs only `net_b`. The two agree in expectation, not distribution.

## Training

Systems train from the command line or from a notebook:

```bash
python -m eesi.systems.xy.train --steps 2000 --batch 256 --J 1.0
python -m eesi.systems.lj13.train --steps 2000 --batch 64
```

```python
from eesi.systems.xy.train import train, load_exact_data

data = load_exact_data(N=32, J=1.0)   # exact sampler; load_mc_data is the MC cross-check
model, hist = train(data, steps=2000)
```

`eesi.systems.lj13.train` uses the OSF reference data (`all_data_LJ13-1000.npy`, Klein,
Krämer & Noé 2023, [osf.io/srqg7](https://osf.io/srqg7/)); it is a 1.5 GB download,
not in the repo. See [data/README.md](data/README.md) for where it goes and how to
point elsewhere with `EESI_DATA_DIR`.

Both LJ13 artifacts resolve from the package rather than the cwd, so this works from
any directory:

```python
from eesi.systems.lj13.dynamics import LJ13Dynamics

dyn = LJ13Dynamics.from_checkpoint()   # defaults to the bundled CKPT_PATH
```

### Long runs: TAP from a config file

TAP needs far more training than the others, enough that a flag list and a notebook
cell stop being workable. `eesi.systems.tap.run` reads a YAML config, writes
everything into one run directory, checkpoints as it goes, and resumes exactly where
it stopped. See [experiments/TAP/configs/tap_Pe0.yaml](experiments/TAP/configs/tap_Pe0.yaml),
which is commented knob by knob.

```bash
python -m eesi.systems.tap.run train --config experiments/TAP/configs/tap_Pe0.yaml
python -m eesi.systems.tap.run train --run runs/tap_long --resume
python -m eesi.systems.tap.run status  --run runs/tap_long
python -m eesi.systems.tap.run sample  --run runs/tap_long --n 20000
python -m eesi.systems.tap.run entropy --run runs/tap_long --batches 500
```

Launched detached via tmux, so it outlives the terminal:

```bash
tmux new-session -d -s tap-train \
  'python -m eesi.systems.tap.run train \
       --config experiments/TAP/configs/tap_Pe0.yaml --run runs/tap_long'

tail -f runs/tap_long/log.txt        # follow it from anywhere
tmux attach -t tap-train             # or watch it; detach with C-b then d
tmux send-keys -t tap-train C-c      # stop: the step finishes, a checkpoint lands

tmux new-session -d -s tap-train \
  'python -m eesi.systems.tap.run train --run runs/tap_long --resume'
```

A resume reproduces an uninterrupted run bit for bit on CPU in float64, because the
checkpoint carries the optimizer and the state of all four random generators. On CUDA
the reductions are not deterministic, so a resumed run is very close rather than
identical.

A resume takes its config from the checkpoint, not from the YAML file. By the time
anyone returns to a stopped run the file has usually been edited, and adopting those
edits mid-run is how a model ends up trained against two different priors.

The notebook then only reads what the run produced:

```python
from eesi.systems.tap.run import load_run

run = load_run("runs/tap_long")
K, B, GAMMA, COS0 = run.prior.as_tuple()   # the prior the model was actually trained on
h = run.history()                          # {"global_step": ..., "loss_b": ..., "S_dot": ...}
means_div = run.artifact("entropy", "means_div.npy")
```

Everything generic about this -- the schema vocabulary, the run directory, the
checkpoint format -- is in `eesi/config.py` and `eesi/rundir.py`, so XY or LJ13 can
adopt it without importing anything from `tap`.
