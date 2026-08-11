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
│   ├── ot.py                   # group-agnostic OT helpers (centering, Hungarian, SVD)
│   ├── egnn.py                 # Satorras E(n)-GNN backbone, shared by lj13 and tap
│   ├── config.py               # YAML run configs: schema vocabulary, stages, overrides
│   ├── rundir.py               # run directories: checkpoints, history, logging
│   └── systems/                # one self-contained subpackage per system
│       ├── gmm/                # the 1D/2D pedagogical examples (no interpolant subclass)
│       │   ├── data.py         # GaussianMixture base distribution
│       │   ├── mlp.py          # TimeMLP: time-conditioned MLP field on R^d
│       │   ├── ot.py           # plain minibatch OT, no symmetry group
│       │   └── train.py        # python -m eesi.systems.gmm.train
│       ├── xy/                 # the 1D XY chain
│       │   ├── data.py         # 1D classical XY Monte Carlo (mcxy), exact sampler
│       │   ├── gnn.py          # XYChainGNN: static-graph net for the chain
│       │   ├── interpolant.py  # xyEESI: geodesic interpolant on (S^1)^N
│       │   ├── ot.py           # equivariant OT over O(2) x Z2^site
│       │   └── train.py        # python -m eesi.systems.xy.train
│       ├── lj13/               # the LJ13 cluster
│       │   ├── data.py         # reference data, prior, energies, subspace geometry
│       │   ├── dynamics.py     # LJ13Dynamics + rk4_sample, divergence, free_energy
│       │   ├── interpolant.py  # LJ13EESI: interpolant on the COM-free subspace
│       │   ├── ot.py           # equivariant OT over S(N) x SO(3)
│       │   ├── train.py        # python -m eesi.systems.lj13.train
│       │   └── LJ13_eq_OT_flow_matching   # the released OSF checkpoint (gitignored)
│       └── tap/                # the tangentially active polymer (no energy: see below)
│           ├── data.py         # semiflexible prior, subspace geometry, observables
│           ├── dynamics.py     # TAPDynamics + rk4_sample, divergence
│           ├── interpolant.py  # TAPEESI: interpolant on the tail-anchored subspace
│           ├── ot.py           # equivariant OT over O(3) -- no permutation layer
│           ├── train.py        # python -m eesi.systems.tap.train  (single-shot)
│           ├── config.py       # the TAP run schema and prior resolution
│           └── run.py          # python -m eesi.systems.tap.run    (config-driven)
├── experiments/                # notebooks -- where the physics gets done
├── tests/                      # correctness, mirroring the source layout
│   ├── core/                   # + ot_reference.py, the scipy OT oracle
│   ├── gmm/  xy/  lj13/  tap/
└── data/                       # large third-party downloads (gitignored) -- see data/README.md
```

The package is grouped by **system**, not by layer: everything the XY chain needs
sits in `eesi/systems/xy/`, everything LJ13 needs in `eesi/systems/lj13/`, and the
two never import from each other. What is left at the top level is the part that is
genuinely general — the `EESI` interpolant, the group-agnostic OT helpers, and the
E(n)-GNN backbone. The dependency rule is that a system may import from the core and
from its own siblings, never the reverse. When two systems need the same component it
moves into the core rather than one system importing the other: that is why `egnn.py`
sits beside `interpolant.py` even though LJ13 was its only user for a while.

Within a system, the line between `data.py` and the model modules is whether a symbol
needs a trained field to mean anything. `lj13/data.py` holds closed-form facts — the
reference data, the prior (`sample_prior`, `log_prior`), the energies, the geometry of
the COM-free subspace (`DOF`, `subspace_dirs`) — all true whether or not a model was
ever fit. `lj13/dynamics.py` holds the flow and everything you get by running it:
`rk4_sample`, `divergence`, `integrate_with_logdet`, `free_energy`. The dependency
runs one way, `dynamics → data`.

`tap/` is the one system with **no energy function at all**, and that is a fact about
the physics rather than an omission. A tangentially active polymer is driven by a force
along its own backbone tangent, which is not the gradient of any potential: the steady
state is not a Boltzmann distribution, so there is nothing to reweight toward and no
`free_energy` analogue. Estimating its entropy is exactly the case where an interpolant
buys something no importance-sampling scheme can.

`tests/` checks correctness. `tests/core/ot_reference.py` is the slow,
obviously-correct scipy oracle that all three OT couplings are validated against — it is
test scaffolding, not shipped code, which is why it lives under `tests/`.

## Install

```bash
pip install -e .            # core
pip install -e .[dev,log]   # tests + tensorboard
```

## Quickstart

```python
from eesi import EESI, GaussianMixture, TimeMLP

d, B = 8, 64
base = GaussianMixture(d=d, n=3, sigma=4.0, seed=0)   # the t=0 distribution

net_b = TimeMLP(d=d)   # velocity field b(t, x)
net_s = TimeMLP(d=d)   # score field s(t, x)
# path in {"linear", "trig", "trig2", "encdec"}, gamma in {"none", "quad", "sqrt", "sin2"}
model = EESI(net_b=net_b, net_s=net_s, d=d, path="linear", gamma="quad")

# training: x0 (base) and x1 (data) are both [B, d]
x0 = base.sample(B)
losses = model.loss(x1, x0)                # {"b": ..., "s": ...}
(losses["b"] + losses["s"]).backward()
```

Sampling is a single call for the whole family — `eps` picks ODE vs SDE,
`entropy` adds the entropy channel, `return_traj` keeps the path:

```python
x1_hat           = model.sample(x0, n_steps=100)                      # probability-flow ODE
x1_hat           = model.sample(x0, n_steps=100, eps=0.1)             # reverse-time SDE
x1_hat, ent      = model.sample(x0, n_steps=100, entropy="dot")       # ent is [B]
traj, ts         = model.sample(x0, n_steps=100, return_traj=True)    # [n_steps+1, B, d]
```

`entropy="dot"` accumulates `-∫ b·s dt` and needs `net_s`; `entropy="div"`
accumulates `∫ div(b) dt` and needs only `net_b`. The two agree in expectation.

## Training

Both systems train from the command line or from a notebook:

```bash
python -m eesi.systems.xy.train --steps 2000 --batch 256 --J 1.0
python -m eesi.systems.lj13.train --steps 2000 --batch 64
```

```python
from eesi.systems.xy.train import train, load_exact_data

data = load_exact_data(N=32, J=1.0)   # exact sampler; load_mc_data is the MC cross-check
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

### Long runs: TAP from a config file

TAP needs far more training than the others, enough that a flag list and a notebook
cell stop being workable. `eesi.systems.tap.run` reads a YAML config, writes
everything into one run directory, checkpoints as it goes, and resumes exactly where
it stopped. See [experiments/TAP/configs/tap_N20_Pe0.yaml](experiments/TAP/configs/tap_N20_Pe0.yaml),
which is commented knob by knob.

```bash
python -m eesi.systems.tap.run train --config experiments/TAP/configs/tap_N20_Pe0.yaml
python -m eesi.systems.tap.run train --run runs/tap_long --resume
python -m eesi.systems.tap.run status  --run runs/tap_long
python -m eesi.systems.tap.run sample  --run runs/tap_long --n 20000
python -m eesi.systems.tap.run entropy --run runs/tap_long --batches 500
```

The config names the data file, says where each of the prior's four parameters comes
from (given, measured from the data, or fitted to match `E[Re^2]`), sets the network
and interpolant hyperparameters, and lists the learning-rate stages:

```yaml
prior:
  k: 100.0                                              # the Hamiltonian's value
  gamma: {mode: measure, estimator: inv_two_var_cos}    # read off the data
  cos_theta_0: {mode: solve, target: end_to_end_sq}     # fitted to match E[Re^2]
train:
  stages:
    - {name: coarse, steps: 100000, lr: 1.0e-4}
    - {name: fine,   steps: 100000, lr: 1.0e-5}
```

Launched detached, so it outlives the terminal:

```bash
tmux new-session -d -s tap-train \
  'python -m eesi.systems.tap.run train \
       --config experiments/TAP/configs/tap_N20_Pe0.yaml --run runs/tap_long'

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
adopt it without importing anything from `tap`. `python -m eesi.systems.tap.train` is
unchanged and remains the single-shot path for smoke tests and the coupling ablations.
