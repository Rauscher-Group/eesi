# eesi

**Entropy Estimation by Stochastic Interpolants** — general stochastic interpolants
on Euclidean space R^d. The `EESI` model operates on `[B, d]` vectors with a
time-conditioned MLP backbone; a cutoff-EGNN for `[B, N, d]` particle systems is
also included.

## Layout

```
eesi/
├── pyproject.toml
├── README.md
└── eesi/
    ├── __init__.py
    ├── mlp.py     # TimeMLP: time-conditioned MLP field on R^d
    ├── egnn.py    # radius graph + EGCL + global attention (particle backbone)
    ├── model.py   # EESI: loss, sample_ode, sample_sde, entropy samplers
    ├── utils.py   # EMA, checkpointing, seeding, eval hook
    └── datasets/
        ├── __init__.py
        ├── base.py # GaussianMixture base distribution
        └── data.py # ParticleDataset and make_loader
```

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
