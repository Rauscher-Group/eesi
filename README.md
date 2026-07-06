# eesi

**Entropy Estimation by Stochastic Interpolants** — general stochastic interpolants
on Euclidean space for amorphous (particle) systems, with a cutoff-EGNN backbone.

## Layout

```
eesi/
├── pyproject.toml
├── README.md
└── eesi/
    ├── __init__.py
    ├── egnn.py     # radius graph + EGCL + global attention + backbone
    ├── model.py    # EESI: loss, sample_ode, sample_sde, entropy samplers
    ├── utils.py    # EMA, checkpointing, seeding, eval hook
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
from eesi import EGNN, EESI, GaussianMixture

net_b = EGNN(d=3, r_cut=2.5)
net_s = EGNN(d=3, r_cut=2.5)
# path in {"linear", "trig", "encdec"}, gamma in {"none", "quad", "sqrt"}
model = EESI(net_b=net_b, net_s=net_s, d=3, path="linear", gamma="quad")

# training
losses = model.loss(x1, x0)
(losses["b"] + losses["s"]).backward()

# sampling
x1_hat = model.sample_ode(x0, n_steps=100)
x1_hat, entropy = model.sample_ode_entropy(x0, n_steps=100)
```
