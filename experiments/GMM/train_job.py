"""Standalone version of the training cell in experiments/GMM/GMM.ipynb.

Run it under tmux with `./experiments/GMM/tmux_job.sh`, or directly:

    python experiments/GMM/train_job.py --steps 100000 --out gmm

Writes three files next to this script: `<out>_b.pth` and `<out>_s.pth` (the drift
and score state_dicts, the two files the notebook's §2 load lines expect) and
`<out>_hist.npy`, the (steps, 4) history array whose columns are
(loss_b, loss_s, S_dot, S_zdot) -- the notebook's `hist_b, hist_s, hist_S_dot,
hist_S_zdot = np.asarray(htot).T`.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from eesi import EESI, TimeMLP, GaussianMixture
from eesi.systems.gmm.train import train

HERE = pathlib.Path(__file__).resolve().parent

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--batch", type=int, default=500)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--out", default="gmm", help="prefix for the output files")
p.add_argument("--init", default=None,
               help="prefix of an existing checkpoint to warm-start from: loads "
                    "'{init}_b.pth' and '{init}_s.pth' (as written by a previous "
                    "--out) into net_b/net_s before training")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

d = 40
target = GaussianMixture(dim=d, n_mixes=16, loc_scaling=2, log_var_scaling=-3,
                         seed=10, device=a.device)

torch.manual_seed(1)
model = EESI(TimeMLP(d=d, hidden=512, n_layers=4, activation="silu"),
             TimeMLP(d=d, hidden=3072, n_layers=2, activation="silu"),
             d=d, path="linear", gamma="sqrt", gamma_scale=0.2).to(a.device)

if a.init:
    b_path, s_path = HERE / f"{a.init}_b.pth", HERE / f"{a.init}_s.pth"
    model.net_b.load_state_dict(torch.load(b_path, map_location=a.device))
    model.net_s.load_state_dict(torch.load(s_path, map_location=a.device))
    print(f"warm start from {b_path} and {s_path}", flush=True)

print(f"device={a.device}  steps={a.steps}  batch={a.batch}  lr={a.lr}", flush=True)
model, htot = train(target, steps=a.steps, batch=a.batch, lr=a.lr, batch_ot=False,
                    device=a.device, model=model, log_every=100, entropy="both")

np.save(HERE / f"{a.out}_hist.npy", np.asarray(htot))
torch.save(model.net_b.state_dict(), HERE / f"{a.out}_b.pth")
torch.save(model.net_s.state_dict(), HERE / f"{a.out}_s.pth")
print(f"saved -> {HERE / a.out}_{{hist.npy,b.pth,s.pth}}")
