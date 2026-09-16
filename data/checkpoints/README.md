# data/checkpoints

Trained model weights, mirrored by system (matching `eesi/systems/`). Everything here
except this file is gitignored — the binaries are meant to live in a separate
checkpoints repo (not yet created) that `eesi-fetch-data --checkpoints` will pull from
once its URL is filled in (see `eesi/fetch.py`). Until then this tree is populated by
hand, as a copy of whatever the notebooks under `experiments/` produced.

None of these paths are wired into the notebooks yet — the notebooks still save/load
via bare relative filenames next to themselves (e.g. `./gmm_b.pth`), or, for TAP, under
`runs/tap_<name>/nets/`. This directory is a staging copy for when those cells are
repointed by hand, at which point `eesi.CHECKPOINTS_DIR` (`eesi.paths.CHECKPOINTS_DIR`,
honoring `EESI_DATA_DIR`) is what a notebook cell should resolve against rather than a
bare relative path:

```python
from eesi import CHECKPOINTS_DIR

model.net_b.load_state_dict(torch.load(CHECKPOINTS_DIR / "gmm" / "gmm_b.pth"))
```

## `gmm/`

`gmm_b.pth`, `gmm_s.pth` — velocity/score `TimeMLP` weights trained in
`experiments/GMM/GMM.ipynb`. Consumed there via
`model.net_b.load_state_dict(torch.load(...))`.

## `xy/`

`xy_b_N10.pth`, `xy_s_N10.pth` — velocity/score weights for the `N=10` 1D XY chain,
trained in `experiments/XY/XY.ipynb`.

## `lj13/`

`lj13_b_avg.pth`, `lj13_s_avg.pth` — the in-house fine-tuned (EMA-averaged) velocity and
score weights from `experiments/LJ/LJ13.ipynb`'s `train_si` stage. **Not** the OSF-released
checkpoint: that one (`LJ13_eq_OT_flow_matching`, Klein, Krämer & Noé 2023) lives beside
`eesi/systems/lj13/dynamics.py` and resolves via `LJ13Dynamics.from_checkpoint()`
(`CKPT_PATH`) — see the root `README.md` and `../README.md` for how to fetch it.

## `tap/`

One subdirectory per `runs/tap_<name>/nets/`, name with the `tap_` prefix stripped, each
holding `tap_b_si_N20_B256.pth` and `tap_s_si_N20_B256.pth` — the final SI network
weights `eesi.systems.tap.run` wrote out for that sweep run. Subdirectory names track the
`Pe*` labels used in `experiments/TAP/results/*.{csv,json}` (Peclet number sweep), except
for `Pe0_train`, `Pe0_train2`, and `Pe0_100k`, which are three separate `Pe=0` training
attempts — nothing here has determined which (if any) is canonical.

Only the final `nets/*.pth` are mirrored. `runs/tap_<name>/checkpoints/*.pt`
(optimizer + RNG state, for resuming training) is deliberately excluded: it's larger,
it's not what any notebook reads, and it isn't a "result" in the same sense as the other
systems' checkpoints.
