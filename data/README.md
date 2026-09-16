# data

Large third-party artifacts, downloaded rather than generated. Everything here except
this file is gitignored.

Kept outside the `eesi/` package so that installing or building the package never has
to copy gigabytes around.

## `all_data_LJ13-1000.npy` (1.5 GB)

LJ13 reference MCMC samples: 10M configurations x 39, COM-free, memory-mapped by the
consumers rather than loaded whole.

From Klein, Krämer & Noé 2023, "Equivariant flow matching", OSF
[srqg7](https://osf.io/srqg7/).

Referenced as `eesi.systems.lj13.data.REF_DATA_PATH`, which resolves to this directory.
Set `EESI_DATA_DIR` to point elsewhere (a scratch disk, a shared copy):

```bash
EESI_DATA_DIR=/scratch/eesi-data python -m eesi.systems.lj13.train --steps 2000
```

Consumers: `experiments/lj13_sampling.ipynb`, `eesi/train/lj13.py` (`--data`),
`benchmarks/bench_ot.py` (`--data`, optional).

## Related

The matching flow-matching checkpoint, `LJ13_eq_OT_flow_matching` (124 KB), is from the
same OSF record but lives beside the code that loads it, at
`eesi/systems/lj13/LJ13_eq_OT_flow_matching` (`eesi.systems.lj13.dynamics.CKPT_PATH`). It is
small enough that keeping it next to `LJ13Dynamics.from_checkpoint` costs nothing.

Both of the above can be fetched automatically with `eesi-fetch-data --osf` instead of
by hand.

## `checkpoints/`

Trained weights for every system, mirrored by system name -- see
[checkpoints/README.md](checkpoints/README.md). Unlike the OSF files above, these come
from a companion checkpoints repo (not yet created) rather than a third party;
`eesi-fetch-data --checkpoints` will pull it once `EESI_CHECKPOINTS_REPO` is set.
