"""Repo-level filesystem locations, resolved once so every consumer agrees on them.

    DATA_DIR            the top-level `data/` directory (large downloads, gitignored)
    CHECKPOINTS_DIR      `data/checkpoints/`, trained weights mirrored by system

Anchored to this file's location rather than the caller's cwd, so both resolve
correctly whether the caller is a notebook launched from `experiments/<system>/`, a
script run from the repo root, or an installed (non-editable) copy elsewhere.
`EESI_DATA_DIR` overrides the whole tree at once -- a scratch disk, a shared copy --
which is why every consumer imports `DATA_DIR` from here rather than recomputing its
own `parents[N] / "data"` (`eesi.systems.lj13.data` and `eesi.systems.tap.data` did
exactly that independently before this module existed).
"""
from __future__ import annotations

import os
import pathlib

DATA_DIR = pathlib.Path(
    os.environ.get("EESI_DATA_DIR", pathlib.Path(__file__).resolve().parent.parent / "data")
)
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
