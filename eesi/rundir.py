"""Run directories: disk layout, atomic checkpoints, append-only history, logging.

System-agnostic, and the piece that makes a multi-hour job survivable. A run is a
directory, and everything about it -- what produced it, how far it got, and what it
found -- is a file inside:

    runs/tap_N20_Pe0/
    |-- config.yaml     byte-for-byte copy of the input config
    |-- resolved.json   fully-defaulted config + whatever the system resolved from
    |                   data (a prior, a normalisation) + versions + git sha
    |-- log.txt         append-only; everything that went to stdout
    |-- history.csv     append-only, one row per training step
    |-- checkpoints/    latest.pt, plus rolling per-stage checkpoints
    |-- nets/           bare per-network weights, the notebooks' old format
    |-- samples/        generated configurations
    `-- entropy/        entropy estimates

Three properties are load-bearing, and each costs a few lines here rather than a lost
run later:

  * Checkpoints are written to a temporary file and then `os.replace`d, which is
    atomic on POSIX. A crash during a write leaves the previous checkpoint intact
    rather than a truncated file that loads as garbage.
  * History is CSV, not `.npy`. It can be appended to without rewriting, truncated
    back to a checkpoint's step count on resume, and read with `tail -f` while the
    job runs.
  * `RunDir.create` refuses to write into a directory that already holds a run. A
    mistyped `--run` must not quietly overwrite a week of training.

A checkpoint holds enough to continue the run bit-for-bit: the whole model, the
optimizer, where the stages got to, and the RNG state of all four generators. It also
holds the config that produced it, so a run directory can be understood without the
config file that started it -- which will have been edited by then.
"""
from __future__ import annotations

import csv
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

__all__ = [
    "RunDir", "Logger", "CKPT_FORMAT", "CKPT_VERSION",
    "save_checkpoint", "load_checkpoint", "split_nets",
    "rng_state", "set_rng_state", "env_info", "utc_now",
    "history_header", "append_history", "truncate_history", "load_history",
]

CKPT_FORMAT = "eesi.checkpoint"
CKPT_VERSION = 1

#: Columns every run's history carries, around the per-step loss columns the system
#: contributes. `stage` is the name, not the index, so the file stays readable when a
#: config is edited between runs.
_HISTORY_PRE = ("global_step", "stage", "stage_step")
_HISTORY_POST = ("transport", "elapsed_s")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the directory ----------------------------------------------------------


@dataclass
class RunDir:
    """A run's directory, and the paths inside it. Creating one makes the subdirs."""
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

    @classmethod
    def create(cls, root: str | Path, *, resume: bool = False,
               fresh: bool = False) -> "RunDir":
        """Make (or reopen) a run directory.

        Refuses a non-empty directory unless `resume` (continue the run that is there)
        or `fresh` (discard it and start over). `fresh` will only delete something that
        looks like a run directory -- one holding a `config.yaml` -- so a mistyped
        `--run ~/Projects --fresh` fails instead of doing something unrecoverable.
        """
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            if fresh:
                if not (root / "config.yaml").exists():
                    raise ValueError(
                        f"refusing --fresh on {root}: it is not empty and holds no "
                        f"config.yaml, so it does not look like a run directory. "
                        f"Delete it by hand if that is really what you meant.")
                shutil.rmtree(root)
            elif not resume:
                raise FileExistsError(
                    f"{root} already holds a run. Pass --resume to continue it, "
                    f"--fresh to discard it, or --run with a different path.")
        elif resume:
            raise FileNotFoundError(f"nothing to resume: {root} does not exist or is empty")

        rd = cls(root)
        for d in (root, rd.checkpoints, rd.nets, rd.samples, rd.entropy):
            d.mkdir(parents=True, exist_ok=True)
        return rd

    # paths ------------------------------------------------------------------
    @property
    def config_yaml(self) -> Path: return self.root / "config.yaml"

    @property
    def resolved_json(self) -> Path: return self.root / "resolved.json"

    @property
    def log_txt(self) -> Path: return self.root / "log.txt"

    @property
    def history_csv(self) -> Path: return self.root / "history.csv"

    @property
    def checkpoints(self) -> Path: return self.root / "checkpoints"

    @property
    def nets(self) -> Path: return self.root / "nets"

    @property
    def samples(self) -> Path: return self.root / "samples"

    @property
    def entropy(self) -> Path: return self.root / "entropy"

    def latest(self) -> Path | None:
        """The checkpoint to resume from, or None if the run never got that far."""
        p = self.checkpoints / "latest.pt"
        return p if p.exists() else None

    def prune_checkpoints(self, keep: int, *, protect: Sequence[str] = ()) -> list[Path]:
        """Drop the oldest rolling checkpoints, keeping `keep` of them.

        `latest.pt` and anything ending `_final.pt` are never pruned: the first is the
        resume point and the second is a stage boundary, which is the checkpoint you
        actually want months later.
        """
        protected = {"latest.pt", *protect}
        rolling = sorted((p for p in self.checkpoints.glob("*.pt")
                          if p.name not in protected and not p.name.endswith("_final.pt")),
                         key=lambda p: p.stat().st_mtime)
        dropped = rolling[:max(0, len(rolling) - keep)] if keep >= 0 else []
        for p in dropped:
            p.unlink()
        return dropped


class Logger:
    """Print a line and append it to the run's log, flushed. That is the whole job.

    A long job's stdout lives in a tmux pane that will be gone by morning, so every
    line has to reach the file as it happens -- not on a buffer flush at exit that a
    kill -9 never gets to.
    """

    def __init__(self, path: str | Path, echo: bool = True):
        self.path = Path(path)
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str = "") -> None:
        if self.echo:
            print(msg, flush=True)
        with self.path.open("a") as fh:
            fh.write(msg + "\n")


# --- checkpoints ------------------------------------------------------------


def save_checkpoint(path: str | Path, **fields: Any) -> Path:
    """Write a checkpoint atomically: full write to `.tmp`, then `os.replace`.

    The stamped `format`/`version`/`created` keys are added here so every checkpoint
    in the repo carries them whether or not the caller remembered.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": CKPT_FORMAT, "version": CKPT_VERSION, "created": utc_now(),
               **fields}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict:
    """Load and validate a checkpoint written by `save_checkpoint`.

    `weights_only=False` because a checkpoint deliberately carries its config and
    provenance alongside the tensors. These are files this code wrote, in a directory
    the user owns; the flag is not doing safety work here that the filesystem is not
    already doing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CKPT_FORMAT:
        raise ValueError(
            f"{path} is not an {CKPT_FORMAT} checkpoint (format="
            f"{ckpt.get('format') if isinstance(ckpt, dict) else type(ckpt).__name__!r}). "
            f"Bare state_dicts -- what `python -m eesi.systems.tap.train --out` and the "
            f"notebooks write -- go through `init.net_b`/`init.net_s` instead.")
    if ckpt.get("version") != CKPT_VERSION:
        raise ValueError(f"{path} is checkpoint version {ckpt.get('version')}, but this "
                         f"code reads version {CKPT_VERSION}")
    return ckpt


def split_nets(model_sd: dict, prefixes: Sequence[str] = ("net_b.", "net_s.")) -> tuple[dict, ...]:
    """A wrapper's state_dict -> one bare state_dict per sub-network.

    The interpolant saves `net_b.*` / `net_s.*` in one dict; the notebooks have always
    saved the two fields as separate files, and `TAPDynamics.load_state_dict(sd,
    strict=True)` is what reads them back. Keeping both formats reachable is what lets
    the notebook's existing load cell go on working against a run directory.
    """
    out: tuple[dict, ...] = tuple({} for _ in prefixes)
    for key, value in model_sd.items():
        for prefix, dst in zip(prefixes, out):
            if key.startswith(prefix):
                dst[key[len(prefix):]] = value
    missing = [p for p, d in zip(prefixes, out) if not d]
    if missing:
        raise ValueError(f"no keys with prefix(es) {missing} in the state_dict; "
                         f"it holds {sorted(model_sd)[:4]}...")
    return out


# --- reproducibility --------------------------------------------------------


def rng_state() -> dict:
    """All four generators, so a resumed run continues the same random stream."""
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate()}


def set_rng_state(state: dict) -> None:
    """Restore what `rng_state` captured. CUDA states are skipped if the device count
    changed, since restoring a 2-GPU state onto 1 GPU is not meaningful."""
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"])
                        else state["torch"])
    cuda = state.get("cuda") or []
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])


def env_info(root: str | Path | None = None) -> dict:
    """Versions and the git sha, for the record. Never raises: a run must not fail
    because it was launched from outside a git checkout."""
    info = {"torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "recorded": utc_now()}
    try:
        info["git_sha"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root) if root else Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except Exception:
        info["git_sha"] = None
    return info


# --- history ----------------------------------------------------------------


def history_header(columns: Sequence[str]) -> list[str]:
    """The CSV header: the fixed run columns around the system's per-step `columns`.

    `columns` comes from the system's own label table (`_HIST_LABELS` in each
    `train.py`), so the width follows the `entropy` setting exactly as the in-memory
    history does.
    """
    return [*_HISTORY_PRE, *columns, *_HISTORY_POST]


def append_history(path: str | Path, rows: Iterable[Sequence[Any]],
                   header: Sequence[str] | None = None) -> None:
    """Append rows, writing `header` first if the file is new. Flushed on close."""
    path = Path(path)
    rows = list(rows)
    new = not path.exists() or path.stat().st_size == 0
    if new and header is None and rows:
        raise ValueError(f"{path} is new, so append_history needs a header")
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new and header is not None:
            w.writerow(header)
        w.writerows(rows)


def truncate_history(path: str | Path, n_rows: int) -> int:
    """Keep the header and the first `n_rows` data rows. Returns how many were dropped.

    Rows are flushed before the checkpoint that counts them is written, so after a
    crash the file can hold steps the checkpoint does not know about. Those steps are
    about to be replayed, and leaving them would duplicate them in the history.
    """
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return 0
    head, data = rows[0], rows[1:]
    if len(data) <= n_rows:
        return 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(data[:n_rows])
    return len(data) - n_rows


def load_history(path: str | Path) -> dict[str, np.ndarray]:
    """The history as {column: array}. Numeric columns become floats, `stage` stays str.

    This is what the notebook calls in place of `np.load("tap_train_hist.npy")`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"history not found: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    head, data = rows[0], rows[1:]
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(head):
        col = [r[i] if i < len(r) else "" for r in data]
        if name == "stage":
            out[name] = np.array(col, dtype=object)
        elif name in ("global_step", "stage_step"):
            out[name] = np.array([int(v) for v in col], dtype=np.int64)
        else:
            out[name] = np.array([float(v) if v != "" else np.nan for v in col])
    return out
