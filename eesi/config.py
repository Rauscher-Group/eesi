"""YAML run configuration: loading, overrides, and the validation primitives.

System-agnostic. What lives here is the part of a run's description that has nothing
to do with which physical system is being trained -- where it runs, in what precision,
and the sequence of learning-rate stages -- plus the small validator vocabulary that
each system's own schema module is built out of (`eesi.systems.tap.config`).

The design rule for the validator is that a typo must fail in the first second of a
200_000-step run, not silently produce a defaulted one. So `_closed` rejects unknown
keys rather than ignoring them, every error names the dotted path it came from, and a
near-miss gets a suggestion:

    ValueError: model: unknown key(s) 'hidden_nff' (did you mean 'hidden_nf'?).
    Allowed: ['bond_feature', 'eps', 'gamma', ...]

Two YAML traps are handled here rather than left to bite:

  * PyYAML resolves `1e-4` as the STRING "1e-4" -- its float pattern demands both a
    decimal point and a signed exponent, so only `1.0e-4` is a number. Rather than
    make every config author remember that, `_typed` accepts a string that parses as
    a float. Both spellings work and mean the same thing.
  * `bool` is a subclass of `int` in Python, so `steps: true` would satisfy a naive
    isinstance check against `int`. It is rejected explicitly.

Nothing here imports torch beyond the two `resolve_*` helpers, which exist so that
"auto"/"float64" in a config file become real torch objects in exactly one place.
"""
from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

__all__ = [
    "load_yaml", "apply_overrides", "resolve_device", "resolve_dtype",
    "StageConfig", "CheckpointConfig", "stages_from_dict", "checkpoint_from_dict",
    "ENTROPY_CHANNELS",
]

#: The `entropy` settings `EESI.loss` accepts. Mirrors `EESI._check_entropy_channel`;
#: validating here means a bad value fails at config load instead of on step 0.
ENTROPY_CHANNELS = (None, "dot", "zdot", "both")

_MISSING = object()


# --- validation primitives --------------------------------------------------


def _closed(raw: dict, allowed: Sequence[str], *, where: str) -> None:
    """Reject unknown keys, suggesting the intended one. Call BEFORE reading fields."""
    unknown = set(raw) - set(allowed)
    if not unknown:
        return
    parts = []
    for key in sorted(map(str, unknown)):
        near = difflib.get_close_matches(key, list(allowed), n=1)
        parts.append(f"{key!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
    raise ValueError(f"{where}: unknown key(s) {', '.join(parts)}. "
                     f"Allowed: {sorted(allowed)}")


def _typed(raw: dict, key: str, kind: type, default: Any = _MISSING, *, where: str) -> Any:
    """One scalar field: presence, type, and an error that names the dotted path.

    `float` fields accept ints and float-parseable strings -- see the module docstring
    on PyYAML's `1e-4`. `bool` is never accepted where an int or float is asked for.
    """
    if key not in raw:
        if default is _MISSING:
            raise ValueError(f"{where}: missing required key {key!r}")
        return default
    v = raw[key]
    if kind is not bool and isinstance(v, bool):
        raise ValueError(f"{where}.{key}: expected {kind.__name__}, got {v!r}")
    if kind is float:
        if isinstance(v, int):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                pass
    if not isinstance(v, kind):
        raise ValueError(f"{where}.{key}: expected {kind.__name__}, got {v!r}")
    return v


def _optional(raw: dict, key: str, kind: type, *, where: str) -> Any:
    """Like `_typed` with default None, but an explicit `null` is also allowed."""
    if raw.get(key, None) is None:
        return None
    return _typed(raw, key, kind, where=where)


def _choice(v: Any, options: Sequence[Any], *, where: str) -> Any:
    if v not in options:
        raise ValueError(f"{where}: must be one of "
                         f"{sorted(map(str, options))}, got {v!r}")
    return v


def _positive(v: float | int, *, where: str) -> float | int:
    if not v > 0:
        raise ValueError(f"{where}: must be positive, got {v!r}")
    return v


def _mapping(raw: dict, key: str, *, where: str) -> dict:
    """A sub-block, defaulting to empty so every field inside can carry its own default."""
    v = raw.get(key, None)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"{where}.{key}: expected a mapping, got {v!r}")
    return v


def _str_seq(raw: dict, key: str, default: Sequence[str], *, where: str) -> tuple[str, ...]:
    """A list of strings; a bare string is accepted as a one-element list."""
    v = raw.get(key, _MISSING)
    if v is _MISSING:
        return tuple(default)
    if isinstance(v, str):
        return (v,)
    if not isinstance(v, (list, tuple)) or not all(isinstance(s, str) for s in v):
        raise ValueError(f"{where}.{key}: expected a list of strings, got {v!r}")
    return tuple(v)


# --- YAML loading and CLI overrides -----------------------------------------


def load_yaml(path: str | Path) -> dict:
    """Parse a YAML config into a plain dict. An empty file is an empty config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return raw


def apply_overrides(raw: dict, overrides: Sequence[str]) -> dict:
    """Apply `--set a.b.1.lr=1e-6` style overrides to a parsed config. Returns a copy.

    The right-hand side is parsed as YAML, so `--set data.n=null`, `--set model.eps=1e-3`
    and `--set entropy.methods=[dot,div]` all do what they look like.

    An integer path component indexes a list, which is how a single stage is reached.
    The leaf must ALREADY EXIST: `--set modle.hidden_nf=64` fails here, naming the
    missing key, rather than adding one that `_closed` would later reject with a
    confusing message about an unknown key the user never typed.
    """
    out = copy.deepcopy(raw)
    for item in overrides:
        path, sep, rhs = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r}: expected KEY=VALUE")
        parts = path.strip().split(".")
        if not all(parts):
            raise ValueError(f"override {item!r}: empty path component")
        node: Any = out
        for depth, part in enumerate(parts[:-1]):
            node = _descend(node, part, parts[:depth + 1], item)
        _assign(node, parts[-1], yaml.safe_load(rhs), parts, item)
    return out


def _descend(node: Any, part: str, seen: Sequence[str], item: str) -> Any:
    trail = ".".join(seen)
    if isinstance(node, list):
        idx = _index(part, trail, item)
        if not -len(node) <= idx < len(node):
            raise ValueError(f"override {item!r}: index {idx} out of range at {trail!r} "
                             f"({len(node)} entries)")
        return node[idx]
    if not isinstance(node, dict) or part not in node:
        raise ValueError(f"override {item!r}: no key {trail!r} in the config")
    return node[part]


def _assign(node: Any, part: str, value: Any, parts: Sequence[str], item: str) -> None:
    trail = ".".join(parts)
    if isinstance(node, list):
        idx = _index(part, trail, item)
        if not -len(node) <= idx < len(node):
            raise ValueError(f"override {item!r}: index {idx} out of range at {trail!r} "
                             f"({len(node)} entries)")
        node[idx] = value
        return
    if not isinstance(node, dict) or part not in node:
        raise ValueError(f"override {item!r}: no key {trail!r} in the config. Overrides "
                         f"only change values that are already written out, so a typo "
                         f"fails here rather than becoming a new key.")
    node[part] = value


def _index(part: str, trail: str, item: str) -> int:
    try:
        return int(part)
    except ValueError:
        raise ValueError(f"override {item!r}: {trail!r} is a list, so the path component "
                         f"{part!r} must be an integer index") from None


# --- torch handles ----------------------------------------------------------


def resolve_device(spec: str) -> torch.device:
    """"auto" -> cuda when there is one, else cpu. Anything else goes to torch."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


_DTYPES = {"float64": torch.float64, "double": torch.float64,
           "float32": torch.float32, "float": torch.float32}


def resolve_dtype(spec: str) -> torch.dtype:
    if spec not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {spec!r}")
    return _DTYPES[spec]


# --- stages -----------------------------------------------------------------


@dataclass(frozen=True)
class StageConfig:
    """One learning-rate stage: a `train_si` call with its own lr, and its own seed.

    The convention this encodes is the notebooks': stages are separate optimizer runs
    at decreasing lr, not a scheduler. `reset_optimizer=False` carries Adam's moments
    across the boundary instead, which is a different experiment and is why it is a
    knob rather than a fixed behaviour.
    """
    name: str
    steps: int
    lr: float
    batch: int = 64
    seed: int | None = None       # None inherits the run's base seed
    align: bool = True
    batch_ot: bool = True
    entropy: str | None = None
    log_every: int = 200
    ckpt_every: int = 2000
    reset_optimizer: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    keep: int = 3
    export_nets: bool = True


#: Fields a stage may inherit from the enclosing `train:` block. `name` is deliberately
#: absent: stages sharing one name would make the checkpoint filenames collide.
_STAGE_INHERITED = ("steps", "lr", "batch", "seed", "align", "batch_ot", "entropy",
                    "log_every", "ckpt_every", "reset_optimizer")
_STAGE_KEYS = ("name",) + _STAGE_INHERITED
_TRAIN_KEYS = _STAGE_INHERITED + ("stages",)


def stages_from_dict(raw: dict, *, where: str = "train") -> tuple[StageConfig, ...]:
    """The `train:` block -> a tuple of stages, with block-level keys as defaults.

    `steps` and `lr` may be given at block level (every stage the same length) or per
    stage; either way each stage must end up with both. All stages must agree on
    `entropy`, because it sets the width of the run's single history file.
    """
    _closed(raw, _TRAIN_KEYS, where=where)
    entries = raw.get("stages", _MISSING)
    if entries is _MISSING:
        raise ValueError(f"{where}: missing required key 'stages'")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{where}.stages: expected a non-empty list of stages, "
                         f"got {entries!r}")

    defaults = {k: raw[k] for k in _STAGE_INHERITED if k in raw}
    stages = []
    for i, entry in enumerate(entries):
        at = f"{where}.stages[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{at}: expected a mapping, got {entry!r}")
        _closed(entry, _STAGE_KEYS, where=at)
        merged = {**defaults, **entry}
        stages.append(StageConfig(
            name=_typed(merged, "name", str, f"stage{i}", where=at),
            steps=_positive(_typed(merged, "steps", int, where=at), where=f"{at}.steps"),
            lr=_positive(_typed(merged, "lr", float, where=at), where=f"{at}.lr"),
            batch=_positive(_typed(merged, "batch", int, 64, where=at), where=f"{at}.batch"),
            seed=_optional(merged, "seed", int, where=at),
            align=_typed(merged, "align", bool, True, where=at),
            batch_ot=_typed(merged, "batch_ot", bool, True, where=at),
            entropy=_choice(merged.get("entropy", None), ENTROPY_CHANNELS,
                            where=f"{at}.entropy"),
            log_every=_typed(merged, "log_every", int, 200, where=at),
            ckpt_every=_positive(_typed(merged, "ckpt_every", int, 2000, where=at),
                                 where=f"{at}.ckpt_every"),
            reset_optimizer=_typed(merged, "reset_optimizer", bool, True, where=at),
        ))

    channels = {s.entropy for s in stages}
    if len(channels) > 1:
        raise ValueError(
            f"{where}: every stage must use the same `entropy` setting, got "
            f"{sorted(map(str, channels))}. It sets how many loss columns a step "
            f"records, and the run writes one history file with one header.")

    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: stage names must be unique, got {names}")
    return tuple(stages)


def checkpoint_from_dict(raw: dict, *, where: str = "checkpoint") -> CheckpointConfig:
    _closed(raw, ("keep", "export_nets"), where=where)
    return CheckpointConfig(
        keep=_typed(raw, "keep", int, 3, where=where),
        export_nets=_typed(raw, "export_nets", bool, True, where=where),
    )


def stage_to_dict(stage: StageConfig) -> dict:
    """A stage as plain YAML-able data, fully defaulted. Round-trips through
    `stages_from_dict`, which is what makes a run's `resolved.json` reproducible."""
    return {"name": stage.name, "steps": stage.steps, "lr": stage.lr,
            "batch": stage.batch, "seed": stage.seed, "align": stage.align,
            "batch_ot": stage.batch_ot, "entropy": stage.entropy,
            "log_every": stage.log_every, "ckpt_every": stage.ckpt_every,
            "reset_optimizer": stage.reset_optimizer}


def checkpoint_to_dict(cfg: CheckpointConfig) -> dict:
    return {"keep": cfg.keep, "export_nets": cfg.export_nets}
