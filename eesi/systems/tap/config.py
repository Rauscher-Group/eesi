"""The TAP run schema: YAML in, a validated `RunConfig` and a resolved prior out.

Built on `eesi.config`'s validator vocabulary; what is here is only the part that
knows about tangentially active polymers. Two jobs:

**The schema.** Every knob the notebook reaches for, including the ones the older
argparse CLI never exposed (`hidden_nf`, `n_layers`, `path`, `gamma_scale`, `eps`).
Two are deliberately absent, and their absence is load-bearing rather than an
oversight:

  * `learn_score` -- `TAPEESI.__init__` forces it False (the subspace divergence runs
    under no_grad and cannot feed ISM), so a config setting it True would be silently
    ignored, which is worse than not offering it.
  * `coords_range` -- `make_si_model(**kw)` forwards to `TAPEESI`, not to
    `TAPDynamics`, so there is no path from a config to the EGNN's own kwargs.

**The prior.** `sample_prior` takes four parameters, and where each one comes from is
a physics decision the notebook made in prose and comments. `resolve_prior` makes it
explicit and recorded:

    k: 100.0                                       fixed -- the simulation Hamiltonian
    gamma: {mode: measure, estimator: inv_two_var_cos}     read off the target data
    cos_theta_0: {mode: solve, target: end_to_end_sq}      fitted to match E[Re^2]

and writes a provenance string for each, so a run directory says which of the three
routes produced the number rather than leaving it to be reconstructed from a
notebook cell that has since been re-run.

The reason `k` and `b` default to the Hamiltonian's values (100.0, 1.0) rather than
the data's moments (~91.5, ~1.024) is the thermodynamic-integration bridge: it
reconciles the interpolant's dS against an external TI result by subtracting an
analytic prior-vs-ideal offset, and that subtraction is only valid if the prior's
bond law IS the simulation's. The `measure` route is correct whenever no such
reconciliation is being done, which is why both stay available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from ...config import (CheckpointConfig, StageConfig, _choice, _closed, _mapping,
                       _optional, _positive, _str_seq, _typed, apply_overrides,
                       checkpoint_from_dict, checkpoint_to_dict, load_yaml,
                       stage_to_dict, stages_from_dict)
from ...interpolant import _GAMMAS, _PATHS
from .data import (N_DEFAULT, N_DIMS, REF_DATA_PATH, angle_moments, bond_cosines,
                   bond_vectors, end_to_end_mean_sq, end_to_end_sq, gyration_sq,
                   prior_entropy, sample_prior, solve_cos_theta_0)

__all__ = [
    "DataConfig", "ParamSpec", "PriorConfig", "ModelConfig", "InitConfig",
    "SampleConfig", "EntropyConfig", "RunConfig", "ResolvedPrior",
    "load_config", "config_from_dict", "config_to_dict",
    "resolve_prior", "prior_report", "PRIOR_ESTIMATORS",
]

#: `EESI.entropy_estimate` methods. "bdot" is included because the method exists and
#: an ablation may want it; it is not what the notebook reports.
ENTROPY_METHODS = ("dot", "zdot", "div", "bdot")
#: What `EESI.sample(entropy=...)` accepts -- a strict subset of the above.
FLOW_CHANNELS = ("div", "dot")
INTEGRATORS = ("rk4", "heun", "euler", "sde")


# --- schema -----------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    path: str | None = None          # None -> data.REF_DATA_PATH ($EESI_DATA_DIR)
    n: int | None = None             # None -> the whole file
    n_particles: int = N_DEFAULT
    n_dims: int = N_DIMS

    @property
    def resolved_path(self) -> Path:
        return Path(self.path) if self.path else Path(REF_DATA_PATH)


@dataclass(frozen=True)
class ParamSpec:
    """Where one prior parameter comes from: given, measured, or fitted."""
    mode: str                        # "fixed" | "measure" | "solve"
    value: float | None = None       # mode="fixed"
    estimator: str | None = None     # mode="measure"
    target: str | float | None = None  # mode="solve"


@dataclass(frozen=True)
class PriorConfig:
    k: ParamSpec
    b: ParamSpec
    gamma: ParamSpec
    cos_theta_0: ParamSpec
    report_samples: int = 200_000


@dataclass(frozen=True)
class ModelConfig:
    hidden_nf: int = 32
    n_layers: int = 3
    index_feature: bool = True
    bond_feature: bool = True
    time_order: int = 4
    index_order: int = 4
    path: str = "linear"
    gamma: str = "sqrt"              # the LATENT SCHEDULE, not the bending constant
    gamma_scale: float = 0.2
    eps: float = 1e-6
    score_div_method: str = "hutchinson"
    n_hutchinson_probes: int = 32


@dataclass(frozen=True)
class InitConfig:
    """Warm start. `checkpoint` is a run checkpoint; `net_b`/`net_s` are the bare
    weight files the notebooks and `train.py --out` write. Distinct from `--resume`,
    which continues the same run with the config read back out of the checkpoint."""
    checkpoint: str | None = None
    net_b: str | None = None
    net_s: str | None = None


@dataclass(frozen=True)
class SampleConfig:
    n: int = 5_000
    n_steps: int = 100
    chunk: int = 500
    integrator: str = "rk4"
    sde_eps: float = 0.0
    seed: int = 4
    save_prior: bool = False


@dataclass(frozen=True)
class EntropyConfig:
    methods: tuple[str, ...] = ("dot", "zdot", "div")
    batches: int = 100               # 0 skips the batch estimates
    batch: int = 256
    align: bool = True
    batch_ot: bool = True
    seed: int = 5
    flow_n: int = 2_000              # 0 skips the running dS(t)
    flow_steps: int = 100
    flow_channels: tuple[str, ...] = ("div", "dot")
    flow_chunk: int = 500
    flow_seed: int = 6


@dataclass(frozen=True)
class RunConfig:
    name: str
    data: DataConfig
    prior: PriorConfig
    model: ModelConfig
    stages: tuple[StageConfig, ...]
    out_dir: str = "runs"
    device: str = "auto"
    dtype: str = "float64"
    seed: int = 0
    init: InitConfig = InitConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    sample: SampleConfig = SampleConfig()
    entropy: EntropyConfig = EntropyConfig()
    source: str = "<dict>"

    @property
    def entropy_channel(self) -> str | None:
        """The `entropy` setting shared by every stage -- validation guarantees one."""
        return self.stages[0].entropy

    @property
    def run_root(self) -> Path:
        return Path(self.out_dir) / self.name


_TOP_KEYS = ("name", "out_dir", "device", "dtype", "seed", "data", "prior", "model",
             "init", "train", "checkpoint", "sample", "entropy")


# --- prior parameter specs --------------------------------------------------


def _inv_var_bond(x: torch.Tensor) -> float:
    return 1.0 / bond_vectors(x).norm(dim=-1).var().item()


def _mean_bond(x: torch.Tensor) -> float:
    return bond_vectors(x).norm(dim=-1).mean().item()


def _inv_two_var_cos(x: torch.Tensor) -> float:
    return 1.0 / (2.0 * bond_cosines(x).var().item())


def _mean_cos(x: torch.Tensor) -> float:
    return bond_cosines(x).mean().item()


#: Which estimators are legal for which parameter. Keeping the table per-parameter
#: rather than global means `k: {mode: measure, estimator: mean_cos}` is caught at
#: load rather than producing a prior whose bond stiffness is an angle statistic.
PRIOR_ESTIMATORS: dict[str, dict[str, Any]] = {
    "k": {"inv_var_bond": _inv_var_bond},
    "b": {"mean_bond": _mean_bond},
    "gamma": {"inv_two_var_cos": _inv_two_var_cos},
    "cos_theta_0": {"mean_cos": _mean_cos},
}
_SOLVE_TARGETS = ("end_to_end_sq",)
_PARAM_KEYS = ("mode", "value", "estimator", "target")


def _param_spec(raw: Any, name: str, *, where: str) -> ParamSpec:
    """One prior parameter. A bare number is shorthand for {mode: fixed, value: n}."""
    at = f"{where}.{name}"
    if isinstance(raw, bool):
        raise ValueError(f"{at}: expected a number or a mapping, got {raw!r}")
    if isinstance(raw, (int, float)):
        return ParamSpec("fixed", value=float(raw))
    if isinstance(raw, str):                       # PyYAML's 1e-4 -> "1e-4"
        try:
            return ParamSpec("fixed", value=float(raw))
        except ValueError:
            raise ValueError(f"{at}: expected a number or a mapping, got {raw!r}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{at}: expected a number or a mapping, got {raw!r}")

    _closed(raw, _PARAM_KEYS, where=at)
    mode = _choice(_typed(raw, "mode", str, where=at), ("fixed", "measure", "solve"),
                   where=f"{at}.mode")

    if mode == "fixed":
        return ParamSpec("fixed", value=_typed(raw, "value", float, where=at))

    if mode == "measure":
        allowed = PRIOR_ESTIMATORS[name]
        default = next(iter(allowed))
        est = _typed(raw, "estimator", str, default, where=at)
        if est not in allowed:
            raise ValueError(f"{at}.estimator: must be one of {sorted(allowed)} for "
                             f"{name!r}, got {est!r}. Each parameter has its own "
                             f"estimators -- a bond statistic cannot set a bending "
                             f"constant.")
        return ParamSpec("measure", estimator=est)

    if name != "cos_theta_0":
        raise ValueError(
            f"{at}.mode: 'solve' is only defined for cos_theta_0, which is fitted so "
            f"the prior's E[Re^2] matches the data's (see solve_cos_theta_0). "
            f"{name!r} must be 'fixed' or 'measure'.")
    target = raw.get("target", "end_to_end_sq")
    if isinstance(target, str) and target not in _SOLVE_TARGETS:
        try:
            target = float(target)
        except ValueError:
            raise ValueError(f"{at}.target: must be a number or one of "
                             f"{list(_SOLVE_TARGETS)}, got {target!r}") from None
    elif isinstance(target, (int, float)) and not isinstance(target, bool):
        target = float(target)
    elif not isinstance(target, str):
        raise ValueError(f"{at}.target: must be a number or one of "
                         f"{list(_SOLVE_TARGETS)}, got {target!r}")
    return ParamSpec("solve", target=target)


def _prior_from_dict(raw: dict, *, where: str = "prior") -> PriorConfig:
    _closed(raw, ("k", "b", "gamma", "cos_theta_0", "report_samples"), where=where)
    for key in ("k", "b", "gamma", "cos_theta_0"):
        if key not in raw:
            raise ValueError(
                f"{where}: missing required key {key!r}. All four prior parameters must "
                f"be stated: they are properties of the polymer, and defaulting one "
                f"silently would put a different chain under the flow.")
    specs = {key: _param_spec(raw[key], key, where=where)
             for key in ("k", "b", "gamma", "cos_theta_0")}
    n_report = _typed(raw, "report_samples", int, 200_000, where=where)
    if n_report < 0:
        raise ValueError(f"{where}.report_samples: must be >= 0, got {n_report}")
    return PriorConfig(report_samples=n_report, **specs)


# --- blocks -----------------------------------------------------------------


def _data_from_dict(raw: dict, *, where: str = "data") -> DataConfig:
    _closed(raw, ("path", "n", "n_particles", "n_dims"), where=where)
    n = _optional(raw, "n", int, where=where)
    if n is not None:
        _positive(n, where=f"{where}.n")
    n_dims = _typed(raw, "n_dims", int, N_DIMS, where=where)
    if n_dims != 3:
        raise ValueError(f"{where}.n_dims: the TAP prior is 3D only (see "
                         f"eesi.systems.tap.data._require_3d), got {n_dims}")
    return DataConfig(
        path=_optional(raw, "path", str, where=where),
        n=n,
        n_particles=_positive(_typed(raw, "n_particles", int, N_DEFAULT, where=where),
                              where=f"{where}.n_particles"),
        n_dims=n_dims,
    )


def _model_from_dict(raw: dict, *, where: str = "model") -> ModelConfig:
    fields = ("hidden_nf", "n_layers", "index_feature", "bond_feature", "time_order",
              "index_order", "path", "gamma", "gamma_scale", "eps",
              "score_div_method", "n_hutchinson_probes")
    # Before `_closed`, so these two get the reason rather than the generic
    # unknown-key message: they are plausible things to reach for, and "unknown key"
    # would read as an oversight in the schema rather than a fact about the model.
    for gone, why in (("learn_score", "TAPEESI forces it False -- the subspace "
                                      "divergence runs under no_grad, so ISM is "
                                      "unavailable and setting this would be ignored"),
                      ("coords_range", "make_si_model forwards **kw to TAPEESI, not to "
                                       "TAPDynamics, so the EGNN's own kwargs are not "
                                       "reachable from a config")):
        if gone in raw:
            raise ValueError(f"{where}.{gone}: not configurable -- {why}.")
    _closed(raw, fields, where=where)

    cfg = ModelConfig(
        hidden_nf=_positive(_typed(raw, "hidden_nf", int, 32, where=where),
                            where=f"{where}.hidden_nf"),
        n_layers=_positive(_typed(raw, "n_layers", int, 3, where=where),
                           where=f"{where}.n_layers"),
        index_feature=_typed(raw, "index_feature", bool, True, where=where),
        bond_feature=_typed(raw, "bond_feature", bool, True, where=where),
        time_order=_typed(raw, "time_order", int, 4, where=where),
        index_order=_typed(raw, "index_order", int, 4, where=where),
        path=_choice(_typed(raw, "path", str, "linear", where=where), tuple(_PATHS),
                     where=f"{where}.path"),
        gamma=_choice(_typed(raw, "gamma", str, "sqrt", where=where), tuple(_GAMMAS),
                      where=f"{where}.gamma"),
        gamma_scale=_typed(raw, "gamma_scale", float, 0.2, where=where),
        eps=_positive(_typed(raw, "eps", float, 1e-6, where=where), where=f"{where}.eps"),
        score_div_method=_choice(_typed(raw, "score_div_method", str, "hutchinson",
                                        where=where), ("hutchinson", "exact"),
                                 where=f"{where}.score_div_method"),
        n_hutchinson_probes=_positive(
            _typed(raw, "n_hutchinson_probes", int, 32, where=where),
            where=f"{where}.n_hutchinson_probes"),
    )
    for name, order in (("time_order", cfg.time_order), ("index_order", cfg.index_order)):
        if order < 0:
            raise ValueError(f"{where}.{name}: must be >= 0, got {order}")
    return cfg


def _init_from_dict(raw: dict, *, where: str = "init") -> InitConfig:
    _closed(raw, ("checkpoint", "net_b", "net_s"), where=where)
    cfg = InitConfig(checkpoint=_optional(raw, "checkpoint", str, where=where),
                     net_b=_optional(raw, "net_b", str, where=where),
                     net_s=_optional(raw, "net_s", str, where=where))
    if (cfg.net_b is None) != (cfg.net_s is None):
        raise ValueError(f"{where}: net_b and net_s must be given together -- half a "
                         f"warm start is a drift field paired with a random score field.")
    if cfg.checkpoint is not None and cfg.net_b is not None:
        raise ValueError(f"{where}: give either `checkpoint` or `net_b`/`net_s`, not "
                         f"both; they are two ways to initialise the same weights.")
    return cfg


def _sample_from_dict(raw: dict, *, where: str = "sample") -> SampleConfig:
    _closed(raw, ("n", "n_steps", "chunk", "integrator", "sde_eps", "seed",
                  "save_prior"), where=where)
    cfg = SampleConfig(
        n=_positive(_typed(raw, "n", int, 5_000, where=where), where=f"{where}.n"),
        n_steps=_positive(_typed(raw, "n_steps", int, 100, where=where),
                          where=f"{where}.n_steps"),
        chunk=_positive(_typed(raw, "chunk", int, 500, where=where), where=f"{where}.chunk"),
        integrator=_choice(_typed(raw, "integrator", str, "rk4", where=where), INTEGRATORS,
                           where=f"{where}.integrator"),
        sde_eps=_typed(raw, "sde_eps", float, 0.0, where=where),
        seed=_typed(raw, "seed", int, 4, where=where),
        save_prior=_typed(raw, "save_prior", bool, False, where=where),
    )
    if cfg.integrator == "sde" and cfg.sde_eps <= 0:
        raise ValueError(f"{where}.sde_eps: integrator 'sde' needs a positive diffusion "
                         f"coefficient; at 0 it is the probability-flow ODE, which is "
                         f"what integrator 'heun' already is.")
    return cfg


def _entropy_from_dict(raw: dict, *, where: str = "entropy") -> EntropyConfig:
    _closed(raw, ("methods", "batches", "batch", "align", "batch_ot", "seed",
                  "flow_n", "flow_steps", "flow_channels", "flow_chunk", "flow_seed"),
            where=where)
    methods = _str_seq(raw, "methods", ("dot", "zdot", "div"), where=where)
    channels = _str_seq(raw, "flow_channels", ("div", "dot"), where=where)
    for m in methods:
        _choice(m, ENTROPY_METHODS, where=f"{where}.methods")
    for c in channels:
        _choice(c, FLOW_CHANNELS, where=f"{where}.flow_channels")

    cfg = EntropyConfig(
        methods=methods, flow_channels=channels,
        batches=_typed(raw, "batches", int, 100, where=where),
        batch=_positive(_typed(raw, "batch", int, 256, where=where), where=f"{where}.batch"),
        align=_typed(raw, "align", bool, True, where=where),
        batch_ot=_typed(raw, "batch_ot", bool, True, where=where),
        seed=_typed(raw, "seed", int, 5, where=where),
        flow_n=_typed(raw, "flow_n", int, 2_000, where=where),
        flow_steps=_positive(_typed(raw, "flow_steps", int, 100, where=where),
                             where=f"{where}.flow_steps"),
        flow_chunk=_positive(_typed(raw, "flow_chunk", int, 500, where=where),
                             where=f"{where}.flow_chunk"),
        flow_seed=_typed(raw, "flow_seed", int, 6, where=where),
    )
    for name, v in (("batches", cfg.batches), ("flow_n", cfg.flow_n)):
        if v < 0:
            raise ValueError(f"{where}.{name}: must be >= 0 (0 skips it), got {v}")
    if cfg.batches == 0 and cfg.flow_n == 0:
        raise ValueError(f"{where}: batches and flow_n are both 0, so the job would do "
                         f"nothing. Set one of them.")
    return cfg


def _check_entropy_against_schedule(channel: str | None, methods: Sequence[str],
                                    model: ModelConfig) -> None:
    """The gamma='none' restrictions, checked at load rather than on step 0.

    Mirrors `EESI._check_entropy_channel`: with no latent schedule there is no z, so
    the conditional score -z/gamma does not exist and 'zdot' is undefined.
    """
    if model.gamma != "none":
        return
    if channel in ("zdot", "both"):
        raise ValueError(
            f"train.entropy={channel!r} needs a latent schedule: with model.gamma='none' "
            f"the interpolant carries no latent z and the conditional score -z/gamma is "
            f"undefined. Use entropy='dot', or a non-zero model.gamma.")
    for m in methods:
        if m == "zdot":
            raise ValueError(
                f"entropy.methods includes 'zdot', which is undefined at "
                f"model.gamma='none' (no latent z). Drop it or change the schedule.")


# --- assembly ---------------------------------------------------------------


def config_from_dict(raw: dict, source: str = "<dict>") -> RunConfig:
    """Validate a parsed config into a `RunConfig`. Every error names its key."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top level must be a mapping, got {type(raw).__name__}")
    _closed(raw, _TOP_KEYS, where=source)

    model = _model_from_dict(_mapping(raw, "model", where=source))
    stages = stages_from_dict(_mapping(raw, "train", where=source))
    entropy = _entropy_from_dict(_mapping(raw, "entropy", where=source))
    _check_entropy_against_schedule(stages[0].entropy, entropy.methods, model)

    return RunConfig(
        name=_typed(raw, "name", str, where=source),
        out_dir=_typed(raw, "out_dir", str, "runs", where=source),
        device=_typed(raw, "device", str, "auto", where=source),
        dtype=_choice(_typed(raw, "dtype", str, "float64", where=source),
                      ("float64", "float32"), where=f"{source}.dtype"),
        seed=_typed(raw, "seed", int, 0, where=source),
        data=_data_from_dict(_mapping(raw, "data", where=source)),
        prior=_prior_from_dict(_mapping(raw, "prior", where=source)),
        model=model,
        stages=stages,
        init=_init_from_dict(_mapping(raw, "init", where=source)),
        checkpoint=checkpoint_from_dict(_mapping(raw, "checkpoint", where=source)),
        sample=_sample_from_dict(_mapping(raw, "sample", where=source)),
        entropy=entropy,
        source=source,
    )


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> RunConfig:
    """Read a YAML config, apply `--set` overrides, and validate the result."""
    raw = apply_overrides(load_yaml(path), overrides)
    return config_from_dict(raw, source=str(path))


def _param_to_dict(spec: ParamSpec) -> Any:
    if spec.mode == "fixed":
        return {"mode": "fixed", "value": spec.value}
    if spec.mode == "measure":
        return {"mode": "measure", "estimator": spec.estimator}
    return {"mode": "solve", "target": spec.target}


def config_to_dict(cfg: RunConfig) -> dict:
    """A fully-defaulted, YAML-able form of the config.

    Round-trips through `config_from_dict`, which is what makes a run's
    `resolved.json` a reproducibility artifact rather than a log line.
    """
    return {
        "name": cfg.name, "out_dir": cfg.out_dir, "device": cfg.device,
        "dtype": cfg.dtype, "seed": cfg.seed,
        "data": {"path": cfg.data.path, "n": cfg.data.n,
                 "n_particles": cfg.data.n_particles, "n_dims": cfg.data.n_dims},
        "prior": {**{k: _param_to_dict(getattr(cfg.prior, k))
                     for k in ("k", "b", "gamma", "cos_theta_0")},
                  "report_samples": cfg.prior.report_samples},
        "model": {"hidden_nf": cfg.model.hidden_nf, "n_layers": cfg.model.n_layers,
                  "index_feature": cfg.model.index_feature,
                  "bond_feature": cfg.model.bond_feature,
                  "time_order": cfg.model.time_order,
                  "index_order": cfg.model.index_order, "path": cfg.model.path,
                  "gamma": cfg.model.gamma, "gamma_scale": cfg.model.gamma_scale,
                  "eps": cfg.model.eps,
                  "score_div_method": cfg.model.score_div_method,
                  "n_hutchinson_probes": cfg.model.n_hutchinson_probes},
        "init": {"checkpoint": cfg.init.checkpoint, "net_b": cfg.init.net_b,
                 "net_s": cfg.init.net_s},
        "train": {"stages": [stage_to_dict(s) for s in cfg.stages]},
        "checkpoint": checkpoint_to_dict(cfg.checkpoint),
        "sample": {"n": cfg.sample.n, "n_steps": cfg.sample.n_steps,
                   "chunk": cfg.sample.chunk, "integrator": cfg.sample.integrator,
                   "sde_eps": cfg.sample.sde_eps, "seed": cfg.sample.seed,
                   "save_prior": cfg.sample.save_prior},
        "entropy": {"methods": list(cfg.entropy.methods), "batches": cfg.entropy.batches,
                    "batch": cfg.entropy.batch, "align": cfg.entropy.align,
                    "batch_ot": cfg.entropy.batch_ot, "seed": cfg.entropy.seed,
                    "flow_n": cfg.entropy.flow_n, "flow_steps": cfg.entropy.flow_steps,
                    "flow_channels": list(cfg.entropy.flow_channels),
                    "flow_chunk": cfg.entropy.flow_chunk,
                    "flow_seed": cfg.entropy.flow_seed},
    }


# --- resolving the prior against the data -----------------------------------


@dataclass(frozen=True)
class ResolvedPrior:
    """The four numbers `sample_prior` needs, plus where each one came from."""
    k: float
    b: float
    gamma: float
    cos_theta_0: float
    provenance: dict[str, str]

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.k, self.b, self.gamma, self.cos_theta_0)

    def to_dict(self) -> dict:
        return {"k": self.k, "b": self.b, "gamma": self.gamma,
                "cos_theta_0": self.cos_theta_0, "provenance": dict(self.provenance)}

    @classmethod
    def from_dict(cls, raw: dict) -> "ResolvedPrior":
        return cls(k=float(raw["k"]), b=float(raw["b"]), gamma=float(raw["gamma"]),
                   cos_theta_0=float(raw["cos_theta_0"]),
                   provenance=dict(raw.get("provenance", {})))


def resolve_prior(cfg: PriorConfig, x1: torch.Tensor,
                  n_particles: int = N_DEFAULT) -> ResolvedPrior:
    """Turn the four `ParamSpec`s into four numbers, against the target data.

    Order matters: `cos_theta_0` is resolved last, because `mode: solve` bisects on
    the already-resolved `k`, `b` and `gamma`. `solve_cos_theta_0`'s own ValueError --
    the target E[Re^2] being unreachable at this gamma -- is let through unwrapped,
    since it already says the physical thing to do about it.
    """
    values: dict[str, float] = {}
    provenance: dict[str, str] = {}

    for name in ("k", "b", "gamma"):
        spec: ParamSpec = getattr(cfg, name)
        if spec.mode == "fixed":
            values[name] = float(spec.value)
            provenance[name] = "fixed"
        else:
            values[name] = float(PRIOR_ESTIMATORS[name][spec.estimator](x1))
            provenance[name] = f"measure:{spec.estimator}"

    spec = cfg.cos_theta_0
    if spec.mode == "fixed":
        values["cos_theta_0"] = float(spec.value)
        provenance["cos_theta_0"] = "fixed"
    elif spec.mode == "measure":
        values["cos_theta_0"] = float(PRIOR_ESTIMATORS["cos_theta_0"][spec.estimator](x1))
        provenance["cos_theta_0"] = f"measure:{spec.estimator}"
    else:
        target = (end_to_end_sq(x1).mean().item() if spec.target == "end_to_end_sq"
                  else float(spec.target))
        values["cos_theta_0"] = solve_cos_theta_0(
            values["k"], values["b"], values["gamma"], target, n_particles)
        label = spec.target if isinstance(spec.target, str) else "value"
        provenance["cos_theta_0"] = f"solve:{label}={target:.6g}"

    if not -1.0 <= values["cos_theta_0"] <= 1.0:
        raise ValueError(f"cos_theta_0 resolved to {values['cos_theta_0']}, which is not "
                         f"a cosine. Provenance: {provenance['cos_theta_0']}")
    return ResolvedPrior(provenance=provenance, **values)


def prior_report(prior: ResolvedPrior, x1: torch.Tensor, n_particles: int = N_DEFAULT,
                 n_samples: int = 200_000, generator: torch.Generator | None = None) -> str:
    """The calibration table the notebook prints: prior vs data vs analytic.

    `n_samples=0` skips the Monte-Carlo row, leaving the analytic values and the data's,
    which is what a resume wants -- the numbers are already in the log from the first
    launch, and a 200k-chain draw is not free.
    """
    k, b, gamma, cos0 = prior.as_tuple()
    lines = [
        f"prior  k = {k:.4f}   b = {b:.4f}   gamma = {gamma:.4f}   "
        f"cos_theta_0 = {cos0:.4f}",
        "  provenance: " + ", ".join(f"{key}={val}"
                                     for key, val in prior.provenance.items()),
        f"  prior <cos theta> = {angle_moments(gamma, cos0)[1]:+.4f}   "
        f"(data {bond_cosines(x1).mean().item():+.4f})",
        f"  S[p0] (analytic)  = {prior_entropy(k, b, gamma, cos0, n_particles):+.4f} nats",
        f"{'':>12}{'E[Re^2]':>10}{'E[Rg^2]':>10}{'E[|b|^2]':>10}",
    ]
    if n_samples > 0:
        x0 = sample_prior(n_samples, k, b, gamma, cos0, n_particles=n_particles,
                          dtype=x1.dtype, generator=generator)
        lines.append(f"{'prior':<12}{end_to_end_sq(x0).mean().item():10.4f}"
                     f"{gyration_sq(x0).mean().item():10.4f}"
                     f"{bond_vectors(x0).pow(2).sum(-1).mean().item():10.4f}")
    lines.append(f"{'data':<12}{end_to_end_sq(x1).mean().item():10.4f}"
                 f"{gyration_sq(x1).mean().item():10.4f}"
                 f"{bond_vectors(x1).pow(2).sum(-1).mean().item():10.4f}")
    lines.append(f"{'analytic':<12}"
                 f"{end_to_end_mean_sq(k, b, gamma, cos0, n_particles):10.4f}"
                 f"{'':>10}{'':>10}")
    return "\n".join(lines)
