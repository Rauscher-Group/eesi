"""Config-driven TAP runs: `python -m eesi.systems.tap.run train --config <file>`.

The entry point for training runs long enough that a notebook cannot hold them --
100k steps and up, launched detached, checkpointed as they go, and resumable after a
crash or a deliberate stop. Everything a run needs is in one YAML file
(`eesi.systems.tap.config`), and everything a run produced is in one directory
(`eesi.rundir`).

    # start, into runs/<name>/
    python -m eesi.systems.tap.run train --config experiments/TAP/configs/tap_N20_Pe0.yaml

    # somewhere else, with two settings changed on the command line
    python -m eesi.systems.tap.run train --config experiments/TAP/configs/tap_N20_Pe0.yaml \\
        --run runs/tap_long --set device=cuda:1 --set train.stages.1.steps=250000

    # continue after a stop -- the config is read back out of the checkpoint
    python -m eesi.systems.tap.run train --run runs/tap_long --resume

Under tmux, which is the point of all this:

    tmux new-session -d -s tap-train \\
      'python -m eesi.systems.tap.run train \\
           --config experiments/TAP/configs/tap_N20_Pe0.yaml --run runs/tap_long'
    tail -f runs/tap_long/log.txt        # follow it from anywhere
    tmux send-keys -t tap-train C-c      # stop: the step finishes, a checkpoint lands
    tmux new-session -d -s tap-train \\
      'python -m eesi.systems.tap.run train --run runs/tap_long --resume'

The training arithmetic is not here. This module builds the model, resolves the prior,
and calls `eesi.systems.tap.train.train_si` once per stage, passing a callback that
does the logging, the history file, and the checkpoints. There is one training loop in
the package and this is not it.

`python -m eesi.systems.tap.train` still exists and is unchanged: it is the
single-shot path for smoke tests and the coupling ablation arms. Use this module when
a run needs to survive longer than the terminal it was started in.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ...config import resolve_device, resolve_dtype
from ...ot import transport_cost
from ...rundir import (Logger, RunDir, append_history, env_info, history_header,
                       load_checkpoint, load_history, rng_state, save_checkpoint,
                       set_rng_state, split_nets, truncate_history)
from .config import (DataConfig, ModelConfig, ResolvedPrior, RunConfig, config_from_dict,
                     config_to_dict, load_config, prior_report, resolve_prior)
from .data import (bond_cosines, end_to_end_sq, gyration_sq, load_ref_data,
                   prior_entropy, sample_prior)
from .dynamics import rk4_sample
from .interpolant import TAPEESI
from .ot import tap_ot_couple
from .train import _HIST_LABELS, _hist_keys, make_si_model, train_si

__all__ = ["build_model", "load_data", "model_kwargs", "load_run", "Run",
           "job_train", "job_sample", "job_entropy", "job_export", "job_status",
           "main"]


# --- building the pieces a run needs ----------------------------------------


def model_kwargs(cfg: ModelConfig, n_particles: int, n_dims: int) -> dict:
    """Exactly the arguments `make_si_model` is called with, recorded in the checkpoint.

    Storing these rather than reconstructing them from the config means a checkpoint
    can be reopened even if the schema later grows a field with a different default.
    """
    return {"n_particles": n_particles, "n_dims": n_dims,
            "hidden_nf": cfg.hidden_nf, "n_layers": cfg.n_layers,
            "index_feature": cfg.index_feature, "bond_feature": cfg.bond_feature,
            "time_order": cfg.time_order, "index_order": cfg.index_order,
            "path": cfg.path, "gamma": cfg.gamma, "gamma_scale": cfg.gamma_scale,
            "eps": cfg.eps, "score_div_method": cfg.score_div_method,
            "n_hutchinson_probes": cfg.n_hutchinson_probes}


def build_model(cfg: ModelConfig, n_particles: int, n_dims: int) -> TAPEESI:
    """A TAPEESI from the config's model block. `learn_score` is left alone --
    `TAPEESI` forces it False, which is why the config does not offer it."""
    return make_si_model(**model_kwargs(cfg, n_particles, n_dims))


def load_data(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return load_ref_data(cfg.resolved_path, cfg.n, n_particles=cfg.n_particles,
                         n_dims=cfg.n_dims, dtype=dtype).to(device)


def _apply_init(cfg: RunConfig, model: TAPEESI, log: Logger) -> None:
    """Warm start from a previous run's checkpoint, or from bare per-net weight files.

    Both are a starting point for a NEW run: the step counters begin at zero and the
    config is this run's own. Continuing the SAME run is `--resume`, which is a
    different thing and reads its config back out of the checkpoint.
    """
    init = cfg.init
    if init.checkpoint:
        ckpt = load_checkpoint(init.checkpoint)
        model.load_state_dict(ckpt["model"])
        log(f"warm start from {init.checkpoint} "
            f"(global_step {ckpt.get('global_step')} of run {ckpt.get('config', {}).get('name')})")
    elif init.net_b:
        model.net_b.load_state_dict(torch.load(init.net_b, map_location="cpu",
                                               weights_only=True))
        model.net_s.load_state_dict(torch.load(init.net_s, map_location="cpu",
                                               weights_only=True))
        log(f"warm start from {init.net_b} and {init.net_s}")


def export_nets(rd: RunDir, model: TAPEESI, n_particles: int, batch: int) -> tuple[Path, Path]:
    """Write the two bare per-network weight files the notebooks have always used.

    `tap_{b,s}_si_N{N}_B{batch}.pth`, loadable with
    `TAPDynamics(...).load_state_dict(torch.load(path))`. The run's own checkpoint
    holds all of this and more; these exist so the notebook's existing load cell keeps
    working against a run directory.
    """
    net_b, net_s = split_nets(model.state_dict())
    paths = []
    for name, sd in (("b", net_b), ("s", net_s)):
        p = rd.nets / f"tap_{name}_si_N{n_particles}_B{batch}.pth"
        torch.save(sd, p)
        paths.append(p)
    return tuple(paths)


# --- the per-step callback --------------------------------------------------


class _StageWriter:
    """What `train_si` calls after every optimizer step: log, buffer, checkpoint.

    History rows are buffered and written to disk just before the checkpoint that
    counts them, so the number a checkpoint records is never larger than what is
    actually on disk. If a crash lands between the two, the resume finds extra rows
    and drops them -- which is recoverable. The other order is not.
    """

    def __init__(self, rd: RunDir, cfg: RunConfig, stage, stage_index: int,
                 global_base: int, model: TAPEESI, opt, prior: ResolvedPrior,
                 log: Logger, stop, hist_keys: Sequence[str]):
        self.rd, self.cfg, self.stage, self.si = rd, cfg, stage, stage_index
        self.global_base = global_base
        self.model, self.opt, self.prior = model, opt, prior
        self.log, self.stop = log, stop
        self.hist_keys = tuple(hist_keys)
        self.header = history_header([_HIST_LABELS[k] for k in self.hist_keys])
        self.buf: list[tuple] = []
        self.t0 = time.perf_counter()
        self.last_step = -1

    def __call__(self, step: int, row: tuple, x0: torch.Tensor, x1: torch.Tensor) -> bool:
        self.last_step = step
        self.buf.append((self.global_base + step, self.stage.name, step, *row,
                         transport_cost(x0, x1).item(),
                         round(time.perf_counter() - self.t0, 3)))

        if self.stage.log_every and (step % self.stage.log_every == 0
                                     or step == self.stage.steps - 1):
            self.log(self._line(step))
        if (step + 1) % self.stage.ckpt_every == 0:
            self.flush(step + 1)
        return not self.stop.requested       # False stops the loop, cleanly

    def _line(self, step: int) -> str:
        recent = self.buf[-self.stage.log_every:] if self.stage.log_every else self.buf
        n_fixed = 3                                   # global_step, stage, stage_step
        means = [sum(r[n_fixed + i] for r in recent) / len(recent)
                 for i in range(len(self.hist_keys) + 1)]   # + transport
        cols = "  ".join(f"{_HIST_LABELS[key]} {m:9.4f}"
                         for key, m in zip(self.hist_keys, means))
        return (f"  [{self.stage.name}] step {step:7d}  {cols}  "
                f"transport {means[-1]:7.3f}  ({time.perf_counter() - self.t0:6.1f}s)")

    def flush(self, stage_step: int, *, final: bool = False) -> None:
        """Write the buffered history rows, then a checkpoint that counts them."""
        if self.buf:
            append_history(self.rd.history_csv, self.buf, header=self.header)
            self.buf.clear()
        rows = self.global_base + stage_step
        payload = _checkpoint_payload(self.cfg, self.model, self.opt, self.prior,
                                      stage_index=self.si, stage_name=self.stage.name,
                                      stage_step=stage_step, global_step=rows,
                                      hist_keys=self.hist_keys, history_rows=rows)
        save_checkpoint(self.rd.checkpoints / "latest.pt", **payload)

        name = (f"stage{self.si}_final.pt" if final
                else f"stage{self.si}_step{rows:08d}.pt")
        shutil.copyfile(self.rd.checkpoints / "latest.pt", self.rd.checkpoints / name)
        self.rd.prune_checkpoints(self.cfg.checkpoint.keep)

        if self.cfg.checkpoint.export_nets:
            export_nets(self.rd, self.model, self.cfg.data.n_particles, self.stage.batch)


class _Stop:
    """SIGINT/SIGTERM set a flag; the loop finishes its step and stops on its own.

    Tearing down mid-step would leave the optimizer and the model out of step with
    each other and the checkpoint inconsistent. A second signal is left to the default
    handler, so an impatient Ctrl-C twice still kills the process.
    """

    def __init__(self, log: Logger):
        self.requested = False
        self.log = log
        self._previous: dict = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass                       # not on the main thread; not fatal
        return self

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        return False

    def _handle(self, signum, frame):
        self.requested = True
        signal.signal(signum, self._previous.get(signum, signal.SIG_DFL))
        self.log(f"\ncaught {signal.Signals(signum).name}: finishing this step, writing a "
                 f"checkpoint, then stopping. Signal again to kill it outright.")


# --- checkpoints ------------------------------------------------------------


def _checkpoint_payload(cfg: RunConfig, model: TAPEESI, opt, prior: ResolvedPrior,
                        **fields: Any) -> dict:
    return {"config": config_to_dict(cfg), "config_source": cfg.source,
            "prior": prior.to_dict(),
            "model_kwargs": model_kwargs(cfg.model, cfg.data.n_particles, cfg.data.n_dims),
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            # Without this a resume would restart the random stream, so the replayed
            # steps would draw different batches and different latents.
            "rng": rng_state(),
            "env": env_info(Path(__file__).resolve().parents[3]), **fields}


def _adam(model: TAPEESI, lr: float, state: dict | None) -> torch.optim.Optimizer:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if state is not None:
        opt.load_state_dict(state)
        for group in opt.param_groups:
            group["lr"] = lr               # the stage's lr wins over the stored one
    return opt


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def _start_position(ckpt: dict, cfg: RunConfig) -> tuple[int, int]:
    """Where a resume picks up: (stage index, step within it).

    A checkpoint written at a stage boundary records that stage as complete, so the
    resume moves on to the next one rather than repeating the last step.
    """
    si, step = int(ckpt["stage_index"]), int(ckpt["stage_step"])
    if si < len(cfg.stages) and step >= cfg.stages[si].steps:
        si, step = si + 1, 0
    return si, step


def _warn_on_prior_drift(cfg: RunConfig, stored: ResolvedPrior, x1: torch.Tensor,
                         log: Logger, strict: bool) -> None:
    """The checkpoint's prior always wins on resume. Say so if the config disagrees.

    Re-resolving would quietly put a different base distribution under a
    half-trained flow -- the parameters would no longer be the ones the network was
    trained against, and nothing downstream would notice.
    """
    try:
        fresh = resolve_prior(cfg.prior, x1, cfg.data.n_particles)
    except Exception as exc:                # a config edit that no longer resolves
        log(f"  note: the config's prior block no longer resolves ({exc}); "
            f"the checkpoint's prior is being used.")
        return
    diffs = [f"{k}: {getattr(stored, k):.6g} -> {getattr(fresh, k):.6g}"
             for k in ("k", "b", "gamma", "cos_theta_0")
             if abs(getattr(stored, k) - getattr(fresh, k)) > 1e-9]
    if not diffs:
        return
    msg = ("the config's prior no longer matches the one this run was trained with: "
           + "; ".join(diffs) + ". The checkpoint's prior is being used.")
    if strict:
        raise ValueError(msg + " (--strict-prior)")
    log(f"  WARNING: {msg}")


# --- the train job ----------------------------------------------------------


def job_train(cfg: RunConfig, rd: RunDir, log: Logger, *, resume: bool = False,
              checkpoint: str | Path | None = None,
              strict_prior: bool = False) -> None:
    """Run every stage of `cfg`, checkpointing as it goes. Resumable and interruptible.

    `checkpoint` picks a specific one to resume from; the default is `latest.pt`.
    """
    device, dtype = resolve_device(cfg.device), resolve_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)
    ckpt = None
    if resume:
        path = Path(checkpoint) if checkpoint else rd.latest()
        if path is None:
            raise FileNotFoundError(f"{rd.root} has no checkpoint to resume from")
        ckpt = load_checkpoint(path, map_location=device)

    log(f"{'resuming' if resume else 'starting'} run {cfg.name!r} in {rd.root}")
    log(f"  device={device}  dtype={cfg.dtype}  config={cfg.source}")

    x1 = load_data(cfg.data, device, dtype)
    log(f"  data {tuple(x1.shape)} from {cfg.data.resolved_path}")

    if ckpt is None:
        torch.manual_seed(cfg.seed)
        prior = resolve_prior(cfg.prior, x1, cfg.data.n_particles)
        log(prior_report(prior, x1, cfg.data.n_particles, cfg.prior.report_samples))
        model = build_model(cfg.model, cfg.data.n_particles, cfg.data.n_dims)
        model = model.to(device=device, dtype=dtype)
        _apply_init(cfg, model, log)
        start_stage, start_step = 0, 0
    else:
        prior = ResolvedPrior.from_dict(ckpt["prior"])
        log(prior_report(prior, x1, cfg.data.n_particles, n_samples=0))
        _warn_on_prior_drift(cfg, prior, x1, log, strict_prior)
        model = build_model(cfg.model, cfg.data.n_particles, cfg.data.n_dims)
        model = model.to(device=device, dtype=dtype)
        model.load_state_dict(ckpt["model"])
        start_stage, start_step = _start_position(ckpt, cfg)
        dropped = truncate_history(rd.history_csv, int(ckpt["history_rows"]))
        if dropped:
            log(f"  dropped {dropped} history row(s) past the checkpoint; "
                f"those steps are about to be run again")
        set_rng_state(ckpt["rng"])

    log(f"  model: {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"in_node_nf = {model.net_b.egnn.embedding.in_features}")

    if start_stage >= len(cfg.stages):
        log(f"  every stage is already complete ({ckpt['global_step']} steps). "
            f"Add a stage to the config to train further.")
        return

    hist_keys = _hist_keys(cfg.entropy_channel)
    _write_resolved(rd, cfg, prior, model, device)
    opt = None

    with _Stop(log) as stop:
        for si, stage in enumerate(cfg.stages[start_stage:], start=start_stage):
            step0 = start_step if si == start_stage else 0
            global_base = sum(s.steps for s in cfg.stages[:si])
            mid_stage = ckpt is not None and si == start_stage and step0 > 0
            carry = mid_stage or not stage.reset_optimizer

            if ckpt is not None and si == start_stage and carry:
                opt = _adam(model, stage.lr, ckpt.get("optimizer"))
            elif opt is not None and carry:
                _set_lr(opt, stage.lr)
            else:
                opt = _adam(model, stage.lr, None)

            log(f"\nstage {si} {stage.name!r}: {stage.steps - step0:,} steps "
                f"(of {stage.steps:,}) at lr {stage.lr:g}, batch {stage.batch}, "
                f"entropy={stage.entropy}"
                + (f", resuming at {step0:,}" if step0 else ""))

            writer = _StageWriter(rd, cfg, stage, si, global_base, model, opt, prior,
                                  log, stop, hist_keys)
            train_si_kwargs = dict(
                model=model, opt=opt, steps=stage.steps, start_step=step0,
                batch=stage.batch, lr=stage.lr, align=stage.align,
                batch_ot=stage.batch_ot, entropy=stage.entropy,
                device=str(device), dtype=dtype, log_every=0, callback=writer,
                seed=None if mid_stage else (stage.seed if stage.seed is not None
                                             else cfg.seed))
            train_si(x1, *prior.as_tuple(), **train_si_kwargs)

            # last_step stays -1 only if the loop body never ran, in which case the
            # stage is already where the checkpoint said it was.
            reached = writer.last_step + 1 if writer.last_step >= 0 else step0
            writer.flush(reached, final=reached >= stage.steps)
            if stop.requested:
                log(f"stopped in stage {si} at step {reached:,}; checkpoint written. "
                    f"Resume with:\n"
                    f"  python -m eesi.systems.tap.run train --run {rd.root} --resume")
                return
            log(f"stage {si} {stage.name!r} complete at {global_base + reached:,} "
                f"total steps")
            ckpt = None

    log(f"\nrun {cfg.name!r} finished: {sum(s.steps for s in cfg.stages):,} steps")
    log(f"  history   {rd.history_csv}")
    log(f"  weights   {rd.checkpoints / 'latest.pt'}")
    if cfg.checkpoint.export_nets:
        log(f"  nets      {rd.nets}/tap_{{b,s}}_si_N{cfg.data.n_particles}_"
            f"B{cfg.stages[-1].batch}.pth")


def _write_resolved(rd: RunDir, cfg: RunConfig, prior: ResolvedPrior, model: TAPEESI,
                    device: torch.device) -> None:
    """The record of what actually ran, as opposed to what the config asked for."""
    rd.resolved_json.write_text(json.dumps({
        "config": config_to_dict(cfg),
        "config_source": cfg.source,
        "prior": prior.to_dict(),
        "data_path": str(cfg.data.resolved_path),
        "model_kwargs": model_kwargs(cfg.model, cfg.data.n_particles, cfg.data.n_dims),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "env": env_info(Path(__file__).resolve().parents[3]),
    }, indent=2, default=str) + "\n")


# --- reopening a finished run -----------------------------------------------


@dataclass
class Run:
    """A trained run, reopened: its config, its prior, its model, its artifacts.

    This is the notebook's one import. It replaces the cells that rebuilt the model
    and re-derived the prior by hand, which is where a notebook most easily drifts out
    of step with the weights it is analysing -- re-running the calibration cell against
    a differently-sliced dataset gives a prior the network was never trained against,
    and nothing says so.

        from eesi.systems.tap.run import load_run
        run = load_run("runs/tap_long")
        K, B, GAMMA, COS0 = run.prior.as_tuple()
        h = run.history()
        means_div = run.artifact("entropy", "means_div.npy")
    """
    dir: RunDir
    config: RunConfig
    prior: ResolvedPrior
    model: TAPEESI
    ckpt: dict
    device: torch.device
    dtype: torch.dtype

    def history(self) -> dict[str, np.ndarray]:
        return load_history(self.dir.history_csv)

    def artifact(self, *parts: str) -> np.ndarray:
        return np.load(self.dir.root.joinpath(*parts))

    def data(self) -> torch.Tensor:
        """The target configurations this run was trained against."""
        return load_data(self.config.data, self.device, self.dtype)

    @property
    def global_step(self) -> int:
        return int(self.ckpt.get("global_step", 0))


def _checkpoint_path(rd: RunDir, checkpoint: str | Path = "latest") -> Path:
    if checkpoint in ("latest", None):
        path = rd.latest()
        if path is None:
            raise FileNotFoundError(f"{rd.root} holds no checkpoint yet")
        return path
    path = Path(checkpoint)
    return path if path.is_absolute() or path.exists() else rd.root / path


def load_run(root: str | Path, checkpoint: str | Path = "latest",
             device: str = "cpu") -> Run:
    """Reopen a run directory: config, prior and model, ready to use."""
    rd = RunDir(Path(root))
    ckpt = load_checkpoint(_checkpoint_path(rd, checkpoint), map_location=device)
    cfg = config_from_dict(ckpt["config"], source=ckpt.get("config_source", str(root)))
    dev, dtype = torch.device(device), resolve_dtype(cfg.dtype)
    model = build_model(cfg.model, cfg.data.n_particles, cfg.data.n_dims)
    model = model.to(device=dev, dtype=dtype)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return Run(dir=rd, config=cfg, prior=ResolvedPrior.from_dict(ckpt["prior"]),
               model=model, ckpt=ckpt, device=dev, dtype=dtype)


def _job_meta(run: Run, **fields: Any) -> dict:
    """Stamped beside every job's arrays, so a plotted number can be traced back to
    the weights and the seed that produced it."""
    return {"run": run.config.name, "global_step": run.global_step,
            "checkpoint_created": run.ckpt.get("created"),
            "prior": run.prior.to_dict(), **fields}


def _chunked(fn, x: torch.Tensor, size: int) -> torch.Tensor:
    return torch.cat([fn(c) for c in x.split(size)])


# --- sampling ---------------------------------------------------------------


@torch.no_grad()
def job_sample(run: Run, log: Logger, **overrides: Any) -> Path:
    """Generate configurations by integrating the learned drift from prior draws.

    Writes `samples/x_gen_g<step>_n<n>.npy` and logs the structural comparison the
    notebook makes: the prior it started from, what came out, and the reference data.
    Those three lines are the model-free check on the network -- TAP has no target
    potential, so there is no free-energy cross-check to fall back on.
    """
    cfg = replace(run.config.sample, **{k: v for k, v in overrides.items()
                                       if v is not None})
    torch.manual_seed(cfg.seed)
    N = run.config.data.n_particles
    x0 = sample_prior(cfg.n, *run.prior.as_tuple(), n_particles=N,
                      n_dims=run.config.data.n_dims, dtype=run.dtype, device=run.device)

    log(f"sampling {cfg.n:,} configurations with {cfg.integrator} "
        f"({cfg.n_steps} steps, chunk {cfg.chunk}) from checkpoint at "
        f"step {run.global_step:,}")
    t0 = time.perf_counter()
    if cfg.integrator == "rk4":
        x_gen = _chunked(lambda c: rk4_sample(run.model.net_b, c, n_steps=cfg.n_steps),
                         x0, cfg.chunk)
    else:
        eps = cfg.sde_eps if cfg.integrator == "sde" else 0.0
        method = "euler" if cfg.integrator in ("euler", "sde") else "heun"
        x_gen = _chunked(
            lambda c: run.model.sample(c, n_steps=cfg.n_steps, eps=eps, method=method),
            x0, cfg.chunk)
    log(f"  {time.perf_counter() - t0:.1f}s")

    x1 = run.data()
    log(f"{'':>12}{'E[Re^2]':>10}{'E[Rg^2]':>10}{'<cos th>':>10}")
    for name, x in (("prior", x0), ("generated", x_gen), ("reference", x1[:cfg.n])):
        log(f"{name:<12}{end_to_end_sq(x).mean().item():10.3f}"
            f"{gyration_sq(x).mean().item():10.3f}"
            f"{bond_cosines(x).mean().item():10.3f}")

    out = run.dir.samples / f"x_gen_g{run.global_step}_n{cfg.n}.npy"
    np.save(out, x_gen.cpu().numpy())
    if cfg.save_prior:
        np.save(run.dir.samples / f"x0_g{run.global_step}_n{cfg.n}.npy", x0.cpu().numpy())
    (run.dir.samples / "meta.json").write_text(json.dumps(_job_meta(
        run, job="sample", n=cfg.n, n_steps=cfg.n_steps, integrator=cfg.integrator,
        sde_eps=cfg.sde_eps, seed=cfg.seed, file=out.name,
        seconds=round(time.perf_counter() - t0, 1)), indent=2, default=str) + "\n")
    log(f"  wrote {out}")
    return out


# --- entropy ----------------------------------------------------------------


def _ci(values: Sequence[float]) -> float:
    """A 95% confidence interval on the mean of independent batch estimates."""
    return 1.96 * float(np.std(values)) / max(1.0, float(np.sqrt(len(values))))


@torch.no_grad()
def job_entropy(run: Run, log: Logger, **overrides: Any) -> Path:
    """Entropy estimates: per-batch numbers, and the running dS(t) along the flow.

    Two independent things, both written into `entropy/`, either of which can be
    switched off with `batches: 0` or `flow_n: 0`:

      * `means_<method>.npy` -- one estimate per batch from `entropy_estimate`, which
        samples the interpolant directly rather than integrating it. Their mean is
        the entropy difference; their spread is its error bar.
      * `ent_c_<channel>.npy` -- the entropy accumulated along integrated
        trajectories, shape (flow_steps+1, flow_n), for the dS-versus-t figure.

    Rows are written as they are computed, so a job killed part way through still
    leaves usable data behind.
    """
    cfg = replace(run.config.entropy, **{k: v for k, v in overrides.items()
                                         if v is not None})
    N = run.config.data.n_particles
    s_prior = prior_entropy(*run.prior.as_tuple(), N)
    summary: dict[str, Any] = _job_meta(run, job="entropy", S_prior=s_prior)

    if cfg.batches:
        torch.manual_seed(cfg.seed)
        x1 = run.data()
        means = {m: [] for m in cfg.methods}
        log(f"entropy: {cfg.batches} batches of {cfg.batch}, methods "
            f"{', '.join(cfg.methods)}")
        t0 = time.perf_counter()
        with (run.dir.entropy / "means.csv").open("w") as fh:
            fh.write("batch," + ",".join(cfg.methods) + "\n")
            for i in range(cfg.batches):
                idx = torch.randint(0, x1.shape[0], (cfg.batch,), device=run.device)
                xa = sample_prior(cfg.batch, *run.prior.as_tuple(), n_particles=N,
                                  n_dims=run.config.data.n_dims, dtype=run.dtype,
                                  device=run.device)
                # entropy_estimate takes (x1, x0): data first, prior second.
                xa, xb = tap_ot_couple(xa, x1[idx], align=cfg.align, batch=cfg.batch_ot)
                row = [run.model.entropy_estimate(xb, xa, method=m).mean().item()
                       for m in cfg.methods]
                for m, v in zip(cfg.methods, row):
                    means[m].append(v)
                fh.write(f"{i}," + ",".join(f"{v:.10g}" for v in row) + "\n")
                fh.flush()
                if (i + 1) % max(1, cfg.batches // 10) == 0:
                    log(f"  batch {i + 1:5d}/{cfg.batches}  "
                        + "  ".join(f"{m} {np.mean(means[m]):+8.3f}" for m in cfg.methods)
                        + f"  ({time.perf_counter() - t0:5.1f}s)")

        summary["batches"] = {}
        for m, values in means.items():
            np.save(run.dir.entropy / f"means_{m}.npy", np.asarray(values))
            mean, ci = float(np.mean(values)), _ci(values)
            summary["batches"][m] = {"dS": mean, "ci95": ci, "n": len(values),
                                     "S_target": s_prior + mean}
            log(f"  dS ({m:5s}) {mean:+8.3f} +/- {ci:.3f}    "
                f"S[p1] = {s_prior + mean:+8.3f} nats")
        summary["config"] = {"batches": cfg.batches, "batch": cfg.batch,
                             "methods": list(cfg.methods), "seed": cfg.seed,
                             "align": cfg.align, "batch_ot": cfg.batch_ot}

    if cfg.flow_n:
        torch.manual_seed(cfg.flow_seed)
        x0 = sample_prior(cfg.flow_n, *run.prior.as_tuple(), n_particles=N,
                          n_dims=run.config.data.n_dims, dtype=run.dtype,
                          device=run.device)
        log(f"running dS(t): {cfg.flow_n:,} trajectories, {cfg.flow_steps} steps, "
            f"channels {', '.join(cfg.flow_channels)}")
        for channel in cfg.flow_channels:
            t0 = time.perf_counter()
            pieces, grid = [], None
            for chunk in x0.split(cfg.flow_chunk):
                _, ent_traj, grid = run.model.sample(chunk, n_steps=cfg.flow_steps,
                                                     return_traj=True, entropy=channel)
                pieces.append(ent_traj.cpu())
            ent = torch.cat(pieces, dim=1)
            np.save(run.dir.entropy / f"ent_c_{channel}.npy", ent.numpy())
            summary.setdefault("flow", {})[channel] = {
                "dS": float(ent[-1].mean()), "ci95": _ci(ent[-1].numpy()),
                "n": cfg.flow_n, "steps": cfg.flow_steps}
            log(f"  {channel}: dS(1) = {ent[-1].mean().item():+8.3f}  "
                f"({time.perf_counter() - t0:5.1f}s)")
        np.save(run.dir.entropy / "ts.npy", grid.cpu().numpy())

    out = run.dir.entropy / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log(f"  wrote {out}")
    return out


# --- export and status ------------------------------------------------------


def job_export(run: Run, out: str | Path | None = None) -> tuple[Path, Path]:
    """Write the two bare per-network weight files, optionally somewhere else."""
    if out is None:
        return export_nets(run.dir, run.model, run.config.data.n_particles,
                           run.config.stages[-1].batch)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    net_b, net_s = split_nets(run.model.state_dict())
    N, batch = run.config.data.n_particles, run.config.stages[-1].batch
    paths = []
    for name, sd in (("b", net_b), ("s", net_s)):
        p = dest / f"tap_{name}_si_N{N}_B{batch}.pth"
        torch.save(sd, p)
        paths.append(p)
    return tuple(paths)


def job_status(root: str | Path, log: Logger | None = None) -> dict:
    """Where a run has got to. Cheap enough to poll from a second tmux pane."""
    emit = log or print
    rd = RunDir(Path(root))
    path = rd.latest()
    if path is None:
        emit(f"{rd.root}: no checkpoint yet")
        return {}
    ckpt = load_checkpoint(path)
    cfg = config_from_dict(ckpt["config"], source=str(root))
    total = sum(s.steps for s in cfg.stages)
    step = int(ckpt["global_step"])
    emit(f"{cfg.name}  {step:,}/{total:,} steps ({100 * step / total:.1f}%)  "
         f"stage {ckpt['stage_index']} {ckpt['stage_name']!r} "
         f"at {ckpt['stage_step']:,}/{cfg.stages[int(ckpt['stage_index'])].steps:,}")
    emit(f"  checkpoint written {ckpt.get('created')}  ({path})")
    if rd.history_csv.exists():
        h = load_history(rd.history_csv)
        tail = slice(-min(200, len(h["loss_b"])), None)
        cols = "  ".join(f"{c} {np.mean(h[c][tail]):9.4f}" for c in h
                         if c not in ("global_step", "stage", "stage_step", "elapsed_s"))
        emit(f"  last {len(h['loss_b'][tail])} steps: {cols}")
    return {"global_step": step, "total": total, "stage": ckpt["stage_name"]}


# --- CLI --------------------------------------------------------------------


def _config_for_train(args) -> tuple[RunConfig, Path, Path | None]:
    """The config a `train` invocation should use, and where the run lives.

    A resume takes its config from the checkpoint, not from the file on disk: the
    file will have been edited by the time anyone comes back to a stopped run, and
    silently adopting those edits mid-run is how a model ends up trained against two
    different priors.
    """
    if args.resume:
        if not args.run:
            raise SystemExit("--resume needs --run pointing at the run directory")
        root = Path(args.run)
        ckpt_path = Path(args.resume) if isinstance(args.resume, str) else None
        if ckpt_path is not None and not ckpt_path.is_absolute():
            ckpt_path = root / ckpt_path
        if args.set:
            raise SystemExit("--set is not applied on resume: the config comes from the "
                             "checkpoint. Edit the config and start a new run instead.")
        ckpt_path = ckpt_path or (root / "checkpoints" / "latest.pt")
        ckpt = load_checkpoint(ckpt_path)
        cfg = config_from_dict(ckpt["config"],
                               source=ckpt.get("config_source", str(root)))
        return cfg, root, ckpt_path
    if not args.config:
        raise SystemExit("train needs --config (a new run) or --run ... --resume")
    cfg = load_config(args.config, args.set or [])
    return cfg, Path(args.run) if args.run else cfg.run_root, None


def _cmd_train(args) -> None:
    cfg, root, ckpt_path = _config_for_train(args)
    rd = RunDir.create(root, resume=bool(args.resume), fresh=args.fresh)
    log = Logger(rd.log_txt)

    if not args.resume:
        if args.config:
            shutil.copyfile(args.config, rd.config_yaml)
        if args.set:
            log(f"overrides: {' '.join(args.set)}")
    job_train(cfg, rd, log, resume=bool(args.resume), checkpoint=ckpt_path,
              strict_prior=args.strict_prior)


def _open(args) -> tuple[Run, Logger]:
    """The shared opening move of every post-training job."""
    run = load_run(args.run, checkpoint=args.checkpoint, device=args.device)
    return run, Logger(run.dir.log_txt)


def _cmd_sample(args) -> None:
    run, log = _open(args)
    job_sample(run, log, n=args.n, n_steps=args.n_steps, chunk=args.chunk,
               integrator=args.integrator, seed=args.seed,
               save_prior=True if args.save_prior else None)


def _cmd_entropy(args) -> None:
    run, log = _open(args)
    over: dict[str, Any] = {"batch": args.batch, "seed": args.seed,
                            "flow_steps": args.flow_steps, "flow_seed": args.flow_seed}
    over["batches"] = 0 if args.no_batches else args.batches
    over["flow_n"] = 0 if args.no_flow else args.flow_n
    if args.methods:
        over["methods"] = tuple(args.methods.split(","))
    if args.channels:
        over["flow_channels"] = tuple(args.channels.split(","))
    if over["batches"] == 0 and over["flow_n"] == 0:
        raise SystemExit("--no-batches and --no-flow together leave nothing to do")
    job_entropy(run, log, **over)


def _cmd_export(args) -> None:
    run, _ = _open(args)
    for p in job_export(run, args.out):
        print(p)


def _cmd_status(args) -> None:
    job_status(args.run)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m eesi.systems.tap.run",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="run the config's training stages",
                       description="Train a TAP interpolant from a YAML config.")
    t.add_argument("--config", help="the run's YAML config; required for a new run")
    t.add_argument("--run", help="the run directory (default: <out_dir>/<name>)")
    t.add_argument("--resume", nargs="?", const=True, default=False,
                   metavar="CHECKPOINT",
                   help="continue the run in --run, from latest.pt or from the given "
                        "checkpoint (a path relative to the run directory)")
    t.add_argument("--fresh", action="store_true",
                   help="discard an existing run at this path and start over")
    t.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a config value, e.g. --set train.stages.1.lr=1e-6; "
                        "repeatable")
    t.add_argument("--strict-prior", action="store_true",
                   help="on resume, fail rather than warn if the config's prior no "
                        "longer matches the one in the checkpoint")
    t.set_defaults(func=_cmd_train)

    def _job_parser(name: str, help_: str):
        """The flags every post-training job shares: which run, which checkpoint."""
        q = sub.add_parser(name, help=help_, description=help_)
        q.add_argument("--run", required=True, help="the run directory")
        q.add_argument("--checkpoint", default="latest",
                       help="which checkpoint to use (default: latest.pt); a path "
                            "relative to the run directory, or an absolute one")
        q.add_argument("--device", default="cpu")
        return q

    s = _job_parser("sample", "generate configurations from the trained drift")
    s.add_argument("--n", type=int, help="how many to generate")
    s.add_argument("--n-steps", type=int, help="integration steps")
    s.add_argument("--chunk", type=int, help="how many at once, to cap memory")
    s.add_argument("--integrator", choices=("rk4", "heun", "euler", "sde"))
    s.add_argument("--seed", type=int)
    s.add_argument("--save-prior", action="store_true",
                   help="also write the prior draw the samples came from")
    s.set_defaults(func=_cmd_sample)

    e = _job_parser("entropy", "estimate the entropy difference, and dS along the flow")
    e.add_argument("--batches", type=int, help="how many batch estimates")
    e.add_argument("--batch", type=int, help="configurations per batch")
    e.add_argument("--methods", help="comma-separated, e.g. dot,zdot,div")
    e.add_argument("--seed", type=int)
    e.add_argument("--no-batches", action="store_true",
                   help="skip the batch estimates and do only the running dS(t)")
    e.add_argument("--flow-n", type=int, help="trajectories for the running dS(t)")
    e.add_argument("--flow-steps", type=int)
    e.add_argument("--channels", help="comma-separated, e.g. div,dot")
    e.add_argument("--flow-seed", type=int)
    e.add_argument("--no-flow", action="store_true",
                   help="skip the running dS(t) and do only the batch estimates")
    e.set_defaults(func=_cmd_entropy)

    x = _job_parser("export", "write the bare per-network weight files")
    x.add_argument("--out", help="where to write them (default: the run's nets/)")
    x.set_defaults(func=_cmd_export)

    st = sub.add_parser("status", help="how far a run has got")
    st.add_argument("--run", required=True)
    st.set_defaults(func=_cmd_status)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
