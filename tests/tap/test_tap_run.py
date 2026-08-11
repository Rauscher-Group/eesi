"""Tests for `eesi.systems.tap.run`: the config-driven training job.

Runs as either pytest or a plain script:

    pytest tests/tap/test_tap_run.py
    python tests/tap/test_tap_run.py

Everything is tiny -- a 6-particle chain, an 8-unit single-layer net, a few steps --
because none of this is about model quality. What is being checked is that a run
survives being stopped: that resuming from a checkpoint gives exactly the model an
uninterrupted run would have given, that the history file ends up with each step
once, and that a mistyped path does not destroy an existing run.

The reference data is drawn from a known prior rather than loaded from `data/`, so
the tests do not need the externally generated TAP trajectories.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
import yaml

from eesi.rundir import RunDir, load_checkpoint, load_history
from eesi.systems.tap.config import config_from_dict
from eesi.systems.tap.data import sample_prior
from eesi.systems.tap.dynamics import TAPDynamics
from eesi.systems.tap.run import (build_model, export_nets, job_entropy, job_export,
                                  job_sample, job_status, job_train, load_run, main,
                                  model_kwargs)

N = 6
STEPS = 8


def _write_data(tmp_path: Path, n: int = 64) -> Path:
    """A stand-in for the reference trajectories: chains from a known prior."""
    x = sample_prior(n, 8.0, 1.0, 2.0, 0.4, n_particles=N,
                     generator=torch.Generator().manual_seed(0))
    path = tmp_path / "data.npy"
    np.save(path, x.numpy())
    return path


def _raw(data_path: Path, tmp_path: Path, **patch) -> dict:
    raw = {
        "name": "unit",
        "out_dir": str(tmp_path / "runs"),
        "device": "cpu",
        "seed": 11,
        "data": {"path": str(data_path), "n": 64, "n_particles": N},
        "prior": {"k": 8.0, "b": 1.0, "gamma": 2.0, "cos_theta_0": 0.4},
        "model": {"hidden_nf": 8, "n_layers": 1, "gamma": "sqrt", "gamma_scale": 0.2},
        "train": {"batch": 4, "entropy": "both", "log_every": 0, "ckpt_every": 2,
                  "stages": [{"name": "coarse", "steps": STEPS, "lr": 1.0e-3,
                              "seed": 5}]},
        "prior_report": None,
    }
    del raw["prior_report"]
    raw["prior"]["report_samples"] = 0        # no 200k draw in a unit test
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return raw


def _run(raw: dict, root: Path, *, resume: bool = False, fresh: bool = False,
         checkpoint: Path | None = None):
    from eesi.rundir import Logger
    cfg = config_from_dict(raw, source="<test>")
    rd = RunDir.create(root, resume=resume, fresh=fresh)
    job_train(cfg, rd, Logger(rd.log_txt, echo=False), resume=resume,
              checkpoint=checkpoint)
    return rd


def _params(root: Path) -> dict:
    return load_checkpoint(root / "checkpoints" / "latest.pt")["model"]


# --- a plain run ------------------------------------------------------------


def test_a_run_produces_every_artifact(tmp_path):
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")

    assert rd.resolved_json.exists() and rd.log_txt.exists()
    assert (rd.checkpoints / "latest.pt").exists()
    assert (rd.checkpoints / "stage0_final.pt").exists()
    assert sorted(p.name for p in rd.nets.glob("*.pth")) == \
           [f"tap_b_si_N{N}_B4.pth", f"tap_s_si_N{N}_B4.pth"]

    h = load_history(rd.history_csv)
    assert list(h["global_step"]) == list(range(STEPS))
    assert set(h.keys()) >= {"loss_b", "loss_s", "S_dot", "S_zdot", "transport",
                             "elapsed_s", "stage"}
    assert np.isfinite(h["loss_b"]).all()


def test_the_history_width_follows_the_entropy_setting(tmp_path):
    raw = _raw(_write_data(tmp_path), tmp_path)
    raw["train"] = {**raw["train"], "entropy": None}
    rd = _run(raw, tmp_path / "run")
    h = load_history(rd.history_csv)
    assert "loss_b" in h and "S_dot" not in h


def test_resolved_json_records_what_actually_ran(tmp_path):
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    info = json.loads(rd.resolved_json.read_text())
    assert info["prior"]["k"] == 8.0
    assert info["prior"]["provenance"]["k"] == "fixed"
    assert info["model_kwargs"]["hidden_nf"] == 8
    assert info["n_parameters"] > 0 and info["device"] == "cpu"


def test_the_checkpoint_carries_what_a_resume_needs(tmp_path):
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    ckpt = load_checkpoint(rd.checkpoints / "latest.pt")
    assert set(ckpt) >= {"format", "version", "created", "config", "config_source",
                         "prior", "model_kwargs", "model", "optimizer", "rng", "env",
                         "stage_index", "stage_name", "stage_step", "global_step",
                         "hist_keys", "history_rows"}
    assert ckpt["global_step"] == STEPS and ckpt["stage_step"] == STEPS
    assert ckpt["history_rows"] == STEPS
    assert config_from_dict(ckpt["config"]).name == "unit"


def test_the_exported_weights_load_into_a_bare_network(tmp_path):
    """The notebook's old load cell has to keep working against a run directory."""
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    kwargs = json.loads(rd.resolved_json.read_text())["model_kwargs"]
    net_kwargs = {k: kwargs[k] for k in ("n_particles", "n_dims", "index_feature",
                                         "bond_feature", "time_order", "index_order",
                                         "hidden_nf", "n_layers")}
    for which in ("b", "s"):
        sd = torch.load(rd.nets / f"tap_{which}_si_N{N}_B4.pth", weights_only=True)
        net = TAPDynamics(**net_kwargs).double()
        net.load_state_dict(sd)              # strict by default: the key sets must match
        assert set(sd) == set(net.state_dict())


# --- resume -----------------------------------------------------------------


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path):
    """The property this whole design exists for.

    Four steps, a stop, then a resume for the remaining four, against eight straight
    through. On CPU in float64 with the random state restored, the two must agree
    bit for bit -- not approximately.
    """
    data = _write_data(tmp_path)
    straight = _run(_raw(data, tmp_path), tmp_path / "straight")

    raw = _raw(data, tmp_path)
    raw["train"] = {**raw["train"], "stages": [{"name": "coarse", "steps": 4,
                                                "lr": 1.0e-3, "seed": 5}]}
    _run(raw, tmp_path / "broken")

    full = _raw(data, tmp_path)              # the same config, all eight steps
    _run(full, tmp_path / "broken", resume=True)

    a, b = _params(straight.root), _params(tmp_path / "broken")
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} diverged after a resume"


def test_resume_leaves_the_history_with_every_step_once(tmp_path):
    data = _write_data(tmp_path)
    raw = _raw(data, tmp_path)
    raw["train"] = {**raw["train"], "stages": [{"name": "coarse", "steps": 4,
                                                "lr": 1.0e-3, "seed": 5}]}
    _run(raw, tmp_path / "run")
    _run(_raw(data, tmp_path), tmp_path / "run", resume=True)

    h = load_history((tmp_path / "run") / "history.csv")
    assert list(h["global_step"]) == list(range(STEPS))


def test_extra_history_rows_past_the_checkpoint_are_dropped_on_resume(tmp_path):
    """A crash between writing rows and writing the checkpoint that counts them."""
    data = _write_data(tmp_path)
    raw = _raw(data, tmp_path)
    raw["train"] = {**raw["train"], "stages": [{"name": "coarse", "steps": 4,
                                                "lr": 1.0e-3, "seed": 5}]}
    rd = _run(raw, tmp_path / "run")
    with rd.history_csv.open("a") as fh:     # rows the checkpoint never knew about
        fh.write("4,coarse,4,1,2,3,4,5,0.1\n5,coarse,5,1,2,3,4,5,0.1\n")

    _run(_raw(data, tmp_path), tmp_path / "run", resume=True)
    h = load_history(rd.history_csv)
    assert list(h["global_step"]) == list(range(STEPS))


def test_a_resume_after_every_stage_is_done_stops_without_training(tmp_path):
    data = _write_data(tmp_path)
    rd = _run(_raw(data, tmp_path), tmp_path / "run")
    before = {k: v.clone() for k, v in _params(rd.root).items()}

    _run(_raw(data, tmp_path), tmp_path / "run", resume=True)
    assert "already complete" in rd.log_txt.read_text()
    for key, value in _params(rd.root).items():
        assert torch.equal(before[key], value)


def test_resume_uses_the_checkpoints_prior_and_warns_when_the_config_moved(tmp_path):
    """Re-resolving would put a different base distribution under a half-trained
    flow, and nothing downstream would notice."""
    data = _write_data(tmp_path)
    raw = _raw(data, tmp_path)
    raw["train"] = {**raw["train"], "stages": [{"name": "coarse", "steps": 4,
                                                "lr": 1.0e-3, "seed": 5}]}
    rd = _run(raw, tmp_path / "run")

    moved = _raw(data, tmp_path)
    moved["prior"] = {**moved["prior"], "gamma": 9.0}
    _run(moved, tmp_path / "run", resume=True)

    assert "no longer matches" in rd.log_txt.read_text()
    assert load_checkpoint(rd.checkpoints / "latest.pt")["prior"]["gamma"] == 2.0


def test_a_specific_checkpoint_can_be_resumed_from(tmp_path):
    data = _write_data(tmp_path)
    rd = _run(_raw(data, tmp_path), tmp_path / "run")
    mid = rd.checkpoints / "stage0_step00000004.pt"
    assert mid.exists(), sorted(p.name for p in rd.checkpoints.glob("*.pt"))

    _run(_raw(data, tmp_path), tmp_path / "run", resume=True, checkpoint=mid)
    assert load_checkpoint(rd.checkpoints / "latest.pt")["global_step"] == STEPS


# --- stages -----------------------------------------------------------------


def test_two_stages_run_in_order_with_their_own_learning_rates(tmp_path):
    raw = _raw(_write_data(tmp_path), tmp_path)
    raw["train"] = {**raw["train"],
                    "stages": [{"name": "coarse", "steps": 4, "lr": 1.0e-3},
                               {"name": "fine", "steps": 6, "lr": 1.0e-5}]}
    rd = _run(raw, tmp_path / "run")

    h = load_history(rd.history_csv)
    assert list(h["global_step"]) == list(range(10))
    assert list(h["stage"]) == ["coarse"] * 4 + ["fine"] * 6
    assert list(h["stage_step"]) == list(range(4)) + list(range(6))
    assert (rd.checkpoints / "stage0_final.pt").exists()
    assert (rd.checkpoints / "stage1_final.pt").exists()


def test_a_resume_at_a_stage_boundary_starts_the_next_stage(tmp_path):
    data = _write_data(tmp_path)
    raw = _raw(data, tmp_path)
    stages = [{"name": "coarse", "steps": 4, "lr": 1.0e-3},
              {"name": "fine", "steps": 6, "lr": 1.0e-5}]
    first = copy.deepcopy(raw)
    first["train"] = {**raw["train"], "stages": stages[:1]}
    _run(first, tmp_path / "run")

    both = copy.deepcopy(raw)
    both["train"] = {**raw["train"], "stages": stages}
    rd = _run(both, tmp_path / "run", resume=True)

    h = load_history(rd.history_csv)
    assert list(h["stage"]) == ["coarse"] * 4 + ["fine"] * 6


def test_reset_optimizer_false_carries_adam_moments_across_the_boundary(tmp_path):
    """Both settings must run; they are different experiments, not right and wrong."""
    data = _write_data(tmp_path)
    out = {}
    for reset in (True, False):
        raw = _raw(data, tmp_path)
        raw["train"] = {**raw["train"], "reset_optimizer": reset,
                        "stages": [{"name": "a", "steps": 4, "lr": 1.0e-3},
                                   {"name": "b", "steps": 4, "lr": 1.0e-4}]}
        rd = _run(raw, tmp_path / f"run_{reset}")
        out[reset] = _params(rd.root)
    assert any(not torch.equal(out[True][k], out[False][k]) for k in out[True])


# --- warm start and safety --------------------------------------------------


def test_a_warm_start_from_bare_weight_files_changes_the_initial_model(tmp_path):
    data = _write_data(tmp_path)
    rd = _run(_raw(data, tmp_path), tmp_path / "source")

    raw = _raw(data, tmp_path)
    raw["init"] = {"net_b": str(rd.nets / f"tap_b_si_N{N}_B4.pth"),
                   "net_s": str(rd.nets / f"tap_s_si_N{N}_B4.pth")}
    raw["train"] = {**raw["train"], "stages": [{"name": "c", "steps": 1, "lr": 1.0e-8}]}
    warm = _run(raw, tmp_path / "warm")

    cold = _params(tmp_path / "source")
    got = _params(warm.root)
    assert all(torch.allclose(cold[k], got[k], atol=1e-4) for k in cold), \
        "the warm start did not load the source run's weights"
    assert "warm start from" in warm.log_txt.read_text()


def test_a_warm_start_from_a_run_checkpoint_works_too(tmp_path):
    data = _write_data(tmp_path)
    rd = _run(_raw(data, tmp_path), tmp_path / "source")

    raw = _raw(data, tmp_path)
    raw["init"] = {"checkpoint": str(rd.checkpoints / "latest.pt")}
    raw["train"] = {**raw["train"], "stages": [{"name": "c", "steps": 1, "lr": 1.0e-8}]}
    warm = _run(raw, tmp_path / "warm")
    assert load_checkpoint(warm.checkpoints / "latest.pt")["global_step"] == 1


def test_an_existing_run_is_not_overwritten_by_accident(tmp_path):
    data = _write_data(tmp_path)
    rd = _run(_raw(data, tmp_path), tmp_path / "run")
    before = rd.history_csv.read_text()

    with pytest.raises(FileExistsError, match="already holds a run"):
        _run(_raw(data, tmp_path), tmp_path / "run")
    assert rd.history_csv.read_text() == before


def test_a_stop_request_ends_the_run_with_a_usable_checkpoint(tmp_path):
    """What a Ctrl-C in the tmux pane does: finish the step, checkpoint, stop."""
    from eesi.rundir import Logger
    import eesi.systems.tap.run as run_mod

    raw = _raw(_write_data(tmp_path), tmp_path)
    raw["train"] = {**raw["train"], "ckpt_every": 1000}     # no periodic checkpoint
    cfg = config_from_dict(raw, source="<test>")
    rd = RunDir.create(tmp_path / "run")

    real_writer = run_mod._StageWriter

    class StopAfterThree(real_writer):
        """Stands in for a signal arriving while step 2 is being computed.

        The flag has to be set before the callback reads it, exactly as a real signal
        would land mid-step: setting it afterwards lets one more step through, which
        is correct behaviour but a different scenario.
        """
        def __call__(self, step, row, x0, x1):
            if step == 2:
                self.stop.requested = True
            return super().__call__(step, row, x0, x1)

    run_mod._StageWriter = StopAfterThree
    try:
        job_train(cfg, rd, Logger(rd.log_txt, echo=False))
    finally:
        run_mod._StageWriter = real_writer

    ckpt = load_checkpoint(rd.checkpoints / "latest.pt")
    assert ckpt["stage_step"] == 3 and ckpt["history_rows"] == 3
    assert list(load_history(rd.history_csv)["global_step"]) == [0, 1, 2]
    assert "--resume" in rd.log_txt.read_text()

    _run(raw, rd.root, resume=True)          # and it picks up from there
    assert list(load_history(rd.history_csv)["global_step"]) == list(range(STEPS))


# --- reopening a run, and the analysis jobs ---------------------------------


def test_load_run_gives_back_the_prior_and_model_that_were_trained(tmp_path):
    """The notebook's replacement for rebuilding the model and re-deriving the prior
    by hand, which is where it most easily drifts out of step with the weights."""
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    run = load_run(rd.root)

    assert run.config.name == "unit"
    assert run.prior.as_tuple() == (8.0, 1.0, 2.0, 0.4)
    assert run.global_step == STEPS
    assert list(run.history()["global_step"]) == list(range(STEPS))
    for key, value in _params(rd.root).items():
        assert torch.equal(run.model.state_dict()[key], value)


def test_load_run_can_open_an_earlier_checkpoint(tmp_path):
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    early = load_run(rd.root, checkpoint="checkpoints/stage0_step00000004.pt")
    assert early.global_step == 4
    assert load_run(rd.root).global_step == STEPS


def test_sample_writes_configurations_of_the_right_shape(tmp_path):
    from eesi.rundir import Logger
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    run = load_run(rd.root)
    out = job_sample(run, Logger(rd.log_txt, echo=False), n=32, n_steps=4, chunk=16)

    x = np.load(out)
    assert x.shape == (32, N, 3)
    assert np.isfinite(x).all()
    assert np.abs(x[:, 0]).max() < 1e-12, "generated chains must stay tail-anchored"

    meta = json.loads((rd.samples / "meta.json").read_text())
    assert meta["global_step"] == STEPS and meta["n"] == 32
    assert meta["prior"]["k"] == 8.0


@pytest.mark.parametrize("integrator", ["rk4", "heun", "euler"])
def test_every_integrator_runs(tmp_path, integrator):
    from eesi.rundir import Logger
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    run = load_run(rd.root)
    out = job_sample(run, Logger(rd.log_txt, echo=False), n=8, n_steps=3, chunk=4,
                     integrator=integrator)
    assert np.load(out).shape == (8, N, 3)


def test_sample_can_keep_the_prior_draw_it_started_from(tmp_path):
    from eesi.rundir import Logger
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    job_sample(load_run(rd.root), Logger(rd.log_txt, echo=False), n=8, n_steps=3,
               chunk=8, save_prior=True)
    assert np.load(rd.samples / f"x0_g{STEPS}_n8.npy").shape == (8, N, 3)


def test_sampling_is_reproducible_from_its_seed(tmp_path):
    from eesi.rundir import Logger
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    log = Logger(rd.log_txt, echo=False)
    a = np.load(job_sample(load_run(rd.root), log, n=8, n_steps=3, chunk=8, seed=7))
    b = np.load(job_sample(load_run(rd.root), log, n=8, n_steps=3, chunk=8, seed=7))
    assert np.array_equal(a, b)


def test_entropy_batches_are_saved_and_the_summary_matches_them(tmp_path):
    from eesi.rundir import Logger
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    job_entropy(load_run(rd.root), Logger(rd.log_txt, echo=False),
                batches=6, batch=4, methods=("dot", "div"), flow_n=0)

    summary = json.loads((rd.entropy / "summary.json").read_text())
    for method in ("dot", "div"):
        values = np.load(rd.entropy / f"means_{method}.npy")
        assert values.shape == (6,) and np.isfinite(values).all()
        assert summary["batches"][method]["dS"] == pytest.approx(values.mean())
        assert summary["batches"][method]["ci95"] == pytest.approx(
            1.96 * values.std() / np.sqrt(6))
        # S[p1] is the prior's entropy plus the estimated difference.
        assert summary["batches"][method]["S_target"] == pytest.approx(
            summary["S_prior"] + values.mean())
    assert (rd.entropy / "means.csv").read_text().startswith("batch,dot,div\n")
    assert "flow" not in summary


def test_the_running_entropy_has_one_row_per_integration_step(tmp_path):
    from eesi.rundir import Logger
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    job_entropy(load_run(rd.root), Logger(rd.log_txt, echo=False),
                batches=0, flow_n=12, flow_steps=5, flow_chunk=5,
                flow_channels=("div", "dot"))

    ts = np.load(rd.entropy / "ts.npy")
    assert ts.shape == (6,) and ts[0] == 0.0 and ts[-1] == pytest.approx(1.0)
    summary = json.loads((rd.entropy / "summary.json").read_text())
    for channel in ("div", "dot"):
        ent = np.load(rd.entropy / f"ent_c_{channel}.npy")
        assert ent.shape == (6, 12), "expected (flow_steps + 1, flow_n)"
        assert np.allclose(ent[0], 0.0), "the running entropy starts at zero"
        assert summary["flow"][channel]["dS"] == pytest.approx(ent[-1].mean())
    assert "batches" not in summary


def test_chunking_does_not_change_the_running_entropy(tmp_path):
    """The trajectories are independent, so the chunk size is a memory knob only."""
    from eesi.rundir import Logger
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    log = Logger(rd.log_txt, echo=False)
    job_entropy(load_run(rd.root), log, batches=0, flow_n=8, flow_steps=4, flow_chunk=8)
    whole = np.load(rd.entropy / "ent_c_div.npy").copy()
    job_entropy(load_run(rd.root), log, batches=0, flow_n=8, flow_steps=4, flow_chunk=2)
    assert np.array_equal(whole, np.load(rd.entropy / "ent_c_div.npy"))


def test_export_writes_loadable_weights_somewhere_else(tmp_path):
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    run = load_run(rd.root)
    dest = tmp_path / "elsewhere"
    b_path, s_path = job_export(run, dest)

    assert b_path.parent == dest
    kwargs = model_kwargs(run.config.model, N, 3)
    net_kwargs = {k: kwargs[k] for k in ("n_particles", "n_dims", "index_feature",
                                         "bond_feature", "time_order", "index_order",
                                         "hidden_nf", "n_layers")}
    TAPDynamics(**net_kwargs).double().load_state_dict(
        torch.load(b_path, weights_only=True))


def test_status_reports_progress(tmp_path, capsys):
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    info = job_status(rd.root)
    assert info == {"global_step": STEPS, "total": STEPS, "stage": "coarse"}
    assert "100.0%" in capsys.readouterr().out


def test_status_on_a_run_that_never_checkpointed(tmp_path, capsys):
    RunDir.create(tmp_path / "empty")
    assert job_status(tmp_path / "empty") == {}
    assert "no checkpoint yet" in capsys.readouterr().out


# --- the CLI ----------------------------------------------------------------


def test_the_cli_trains_from_a_config_file_and_then_resumes(tmp_path):
    data = _write_data(tmp_path)
    raw = _raw(data, tmp_path)
    raw["train"] = {**raw["train"], "stages": [{"name": "coarse", "steps": 4,
                                                "lr": 1.0e-3, "seed": 5}]}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    root = tmp_path / "cli"

    main(["train", "--config", str(cfg_path), "--run", str(root)])
    assert (root / "config.yaml").read_text() == cfg_path.read_text()
    assert load_checkpoint(root / "checkpoints" / "latest.pt")["global_step"] == 4

    raw["train"]["stages"][0]["steps"] = STEPS
    cfg_path.write_text(yaml.safe_dump(raw))
    main(["train", "--run", str(root), "--resume"])
    # The config comes from the checkpoint, so the edit above is deliberately ignored.
    assert load_checkpoint(root / "checkpoints" / "latest.pt")["global_step"] == 4


def test_the_cli_applies_set_overrides(tmp_path):
    raw = _raw(_write_data(tmp_path), tmp_path)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    root = tmp_path / "cli"

    main(["train", "--config", str(cfg_path), "--run", str(root),
          "--set", "train.stages.0.steps=3", "--set", "model.hidden_nf=4"])
    ckpt = load_checkpoint(root / "checkpoints" / "latest.pt")
    assert ckpt["global_step"] == 3 and ckpt["model_kwargs"]["hidden_nf"] == 4


def test_the_cli_defaults_the_run_directory_to_out_dir_over_name(tmp_path):
    raw = _raw(_write_data(tmp_path), tmp_path)
    raw["train"] = {**raw["train"], "stages": [{"name": "c", "steps": 1, "lr": 1.0e-3}]}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    main(["train", "--config", str(cfg_path)])
    assert (tmp_path / "runs" / "unit" / "checkpoints" / "latest.pt").exists()


def test_the_cli_runs_the_analysis_jobs_against_a_finished_run(tmp_path):
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    root = str(rd.root)

    main(["sample", "--run", root, "--n", "8", "--n-steps", "3", "--chunk", "4"])
    assert np.load(rd.samples / f"x_gen_g{STEPS}_n8.npy").shape == (8, N, 3)

    main(["entropy", "--run", root, "--batches", "3", "--batch", "4",
          "--methods", "div", "--no-flow"])
    assert np.load(rd.entropy / "means_div.npy").shape == (3,)
    assert not (rd.entropy / "ent_c_div.npy").exists()

    main(["entropy", "--run", root, "--no-batches", "--flow-n", "6",
          "--flow-steps", "4", "--channels", "div"])
    assert np.load(rd.entropy / "ent_c_div.npy").shape == (5, 6)

    dest = tmp_path / "out"
    main(["export", "--run", root, "--out", str(dest)])
    assert (dest / f"tap_b_si_N{N}_B4.pth").exists()

    main(["status", "--run", root])


def test_the_cli_rejects_an_entropy_job_with_nothing_to_do(tmp_path):
    rd = _run(_raw(_write_data(tmp_path), tmp_path), tmp_path / "run")
    with pytest.raises(SystemExit, match="nothing to do"):
        main(["entropy", "--run", str(rd.root), "--no-batches", "--no-flow"])


def test_the_cli_refuses_combinations_that_cannot_mean_anything(tmp_path):
    with pytest.raises(SystemExit, match="needs --config"):
        main(["train"])
    with pytest.raises(SystemExit, match="--resume needs --run"):
        main(["train", "--resume"])

    raw = _raw(_write_data(tmp_path), tmp_path)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    root = tmp_path / "cli"
    main(["train", "--config", str(cfg_path), "--run", str(root),
          "--set", "train.stages.0.steps=2"])
    with pytest.raises(SystemExit, match="--set is not applied on resume"):
        main(["train", "--run", str(root), "--resume", "--set", "seed=3"])


if __name__ == "__main__":
    import tempfile

    tests = [
        test_a_run_produces_every_artifact,
        test_the_history_width_follows_the_entropy_setting,
        test_resolved_json_records_what_actually_ran,
        test_the_checkpoint_carries_what_a_resume_needs,
        test_the_exported_weights_load_into_a_bare_network,
        test_resume_reproduces_an_uninterrupted_run_exactly,
        test_resume_leaves_the_history_with_every_step_once,
        test_extra_history_rows_past_the_checkpoint_are_dropped_on_resume,
        test_a_resume_after_every_stage_is_done_stops_without_training,
        test_resume_uses_the_checkpoints_prior_and_warns_when_the_config_moved,
        test_a_specific_checkpoint_can_be_resumed_from,
        test_two_stages_run_in_order_with_their_own_learning_rates,
        test_a_resume_at_a_stage_boundary_starts_the_next_stage,
        test_reset_optimizer_false_carries_adam_moments_across_the_boundary,
        test_a_warm_start_from_bare_weight_files_changes_the_initial_model,
        test_a_warm_start_from_a_run_checkpoint_works_too,
        test_an_existing_run_is_not_overwritten_by_accident,
        test_a_stop_request_ends_the_run_with_a_usable_checkpoint,
        test_load_run_gives_back_the_prior_and_model_that_were_trained,
        test_load_run_can_open_an_earlier_checkpoint,
        test_sample_writes_configurations_of_the_right_shape,
        test_sample_can_keep_the_prior_draw_it_started_from,
        test_sampling_is_reproducible_from_its_seed,
        test_entropy_batches_are_saved_and_the_summary_matches_them,
        test_the_running_entropy_has_one_row_per_integration_step,
        test_chunking_does_not_change_the_running_entropy,
        test_export_writes_loadable_weights_somewhere_else,
        test_the_cli_runs_the_analysis_jobs_against_a_finished_run,
        test_the_cli_rejects_an_entropy_job_with_nothing_to_do,
        test_the_cli_trains_from_a_config_file_and_then_resumes,
        test_the_cli_applies_set_overrides,
        test_the_cli_defaults_the_run_directory_to_out_dir_over_name,
        test_the_cli_refuses_combinations_that_cannot_mean_anything,
    ]
    failed = 0
    for t in tests:
        try:
            with tempfile.TemporaryDirectory() as d:
                t(Path(d))
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
