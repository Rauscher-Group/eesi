"""Tests for `eesi.rundir`: run directories, atomic checkpoints, history files.

Runs as either pytest or a plain script:

    pytest tests/core/test_rundir.py
    python tests/core/test_rundir.py

No training here -- checkpoints are hand-built dicts. What is being pinned is the
crash behaviour: that a run directory is not clobbered by accident, that a checkpoint
is either wholly old or wholly new, and that history rows survive a resume without
duplicating or disappearing.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch

from eesi.rundir import (CKPT_FORMAT, CKPT_VERSION, Logger, RunDir, append_history,
                         env_info, history_header, load_checkpoint, load_history,
                         rng_state, save_checkpoint, set_rng_state, split_nets,
                         truncate_history)

COLUMNS = ("loss_b", "loss_s", "S_dot", "S_zdot")


def _ckpt(**kw):
    return {"model": {"net_b.w": torch.zeros(2), "net_s.w": torch.ones(2)},
            "global_step": 10, "history_rows": 10, **kw}


def _rows(n, start=0, stage="coarse"):
    return [(start + i, stage, i, 1.0, 2.0, 3.0, 4.0, 5.0, 0.1) for i in range(n)]


# --- the directory ----------------------------------------------------------


def test_create_makes_the_whole_layout(tmp_path):
    rd = RunDir.create(tmp_path / "run")
    for d in (rd.checkpoints, rd.nets, rd.samples, rd.entropy):
        assert d.is_dir()
    assert rd.latest() is None


def test_create_refuses_to_clobber_an_existing_run(tmp_path):
    """A mistyped --run must not overwrite a week of training."""
    rd = RunDir.create(tmp_path / "run")
    rd.config_yaml.write_text("name: run\n")
    with pytest.raises(FileExistsError, match="already holds a run"):
        RunDir.create(tmp_path / "run")
    RunDir.create(tmp_path / "run", resume=True)          # ... but resume is fine


def test_fresh_discards_a_run_but_only_something_that_looks_like_one(tmp_path):
    """`--fresh` deletes, so it checks that its target is a run directory first."""
    rd = RunDir.create(tmp_path / "run")
    rd.config_yaml.write_text("name: run\n")
    (rd.samples / "x.npy").write_bytes(b"junk")
    RunDir.create(tmp_path / "run", fresh=True)
    assert not rd.config_yaml.exists() and not (rd.samples / "x.npy").exists()

    precious = tmp_path / "not_a_run"
    precious.mkdir()
    (precious / "thesis.tex").write_text("years of work")
    with pytest.raises(ValueError, match="does not look like a run directory"):
        RunDir.create(precious, fresh=True)
    assert (precious / "thesis.tex").exists()


def test_resume_on_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to resume"):
        RunDir.create(tmp_path / "never_ran", resume=True)


def test_prune_keeps_latest_and_stage_finals(tmp_path):
    """Rolling checkpoints are disposable; the resume point and the stage boundaries
    are not."""
    rd = RunDir.create(tmp_path / "run")
    names = ["latest.pt", "stage0_final.pt",
             "stage0_step000100.pt", "stage0_step000200.pt", "stage0_step000300.pt"]
    for i, n in enumerate(names):
        p = rd.checkpoints / n
        p.write_bytes(b"x")
        os.utime(p, (1_000_000 + i, 1_000_000 + i))    # pin the age ordering

    dropped = rd.prune_checkpoints(keep=2)
    left = {p.name for p in rd.checkpoints.glob("*.pt")}
    assert {p.name for p in dropped} == {"stage0_step000100.pt"}
    assert left == {"latest.pt", "stage0_final.pt",
                    "stage0_step000200.pt", "stage0_step000300.pt"}


def test_logger_writes_through_immediately(tmp_path):
    """A kill -9 must not take the log with it, so nothing may sit in a buffer."""
    log = Logger(tmp_path / "log.txt", echo=False)
    log("step 0")
    assert (tmp_path / "log.txt").read_text() == "step 0\n"
    log("step 1")
    assert (tmp_path / "log.txt").read_text().splitlines() == ["step 0", "step 1"]


# --- checkpoints ------------------------------------------------------------


def test_checkpoint_round_trip_and_stamping(tmp_path):
    p = save_checkpoint(tmp_path / "c.pt", **_ckpt(stage_name="coarse"))
    got = load_checkpoint(p)
    assert got["format"] == CKPT_FORMAT and got["version"] == CKPT_VERSION
    assert got["created"].endswith("Z")
    assert got["stage_name"] == "coarse" and got["global_step"] == 10
    assert torch.equal(got["model"]["net_b.w"], torch.zeros(2))


def test_save_leaves_no_temporary_behind_and_replaces_in_place(tmp_path):
    p = tmp_path / "c.pt"
    save_checkpoint(p, **_ckpt(global_step=1))
    save_checkpoint(p, **_ckpt(global_step=2))
    assert load_checkpoint(p)["global_step"] == 2
    assert [q.name for q in tmp_path.iterdir()] == ["c.pt"]


def test_a_bare_state_dict_is_rejected_with_a_pointer_to_the_right_door(tmp_path):
    """What `train.py --out` and the notebooks write. Loading one as a run checkpoint
    would half-work and then fail on a missing key deep in the resume path."""
    torch.save({"net_b.w": torch.zeros(2)}, tmp_path / "bare.pth")
    with pytest.raises(ValueError, match=r"init\.net_b"):
        load_checkpoint(tmp_path / "bare.pth")


def test_a_future_checkpoint_version_names_both_versions(tmp_path):
    p = tmp_path / "c.pt"
    torch.save({"format": CKPT_FORMAT, "version": CKPT_VERSION + 1}, p)
    with pytest.raises(ValueError, match=f"version {CKPT_VERSION + 1}.*version {CKPT_VERSION}"):
        load_checkpoint(p)


def test_missing_checkpoint_is_a_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_split_nets_recovers_the_notebook_format():
    """The bridge between the two checkpoint conventions in this repo."""
    sd = {"net_b.egnn.w": torch.zeros(2), "net_b.bias": torch.ones(1),
          "net_s.egnn.w": torch.full((2,), 3.0)}
    b, s = split_nets(sd)
    assert set(b) == {"egnn.w", "bias"} and set(s) == {"egnn.w"}
    assert torch.equal(s["egnn.w"], torch.full((2,), 3.0))


def test_split_nets_refuses_a_state_dict_that_has_no_such_networks():
    with pytest.raises(ValueError, match="no keys with prefix"):
        split_nets({"w": torch.zeros(2)})


# --- rng --------------------------------------------------------------------


def test_rng_state_round_trip_reproduces_the_stream():
    """The property the whole resume design rests on."""
    torch.manual_seed(0)
    np.random.seed(0)
    state = rng_state()
    expect = (torch.randn(4), np.random.rand(4))

    torch.randn(100)                       # advance both streams somewhere else
    np.random.rand(100)
    set_rng_state(state)
    got = (torch.randn(4), np.random.rand(4))

    assert torch.equal(expect[0], got[0])
    assert np.array_equal(expect[1], got[1])


def test_env_info_never_raises_outside_a_checkout(tmp_path):
    info = env_info(tmp_path)
    assert info["torch"] and "git_sha" in info


# --- history ----------------------------------------------------------------


def test_header_wraps_the_systems_columns():
    assert history_header(COLUMNS) == [
        "global_step", "stage", "stage_step",
        "loss_b", "loss_s", "S_dot", "S_zdot", "transport", "elapsed_s"]


def test_append_writes_the_header_once_and_then_only_rows(tmp_path):
    p = tmp_path / "history.csv"
    header = history_header(COLUMNS)
    append_history(p, _rows(3), header=header)
    append_history(p, _rows(2, start=3), header=header)
    h = load_history(p)
    assert list(h["global_step"]) == [0, 1, 2, 3, 4]
    assert p.read_text().count("global_step") == 1


def test_load_history_types_its_columns(tmp_path):
    p = tmp_path / "history.csv"
    append_history(p, _rows(2), header=history_header(COLUMNS))
    h = load_history(p)
    assert h["global_step"].dtype == np.int64
    assert h["loss_b"].dtype == float and h["loss_b"][0] == 1.0
    assert list(h["stage"]) == ["coarse", "coarse"]


def test_truncate_drops_exactly_the_rows_a_resume_will_replay(tmp_path):
    """After a crash the file can hold steps the checkpoint never counted."""
    p = tmp_path / "history.csv"
    append_history(p, _rows(10), header=history_header(COLUMNS))
    assert truncate_history(p, 6) == 4
    h = load_history(p)
    assert list(h["global_step"]) == [0, 1, 2, 3, 4, 5]

    assert truncate_history(p, 6) == 0            # idempotent
    assert truncate_history(p, 99) == 0           # never grows the file
    assert truncate_history(tmp_path / "none.csv", 3) == 0


def test_history_stays_contiguous_across_a_simulated_resume(tmp_path):
    """The end-to-end shape of a crash: 10 rows written, a checkpoint at 6, resume."""
    p = tmp_path / "history.csv"
    header = history_header(COLUMNS)
    append_history(p, _rows(10), header=header)
    ckpt = save_checkpoint(tmp_path / "latest.pt", **_ckpt(history_rows=6))

    truncate_history(p, load_checkpoint(ckpt)["history_rows"])
    append_history(p, _rows(4, start=6), header=header)

    steps = load_history(p)["global_step"]
    assert list(steps) == list(range(10)), "resume duplicated or lost rows"


def test_load_history_on_a_missing_or_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_history(tmp_path / "nope.csv")
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_history(tmp_path / "empty.csv")


if __name__ == "__main__":
    import tempfile

    tests = [
        test_create_makes_the_whole_layout,
        test_create_refuses_to_clobber_an_existing_run,
        test_fresh_discards_a_run_but_only_something_that_looks_like_one,
        test_resume_on_a_missing_directory_is_an_error,
        test_prune_keeps_latest_and_stage_finals,
        test_logger_writes_through_immediately,
        test_checkpoint_round_trip_and_stamping,
        test_save_leaves_no_temporary_behind_and_replaces_in_place,
        test_a_bare_state_dict_is_rejected_with_a_pointer_to_the_right_door,
        test_a_future_checkpoint_version_names_both_versions,
        test_missing_checkpoint_is_a_file_not_found,
        test_split_nets_recovers_the_notebook_format,
        test_split_nets_refuses_a_state_dict_that_has_no_such_networks,
        test_rng_state_round_trip_reproduces_the_stream,
        test_env_info_never_raises_outside_a_checkout,
        test_header_wraps_the_systems_columns,
        test_append_writes_the_header_once_and_then_only_rows,
        test_load_history_types_its_columns,
        test_truncate_drops_exactly_the_rows_a_resume_will_replay,
        test_history_stays_contiguous_across_a_simulated_resume,
        test_load_history_on_a_missing_or_empty_file,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            if "tmp_path" in t.__code__.co_varnames[:t.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
