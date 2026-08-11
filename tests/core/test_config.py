"""Tests for `eesi.config`: the YAML validator vocabulary and the stage schema.

Runs as either pytest or a plain script:

    pytest tests/core/test_config.py
    python tests/core/test_config.py

The thing under test is the failure behaviour, not the happy path. A config that is
wrong must fail at load, loudly, naming the key -- because the alternative is finding
out after six hours of training that `hidden_nff` was ignored.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from eesi.config import (CheckpointConfig, StageConfig, _choice, _closed, _mapping,
                         _optional, _positive, _str_seq, _typed, apply_overrides,
                         checkpoint_from_dict, checkpoint_to_dict, load_yaml,
                         resolve_device, resolve_dtype, stage_to_dict, stages_from_dict)

TRAIN = {"batch": 256, "entropy": "both", "ckpt_every": 2000,
         "stages": [{"name": "coarse", "steps": 100, "lr": 1.0e-4, "seed": 1},
                    {"name": "fine", "steps": 200, "lr": 1.0e-5, "seed": 2}]}


# --- primitives -------------------------------------------------------------


def test_unknown_key_is_an_error_with_a_suggestion():
    with pytest.raises(ValueError, match=r"hidden_nff.*did you mean 'hidden_nf'"):
        _closed({"hidden_nff": 32}, ["hidden_nf", "n_layers"], where="model")


def test_unknown_key_error_names_its_block_and_lists_what_is_allowed():
    with pytest.raises(ValueError, match=r"^model: unknown key\(s\) 'zzz'\. Allowed: "):
        _closed({"zzz": 1}, ["hidden_nf"], where="model")


def test_missing_required_key_names_the_path():
    with pytest.raises(ValueError, match=r"stage: missing required key 'steps'"):
        _typed({}, "steps", int, where="stage")


def test_float_accepts_int_and_the_pyyaml_string_spelling():
    """`lr: 1e-4` is a STRING to PyYAML -- only `1.0e-4` matches its float pattern.

    Rejecting the shorter spelling would be technically correct and practically a
    trap, since it is the form everyone writes. Both are accepted and mean the same.
    """
    assert _typed({"lr": 1}, "lr", float, where="s") == 1.0
    assert _typed({"lr": "1e-4"}, "lr", float, where="s") == 1e-4
    assert _typed({"lr": 1.0e-4}, "lr", float, where="s") == 1e-4
    with pytest.raises(ValueError, match="expected float"):
        _typed({"lr": "fast"}, "lr", float, where="s")


def test_bool_is_not_an_int():
    """`bool` subclasses `int`, so `steps: true` passes a naive isinstance check."""
    with pytest.raises(ValueError, match="expected int"):
        _typed({"steps": True}, "steps", int, where="s")
    with pytest.raises(ValueError, match="expected float"):
        _typed({"lr": False}, "lr", float, where="s")
    assert _typed({"align": True}, "align", bool, where="s") is True


def test_optional_distinguishes_absent_from_null_from_wrong_type():
    assert _optional({}, "seed", int, where="s") is None
    assert _optional({"seed": None}, "seed", int, where="s") is None
    assert _optional({"seed": 7}, "seed", int, where="s") == 7
    with pytest.raises(ValueError, match="expected int"):
        _optional({"seed": "x"}, "seed", int, where="s")


def test_choice_positive_mapping_and_str_seq():
    assert _choice("dot", (None, "dot"), where="s") == "dot"
    with pytest.raises(ValueError, match="must be one of"):
        _choice("nope", (None, "dot"), where="s")
    with pytest.raises(ValueError, match="must be positive"):
        _positive(0, where="s.steps")
    assert _mapping({}, "model", where="c") == {}
    assert _mapping({"model": None}, "model", where="c") == {}
    with pytest.raises(ValueError, match="expected a mapping"):
        _mapping({"model": [1]}, "model", where="c")
    assert _str_seq({"m": "dot"}, "m", (), where="c") == ("dot",)
    assert _str_seq({}, "m", ("div",), where="c") == ("div",)
    with pytest.raises(ValueError, match="list of strings"):
        _str_seq({"m": [1, 2]}, "m", (), where="c")


# --- stages -----------------------------------------------------------------


def test_block_level_keys_become_stage_defaults_and_stages_override_them():
    stages = stages_from_dict(TRAIN)
    assert [s.name for s in stages] == ["coarse", "fine"]
    assert [s.steps for s in stages] == [100, 200]
    assert all(s.batch == 256 and s.entropy == "both" for s in stages)

    override = {**TRAIN, "stages": [{**TRAIN["stages"][0], "batch": 64}]}
    assert stages_from_dict(override)[0].batch == 64


def test_steps_and_lr_may_be_declared_once_for_every_stage():
    stages = stages_from_dict({"steps": 50, "lr": 1.0e-4,
                               "stages": [{"name": "a"}, {"name": "b", "lr": 1.0e-5}]})
    assert [(s.steps, s.lr) for s in stages] == [(50, 1e-4), (50, 1e-5)]


def test_stage_defaults_match_the_dataclass():
    (stage,) = stages_from_dict({"stages": [{"steps": 1, "lr": 0.1}]})
    assert (stage.name, stage.batch, stage.seed, stage.align, stage.batch_ot) == \
           ("stage0", 64, None, True, True)
    assert (stage.entropy, stage.log_every, stage.ckpt_every, stage.reset_optimizer) == \
           (None, 200, 2000, True)


@pytest.mark.parametrize("bad,match", [
    ({"stages": []}, "non-empty"),
    ({"stages": [{"lr": 1.0e-4}]}, "missing required key 'steps'"),
    ({"stages": [{"steps": 0, "lr": 1.0e-4}]}, "steps: must be positive"),
    ({"stages": [{"steps": 10, "lr": 0.0}]}, "lr: must be positive"),
    ({"stages": [{"steps": 10, "lr": 0.1, "batch": 0}]}, "batch: must be positive"),
    ({"stages": [{"steps": 10, "lr": 0.1, "ckpt_every": 0}]}, "ckpt_every: must be positive"),
    ({"stages": [{"steps": 10, "lr": 0.1, "stpes": 2}]}, "did you mean 'steps'"),
    ({"steps": 1, "lr": 0.1, "stages": [{"name": "a"}, {"name": "a"}]}, "must be unique"),
    ({}, "missing required key 'stages'"),
    ({"stages": {"steps": 1}}, "non-empty list"),
])
def test_malformed_stage_blocks_are_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        stages_from_dict(bad)


def test_stages_must_agree_on_the_entropy_channel():
    """One history file, one header -- so one column count for the whole run."""
    raw = {"steps": 1, "lr": 0.1,
           "stages": [{"name": "a", "entropy": "both"}, {"name": "b", "entropy": "dot"}]}
    with pytest.raises(ValueError, match="same `entropy` setting"):
        stages_from_dict(raw)


def test_unknown_entropy_channel_is_rejected_at_load():
    raw = {"steps": 1, "lr": 0.1, "stages": [{"name": "a", "entropy": "bogus"}]}
    with pytest.raises(ValueError, match="entropy: must be one of"):
        stages_from_dict(raw)


def test_name_cannot_be_hoisted_to_the_block_level():
    """Shared stage names would collide in the checkpoint filenames."""
    with pytest.raises(ValueError, match=r"^train: unknown key\(s\) 'name'"):
        stages_from_dict({"name": "x", "steps": 1, "lr": 0.1, "stages": [{}]})


def test_stage_round_trips_through_its_dict_form():
    """What makes `resolved.json` a reproducibility artifact rather than a log."""
    stages = stages_from_dict(TRAIN)
    again = stages_from_dict({"stages": [stage_to_dict(s) for s in stages]})
    assert again == stages


def test_checkpoint_block_defaults_and_round_trip():
    assert checkpoint_from_dict({}) == CheckpointConfig(keep=3, export_nets=True)
    cfg = checkpoint_from_dict({"keep": 5, "export_nets": False})
    assert checkpoint_from_dict(checkpoint_to_dict(cfg)) == cfg
    with pytest.raises(ValueError, match="did you mean 'keep'"):
        checkpoint_from_dict({"kep": 1})


# --- overrides --------------------------------------------------------------


def test_override_reaches_a_nested_key_and_a_stage_by_index():
    raw = {"device": "auto", "model": {"eps": 1.0e-6}, "train": dict(TRAIN)}
    out = apply_overrides(raw, ["device=cpu", "model.eps=1e-3",
                                "train.stages.1.lr=1.0e-6"])
    assert out["device"] == "cpu"
    assert _typed(out["model"], "eps", float, where="model") == 1e-3
    assert _typed(out["train"]["stages"][1], "lr", float, where="s") == 1e-6
    assert raw["device"] == "auto", "the input dict must not be mutated"


def test_override_values_are_parsed_as_yaml():
    """...which means the RHS inherits PyYAML's float quirk, and that is fine.

    `1e-3` arrives as the string "1e-3" exactly as it would from a config file, and
    is floated by `_typed` at the same point in the same way. The alternative --
    coercing here -- cannot work, since `apply_overrides` does not know the target
    type of the key it is writing.
    """
    raw = {"data": {"n": 100}, "entropy": {"methods": ["div"]},
           "model": {"eps": 1.0, "eps2": 1.0}}
    out = apply_overrides(raw, ["data.n=null", "entropy.methods=[dot, div]",
                                "model.eps=1e-3", "model.eps2=1.0e-3"])
    assert out["data"]["n"] is None
    assert out["entropy"]["methods"] == ["dot", "div"]
    assert out["model"]["eps"] == "1e-3" and out["model"]["eps2"] == 1e-3
    assert _typed(out["model"], "eps", float, where="model") == 1e-3


@pytest.mark.parametrize("bad,match", [
    ("modle.hidden_nf=64", "no key 'modle'"),
    ("model.hidden_nff=64", "no key 'model.hidden_nff'"),
    ("model", "expected KEY=VALUE"),
    ("train.stages.9.lr=1e-6", "index 9 out of range"),
    ("train.stages.first.lr=1e-6", "must be an integer index"),
])
def test_overrides_that_would_silently_do_nothing_are_errors(bad, match):
    raw = {"model": {"hidden_nf": 32}, "train": dict(TRAIN)}
    with pytest.raises(ValueError, match=match):
        apply_overrides(raw, [bad])


# --- loading and torch handles ----------------------------------------------


def test_load_yaml_round_trip_and_failures(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: run\nmodel:\n  eps: 1.0e-6\n")
    assert load_yaml(p) == {"name": "run", "model": {"eps": 1e-6}}

    (tmp_path / "empty.yaml").write_text("")
    assert load_yaml(tmp_path / "empty.yaml") == {}

    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_yaml(tmp_path / "list.yaml")

    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_device_and_dtype_resolution():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in ("cpu", "cuda")
    assert resolve_dtype("float64") is torch.float64
    assert resolve_dtype("float32") is torch.float32
    with pytest.raises(ValueError, match="dtype must be one of"):
        resolve_dtype("float16")


if __name__ == "__main__":
    import tempfile

    tests = [
        test_unknown_key_is_an_error_with_a_suggestion,
        test_unknown_key_error_names_its_block_and_lists_what_is_allowed,
        test_missing_required_key_names_the_path,
        test_float_accepts_int_and_the_pyyaml_string_spelling,
        test_bool_is_not_an_int,
        test_optional_distinguishes_absent_from_null_from_wrong_type,
        test_choice_positive_mapping_and_str_seq,
        test_block_level_keys_become_stage_defaults_and_stages_override_them,
        test_steps_and_lr_may_be_declared_once_for_every_stage,
        test_stage_defaults_match_the_dataclass,
        test_stages_must_agree_on_the_entropy_channel,
        test_unknown_entropy_channel_is_rejected_at_load,
        test_name_cannot_be_hoisted_to_the_block_level,
        test_stage_round_trips_through_its_dict_form,
        test_checkpoint_block_defaults_and_round_trip,
        test_override_reaches_a_nested_key_and_a_stage_by_index,
        test_override_values_are_parsed_as_yaml,
        test_device_and_dtype_resolution,
    ]
    for _bad, _m in [({"stages": []}, "non-empty"),
                     ({"stages": [{"lr": 1.0e-4}]}, "missing required key 'steps'"),
                     ({"stages": [{"steps": 0, "lr": 1.0e-4}]}, "must be positive"),
                     ({"stages": [{"steps": 10, "lr": 0.1, "stpes": 2}]}, "did you mean"),
                     ({}, "missing required key 'stages'")]:
        tests.append(lambda b=_bad, m=_m: test_malformed_stage_blocks_are_rejected(b, m))
    for _bad, _m in [("modle.hidden_nf=64", "no key 'modle'"),
                     ("model", "expected KEY=VALUE"),
                     ("train.stages.9.lr=1e-6", "out of range")]:
        tests.append(lambda b=_bad, m=_m: test_overrides_that_would_silently_do_nothing_are_errors(b, m))
    with tempfile.TemporaryDirectory() as _d:
        tests.append(lambda: test_load_yaml_round_trip_and_failures(Path(_d)))

        failed = 0
        for t in tests:
            name = getattr(t, "__name__", "lambda")
            try:
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
