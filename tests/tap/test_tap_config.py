"""Tests for `eesi.systems.tap.config`: the TAP run schema and prior resolution.

Runs as either pytest or a plain script:

    pytest tests/tap/test_tap_config.py
    python tests/tap/test_tap_config.py

Two things are being pinned. First, that a wrong config fails at load with a message
naming the key -- the whole point of the layer. Second, that the three routes to a
prior parameter (given, measured, fitted) produce what the notebook's calibration
cell produced, since a run whose prior has drifted from its data is a run whose
entropy estimate means nothing.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from eesi.systems.tap.config import (ParamSpec, ResolvedPrior, config_from_dict,
                                     config_to_dict, load_config, prior_report,
                                     resolve_prior)
from eesi.systems.tap.data import (bond_cosines, bond_vectors, end_to_end_sq,
                                   sample_prior, solve_cos_theta_0)

N = 6
BASE = {
    "name": "unit",
    "data": {"path": "data/tap_N20_Pe0.npy", "n": 1000, "n_particles": N},
    "prior": {"k": 100.0, "b": 1.0,
              "gamma": {"mode": "measure", "estimator": "inv_two_var_cos"},
              "cos_theta_0": {"mode": "solve", "target": "end_to_end_sq"}},
    "model": {"hidden_nf": 8, "n_layers": 1, "gamma": "sqrt", "gamma_scale": 0.2},
    "train": {"batch": 32, "entropy": "both",
              "stages": [{"name": "coarse", "steps": 10, "lr": 1.0e-4}]},
}


def cfg_dict(**patch):
    raw = copy.deepcopy(BASE)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return raw


def _data(n=4000, k=8.0, b=1.0, gamma=2.0, cos0=0.4, seed=0):
    """Chains from a KNOWN prior, so the measured estimators have a right answer."""
    return sample_prior(n, k, b, gamma, cos0, n_particles=N,
                        generator=torch.Generator().manual_seed(seed))


# --- schema -----------------------------------------------------------------


def test_a_minimal_config_loads_with_the_documented_defaults():
    cfg = config_from_dict(cfg_dict())
    assert cfg.name == "unit" and cfg.out_dir == "runs" and cfg.dtype == "float64"
    assert cfg.device == "auto" and cfg.seed == 0
    assert cfg.model.path == "linear" and cfg.model.eps == 1e-6
    assert cfg.entropy.methods == ("dot", "zdot", "div")
    assert cfg.sample.integrator == "rk4"
    assert cfg.checkpoint.keep == 3
    assert cfg.entropy_channel == "both"
    assert cfg.run_root == Path("runs/unit")


def test_a_bare_number_is_shorthand_for_a_fixed_parameter():
    cfg = config_from_dict(cfg_dict())
    assert cfg.prior.k == ParamSpec("fixed", value=100.0)
    assert cfg.prior.b == ParamSpec("fixed", value=1.0)
    assert cfg.prior.gamma.mode == "measure"
    assert cfg.prior.cos_theta_0 == ParamSpec("solve", target="end_to_end_sq")


def test_the_pyyaml_float_spelling_works_for_a_fixed_parameter():
    """`k: 1e2` reaches this code as the string "1e2" -- see eesi.config."""
    cfg = config_from_dict(cfg_dict(prior={"k": "1e2"}))
    assert cfg.prior.k.value == 100.0


@pytest.mark.parametrize("patch,match", [
    ({"nmae": "x"}, "did you mean 'name'"),
    ({"model": {"hidden_nff": 8}}, "did you mean 'hidden_nf'"),
    ({"model": {"path": "cubic"}}, r"model\.path: must be one of"),
    ({"model": {"gamma": "linear"}}, r"model\.gamma: must be one of"),
    ({"model": {"hidden_nf": 0}}, r"hidden_nf: must be positive"),
    ({"model": {"eps": 0.0}}, r"eps: must be positive"),
    ({"model": {"time_order": -1}}, r"time_order: must be >= 0"),
    ({"data": {"n_dims": 2}}, "3D only"),
    ({"data": {"n": 0}}, r"data\.n: must be positive"),
    ({"dtype": "float16"}, r"dtype: must be one of"),
])
def test_malformed_configs_are_rejected_by_key(patch, match):
    with pytest.raises(ValueError, match=match):
        config_from_dict(cfg_dict(**patch))


def test_missing_name_is_an_error():
    raw = cfg_dict()
    del raw["name"]
    with pytest.raises(ValueError, match="missing required key 'name'"):
        config_from_dict(raw)


@pytest.mark.parametrize("key", ["k", "b", "gamma", "cos_theta_0"])
def test_every_prior_parameter_must_be_stated(key):
    """Defaulting one silently would put a different chain under the flow."""
    raw = cfg_dict()
    del raw["prior"][key]
    with pytest.raises(ValueError, match=f"missing required key '{key}'"):
        config_from_dict(raw)


def test_the_two_non_configurable_model_knobs_say_why():
    with pytest.raises(ValueError, match="TAPEESI forces it False"):
        config_from_dict(cfg_dict(model={"learn_score": True}))
    with pytest.raises(ValueError, match="not reachable from a config"):
        config_from_dict(cfg_dict(model={"coords_range": 5.0}))


def test_solve_is_only_defined_for_cos_theta_0():
    with pytest.raises(ValueError, match="'solve' is only defined for cos_theta_0"):
        config_from_dict(cfg_dict(prior={"k": {"mode": "solve"}}))


def test_estimators_are_per_parameter():
    """A bond statistic must not be reachable as a bending constant."""
    with pytest.raises(ValueError, match="must be one of \\['inv_var_bond'\\] for 'k'"):
        config_from_dict(cfg_dict(prior={"k": {"mode": "measure",
                                               "estimator": "mean_cos"}}))


def test_measure_defaults_to_the_one_estimator_that_parameter_has():
    cfg = config_from_dict(cfg_dict(prior={"gamma": {"mode": "measure"}}))
    assert cfg.prior.gamma.estimator == "inv_two_var_cos"


def test_gamma_none_rejects_the_zdot_channel_at_load_not_on_step_zero():
    with pytest.raises(ValueError, match="needs a latent schedule"):
        config_from_dict(cfg_dict(model={"gamma": "none"}))
    with pytest.raises(ValueError, match="undefined at model.gamma='none'"):
        config_from_dict(cfg_dict(model={"gamma": "none"},
                                  train={**BASE["train"], "entropy": "dot",
                                         "stages": [{"name": "a", "steps": 1,
                                                     "lr": 1.0e-4}]}))


def test_entropy_methods_and_flow_channels_are_checked_against_what_exists():
    with pytest.raises(ValueError, match=r"entropy\.methods: must be one of"):
        config_from_dict(cfg_dict(entropy={"methods": ["dot", "bogus"]}))
    with pytest.raises(ValueError, match=r"flow_channels: must be one of"):
        config_from_dict(cfg_dict(entropy={"flow_channels": ["zdot"]}))


def test_an_entropy_job_that_would_do_nothing_is_rejected():
    with pytest.raises(ValueError, match="would do nothing"):
        config_from_dict(cfg_dict(entropy={"batches": 0, "flow_n": 0}))


def test_half_a_warm_start_is_rejected():
    with pytest.raises(ValueError, match="net_b and net_s must be given together"):
        config_from_dict(cfg_dict(init={"net_b": "b.pth"}))
    with pytest.raises(ValueError, match="not both"):
        config_from_dict(cfg_dict(init={"checkpoint": "c.pt", "net_b": "b.pth",
                                        "net_s": "s.pth"}))


def test_sde_without_a_diffusion_coefficient_is_rejected():
    with pytest.raises(ValueError, match="positive diffusion coefficient"):
        config_from_dict(cfg_dict(sample={"integrator": "sde"}))


def test_config_round_trips_through_its_dict_form():
    """What makes `resolved.json` reproducible rather than merely informative."""
    cfg = config_from_dict(cfg_dict())
    again = config_from_dict(config_to_dict(cfg), source=cfg.source)
    assert again == cfg


def test_round_trip_survives_every_prior_mode():
    raw = cfg_dict(prior={"k": {"mode": "measure", "estimator": "inv_var_bond"},
                          "b": 1.0,
                          "gamma": {"mode": "measure"},
                          "cos_theta_0": {"mode": "solve", "target": 44.28}})
    cfg = config_from_dict(raw)
    assert config_from_dict(config_to_dict(cfg), source=cfg.source) == cfg


def test_load_config_reads_yaml_and_applies_overrides(tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg_dict()))
    cfg = load_config(p, ["name=other", "model.hidden_nf=64",
                          "train.stages.0.lr=1e-6"])
    assert cfg.name == "other" and cfg.model.hidden_nf == 64
    assert cfg.stages[0].lr == 1e-6      # the "1e-6" string, floated by the schema
    assert cfg.source == str(p)


def test_the_shipped_config_still_loads():
    """`experiments/TAP/configs/tap_N20_Pe0.yaml` is documentation people copy from.

    A schema change that invalidates it has to fail here rather than the first time
    someone launches a run from it.
    """
    path = (Path(__file__).resolve().parents[2] / "experiments" / "TAP" / "configs"
            / "tap_N20_Pe0.yaml")
    cfg = load_config(path)
    assert cfg.name == "tap_N20_Pe0"
    assert cfg.data.n_particles == 20 and cfg.dtype == "float64"
    assert [s.name for s in cfg.stages] == ["coarse", "fine"]
    assert [s.lr for s in cfg.stages] == [1e-4, 1e-5]
    assert cfg.entropy_channel == "both" and cfg.stages[0].batch == 256
    # The notebook's model, so a run from this file reproduces its numbers.
    assert (cfg.model.hidden_nf, cfg.model.n_layers) == (32, 3)
    assert (cfg.model.path, cfg.model.gamma, cfg.model.gamma_scale) == \
           ("linear", "sqrt", 0.2)
    assert cfg.prior.k.value == 100.0 and cfg.prior.b.value == 1.0
    assert cfg.prior.gamma.mode == "measure"
    assert cfg.prior.cos_theta_0 == ParamSpec("solve", target="end_to_end_sq")
    assert config_from_dict(config_to_dict(cfg), source=cfg.source) == cfg


# --- prior resolution -------------------------------------------------------


def test_fixed_parameters_pass_through_untouched():
    prior = resolve_prior(config_from_dict(
        cfg_dict(prior={"k": 100.0, "b": 1.0, "gamma": 2.0, "cos_theta_0": 0.5})).prior,
        _data(), n_particles=N)
    assert prior.as_tuple() == (100.0, 1.0, 2.0, 0.5)
    assert set(prior.provenance.values()) == {"fixed"}


def test_the_bond_estimators_recover_a_stiff_chains_own_parameters():
    """`k = 1/var(Q)` and `b = E[Q]` invert the bond law only when the chain is stiff.

    The bond weight carries a q^2 Jacobian, so var(Q) is 1/k only to the extent that
    the Gaussian is narrow against b. At the physical k=100, b=1 that costs a couple
    of percent; at k=8 the same recipe returns 9.7, which is why this test is pinned
    to the regime TAP actually runs in rather than to an arbitrary one.

    The angular recipe is a different animal and is checked separately below.
    """
    k, b = 100.0, 1.0
    x = _data(n=20_000, k=k, b=b, gamma=20.0, cos0=0.9, seed=3)
    cfg = config_from_dict(cfg_dict(prior={
        "k": {"mode": "measure"}, "b": {"mode": "measure"},
        "gamma": {"mode": "measure"}, "cos_theta_0": {"mode": "measure"}})).prior
    prior = resolve_prior(cfg, x, n_particles=N)

    assert prior.k == pytest.approx(k, rel=0.05)
    assert prior.b == pytest.approx(b, rel=0.05)
    assert prior.cos_theta_0 == pytest.approx(bond_cosines(x).mean().item(), rel=1e-9)


def test_the_bending_recipe_is_a_definition_not_an_inverse():
    """`gamma = 1/(2 var(cos theta))` does NOT return the gamma that made the data.

    Worth a test of its own because it looks like it should. The bending weight lives
    on the sphere, so var(cos theta) picks up the sin theta measure as well as the
    truncation to [-1, 1]: generating at gamma=20 and applying the recipe gives ~38.
    That is not a bug -- the recipe is how the notebook DEFINES the prior's bending
    constant from data, and `cos_theta_0` is then fitted to fix the resulting size.
    A future reader who "corrects" this into an inverse would silently change every
    prior the runner builds.
    """
    x = _data(n=20_000, k=100.0, b=1.0, gamma=20.0, cos0=0.9, seed=3)
    cfg = config_from_dict(cfg_dict(prior={"k": 100.0, "b": 1.0,
                                           "gamma": {"mode": "measure"},
                                           "cos_theta_0": 0.9})).prior
    prior = resolve_prior(cfg, x, n_particles=N)

    assert prior.gamma == pytest.approx(1.0 / (2.0 * bond_cosines(x).var().item()))
    assert prior.gamma > 1.5 * 20.0, "the recipe is being read as an inverse of gamma"


def test_each_measured_parameter_reads_its_own_statistic():
    """A transposed estimator table would still produce four plausible numbers."""
    x = _data(n=8000, seed=1)
    cfg = config_from_dict(cfg_dict(prior={
        "k": {"mode": "measure"}, "b": {"mode": "measure"},
        "gamma": {"mode": "measure"}, "cos_theta_0": {"mode": "measure"}})).prior
    prior = resolve_prior(cfg, x, n_particles=N)
    q, cos = bond_vectors(x).norm(dim=-1), bond_cosines(x)

    assert prior.k == pytest.approx(1.0 / q.var().item())
    assert prior.b == pytest.approx(q.mean().item())
    assert prior.gamma == pytest.approx(1.0 / (2.0 * cos.var().item()))
    assert prior.cos_theta_0 == pytest.approx(cos.mean().item())


def test_solve_reproduces_solve_cos_theta_0_exactly():
    """The notebook's cell 7, as a config. Any drift here desynchronises a run's
    prior from the data it was calibrated against."""
    x = _data(n=8000, seed=2)
    prior = resolve_prior(config_from_dict(cfg_dict()).prior, x, n_particles=N)
    expect = solve_cos_theta_0(prior.k, prior.b, prior.gamma,
                               end_to_end_sq(x).mean().item(), N)
    assert prior.cos_theta_0 == expect
    assert prior.k == 100.0 and prior.b == 1.0        # k, b stay the Hamiltonian's


def test_solve_accepts_a_literal_target():
    x = _data(n=4000, seed=4)
    cfg = config_from_dict(cfg_dict(prior={"k": 8.0, "b": 1.0, "gamma": 2.0,
                                           "cos_theta_0": {"mode": "solve",
                                                           "target": 12.0}})).prior
    prior = resolve_prior(cfg, x, n_particles=N)
    assert prior.cos_theta_0 == solve_cos_theta_0(8.0, 1.0, 2.0, 12.0, N)
    assert prior.provenance["cos_theta_0"].startswith("solve:value=12")


def test_an_unreachable_target_raises_the_physical_error_unwrapped():
    """`solve_cos_theta_0` already says what to do about it -- raise gamma."""
    x = _data(n=2000, seed=5)
    cfg = config_from_dict(cfg_dict(prior={"k": 8.0, "b": 1.0, "gamma": 0.0,
                                           "cos_theta_0": {"mode": "solve",
                                                           "target": 1e6}})).prior
    with pytest.raises(ValueError, match="unreachable at gamma"):
        resolve_prior(cfg, x, n_particles=N)


def test_provenance_records_all_four_routes():
    x = _data(n=4000, seed=6)
    cfg = config_from_dict(cfg_dict(prior={
        "k": 100.0, "b": {"mode": "measure"}, "gamma": {"mode": "measure"},
        "cos_theta_0": {"mode": "solve", "target": "end_to_end_sq"}})).prior
    p = resolve_prior(cfg, x, n_particles=N)
    assert p.provenance["k"] == "fixed"
    assert p.provenance["b"] == "measure:mean_bond"
    assert p.provenance["gamma"] == "measure:inv_two_var_cos"
    assert p.provenance["cos_theta_0"].startswith("solve:end_to_end_sq=")


def test_resolved_prior_round_trips_through_the_checkpoint_form():
    """A resume reads the prior back out of the checkpoint, never re-measuring it."""
    p = resolve_prior(config_from_dict(cfg_dict()).prior, _data(seed=7), n_particles=N)
    assert ResolvedPrior.from_dict(p.to_dict()) == p


def test_prior_report_covers_the_notebooks_table():
    x = _data(n=2000, seed=8)
    p = resolve_prior(config_from_dict(cfg_dict()).prior, x, n_particles=N)
    text = prior_report(p, x, n_particles=N, n_samples=500,
                        generator=torch.Generator().manual_seed(0))
    for token in ("provenance", "cos theta", "S[p0]", "E[Re^2]", "prior", "data",
                  "analytic"):
        assert token in text
    assert "prior " in prior_report(p, x, n_particles=N, n_samples=0)


if __name__ == "__main__":
    import tempfile

    simple = [
        test_a_minimal_config_loads_with_the_documented_defaults,
        test_a_bare_number_is_shorthand_for_a_fixed_parameter,
        test_the_pyyaml_float_spelling_works_for_a_fixed_parameter,
        test_missing_name_is_an_error,
        test_the_two_non_configurable_model_knobs_say_why,
        test_solve_is_only_defined_for_cos_theta_0,
        test_estimators_are_per_parameter,
        test_measure_defaults_to_the_one_estimator_that_parameter_has,
        test_gamma_none_rejects_the_zdot_channel_at_load_not_on_step_zero,
        test_entropy_methods_and_flow_channels_are_checked_against_what_exists,
        test_an_entropy_job_that_would_do_nothing_is_rejected,
        test_half_a_warm_start_is_rejected,
        test_sde_without_a_diffusion_coefficient_is_rejected,
        test_config_round_trips_through_its_dict_form,
        test_round_trip_survives_every_prior_mode,
        test_the_shipped_config_still_loads,
        test_fixed_parameters_pass_through_untouched,
        test_the_bond_estimators_recover_a_stiff_chains_own_parameters,
        test_the_bending_recipe_is_a_definition_not_an_inverse,
        test_each_measured_parameter_reads_its_own_statistic,
        test_solve_reproduces_solve_cos_theta_0_exactly,
        test_solve_accepts_a_literal_target,
        test_an_unreachable_target_raises_the_physical_error_unwrapped,
        test_provenance_records_all_four_routes,
        test_resolved_prior_round_trips_through_the_checkpoint_form,
        test_prior_report_covers_the_notebooks_table,
    ]
    for _p, _m in [({"nmae": "x"}, "name"), ({"model": {"hidden_nff": 8}}, "hidden_nf"),
                   ({"model": {"path": "cubic"}}, "path"), ({"data": {"n_dims": 2}}, "3D")]:
        simple.append(lambda p=_p, m=_m: test_malformed_configs_are_rejected_by_key(p, m))
    for _k in ("k", "b", "gamma", "cos_theta_0"):
        simple.append(lambda k=_k: test_every_prior_parameter_must_be_stated(k))
    with tempfile.TemporaryDirectory() as _d:
        simple.append(lambda: test_load_config_reads_yaml_and_applies_overrides(Path(_d)))

        failed = 0
        for t in simple:
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
        print(f"\n{len(simple) - failed}/{len(simple)} passed")
        if failed:
            raise SystemExit(1)
