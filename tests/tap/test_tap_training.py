"""Tests for `eesi.systems.tap.train`: the coupled interpolant training step.

Runs as either pytest or a plain script:

    pytest tests/tap/test_tap_training.py
    python tests/tap/test_tap_training.py

Plumbing, not model quality: that a step is finite, that gradients reach both fields,
that the flags actually arrive where they claim to, and that the coupling stays a
data-pairing step with no gradient through it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from eesi.ot import transport_cost
from eesi.systems.tap.data import sample_prior
from eesi.systems.tap.train import make_si_model, tap_step, train_si

N, D, RE_SQR = 6, 3, 4.0


def _model(seed: int = 0, **kw):
    torch.manual_seed(seed)
    return make_si_model(n_particles=N, n_dims=D, hidden_nf=8, n_layers=1, **kw).double()


def _data(B: int = 8, seed: int = 0) -> torch.Tensor:
    """Stand-in reference data: a stretched ideal chain, which is O(3)-invariant."""
    return sample_prior(B, RE_SQR * 2.5, n_particles=N, n_dims=D,
                        generator=torch.Generator().manual_seed(seed))


def test_step_is_finite_and_backprops():
    model = _model()
    losses, _, _ = tap_step(model, _data(), RE_SQR)
    total = losses["b"] + losses["s"]
    assert total.isfinite()
    total.backward()
    assert any(p.grad is not None for p in model.net_b.parameters())
    assert any(p.grad is not None for p in model.net_s.parameters())


def test_step_endpoints_stay_anchored():
    """Both endpoints handed to the loss are on the subspace."""
    model = _model()
    _, x0, x1 = tap_step(model, _data(), RE_SQR)
    assert x0[:, 0].abs().max() < 1e-12
    assert x1[:, 0].abs().max() < 1e-12


def test_no_gradient_flows_through_the_coupling():
    model = _model()
    _, x0, x1 = tap_step(model, _data(), RE_SQR)
    assert not x0.requires_grad and not x1.requires_grad


@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("batch", [True, False])
def test_all_ablation_arms_train(align, batch):
    model = _model()
    losses, _, _ = tap_step(model, _data(), RE_SQR, align=align, batch=batch)
    (losses["b"] + losses["s"]).backward()
    assert any(p.grad is not None for p in model.net_b.parameters())


def test_coupling_lowers_the_regression_target():
    """The full coupling shortens the transport the drift has to learn."""
    model = _model()
    torch.manual_seed(0)
    data = _data(B=32, seed=1)
    costs = {}
    for align, batch in ((True, True), (False, False)):
        torch.manual_seed(7)                       # same base draw for both arms
        _, a, b = tap_step(model, data, RE_SQR, align=align, batch=batch)
        costs[(align, batch)] = transport_cost(a, b).item()
    assert costs[(True, True)] < costs[(False, False)], costs


def test_re_sqr_reaches_the_prior_draw():
    """The prior scale actually threads through the step, rather than being ignored."""
    model = _model()
    data = _data(B=64)
    torch.manual_seed(1)
    _, small, _ = tap_step(model, data, 1.0, align=False, batch=False)
    torch.manual_seed(1)
    _, big, _ = tap_step(model, data, 16.0, align=False, batch=False)
    assert (big ** 2).sum().item() > 4 * (small ** 2).sum().item()


def test_train_si_runs_and_reduces_nothing_catastrophically():
    """A short run completes, logs, and leaves finite parameters."""
    model = _model(seed=2)
    trained, hist = train_si(_data(B=32, seed=3), RE_SQR, steps=5, batch=4,
                             log_every=0, model=model)
    assert len(hist) == 5
    assert all(torch.isfinite(p).all() for p in trained.parameters())


def test_index_feature_flag_reaches_the_nets():
    for flag in (True, False):
        model = _model(index_feature=flag)
        assert model.net_b.index_feature is flag
        assert model.net_s.index_feature is flag
        expected_in = 2 if flag else 1
        assert model.net_b.egnn.embedding.in_features == expected_in


def test_drift_and_score_nets_are_independent():
    """Two separate fields, not one module aliased twice."""
    model = _model()
    assert model.net_b is not model.net_s
    b_ids = {id(p) for p in model.net_b.parameters()}
    s_ids = {id(p) for p in model.net_s.parameters()}
    assert b_ids.isdisjoint(s_ids)


if __name__ == "__main__":
    tests = [
        test_step_is_finite_and_backprops,
        test_step_endpoints_stay_anchored,
        test_no_gradient_flows_through_the_coupling,
        test_coupling_lowers_the_regression_target,
        test_re_sqr_reaches_the_prior_draw,
        test_train_si_runs_and_reduces_nothing_catastrophically,
        test_index_feature_flag_reaches_the_nets,
        test_drift_and_score_nets_are_independent,
    ]
    for _a in (True, False):
        for _b in (True, False):
            tests.append(lambda a=_a, b=_b: test_all_ablation_arms_train(a, b))
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
