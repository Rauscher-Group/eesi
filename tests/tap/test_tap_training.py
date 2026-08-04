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

N, D = 6, 3
K, B_LEN = 5.0, 1.0     # the prior's bond parameters
K1, B1 = 1.0, 2.5       # a longer, floppier chain, standing in for the reference data


def _model(seed: int = 0, **kw):
    torch.manual_seed(seed)
    return make_si_model(n_particles=N, n_dims=D, hidden_nf=8, n_layers=1, **kw).double()


def _data(B: int = 8, seed: int = 0) -> torch.Tensor:
    """Stand-in reference data: a longer, floppier chain, which is O(3)-invariant."""
    return sample_prior(B, K1, B1, n_particles=N, n_dims=D,
                        generator=torch.Generator().manual_seed(seed))


def test_step_is_finite_and_backprops():
    model = _model()
    losses, _, _ = tap_step(model, _data(), K, B_LEN)
    total = losses["b"] + losses["s"]
    assert total.isfinite()
    total.backward()
    assert any(p.grad is not None for p in model.net_b.parameters())
    assert any(p.grad is not None for p in model.net_s.parameters())


def test_step_endpoints_stay_anchored():
    """Both endpoints handed to the loss are on the subspace."""
    model = _model()
    _, x0, x1 = tap_step(model, _data(), K, B_LEN)
    assert x0[:, 0].abs().max() < 1e-12
    assert x1[:, 0].abs().max() < 1e-12


def test_no_gradient_flows_through_the_coupling():
    model = _model()
    _, x0, x1 = tap_step(model, _data(), K, B_LEN)
    assert not x0.requires_grad and not x1.requires_grad


@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("batch", [True, False])
def test_all_ablation_arms_train(align, batch):
    model = _model()
    losses, _, _ = tap_step(model, _data(), K, B_LEN, align=align, batch=batch)
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
        _, x0, x1 = tap_step(model, data, K, B_LEN, align=align, batch=batch)
        costs[(align, batch)] = transport_cost(x0, x1).item()
    assert costs[(True, True)] < costs[(False, False)], costs


def test_bond_params_reach_the_prior_draw():
    """Both k and b actually thread through the step, rather than being ignored.

    Two knobs now instead of one, so each is moved separately: a longer equilibrium
    bond at fixed stiffness grows the chain (E[Q^2] = 1.93 -> 9.99 for these values),
    and a floppier spring at fixed b grows it too (E[Q^2] = 1.93 -> 5.13).
    """
    model = _model()
    data = _data(B=64)

    def draw(k, b):
        torch.manual_seed(1)
        _, x0, _ = tap_step(model, data, k, b, align=False, batch=False)
        return (x0 ** 2).sum().item()

    base = draw(K, B_LEN)
    assert draw(K, 3.0) > 4 * base, "b does not reach the prior"
    assert draw(1.0, B_LEN) > 2 * base, "k does not reach the prior"


def test_bond_params_are_not_interchangeable():
    """Passing (k, b) in the wrong order gives a visibly different chain.

    The two parameters are adjacent positional floats, so a transposed call site is a
    plausible bug that no type checker would catch and that would otherwise train
    happily against the wrong prior. E[Q^2] is 7.24 for (5, 2.5) and 26.99 for
    (2.5, 5), so the swap is far outside sampling noise.
    """
    model = _model()
    data = _data(B=64)

    def draw(k, b):
        torch.manual_seed(1)
        _, x0, _ = tap_step(model, data, k, b, align=False, batch=False)
        return (x0 ** 2).sum().item()

    assert draw(2.5, 5.0) > 2 * draw(5.0, 2.5)


def test_train_si_runs_and_reduces_nothing_catastrophically():
    """A short run completes, logs, and leaves finite parameters."""
    model = _model(seed=2)
    trained, hist = train_si(_data(B=32, seed=3), K, B_LEN, steps=5, batch=4,
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
        test_bond_params_reach_the_prior_draw,
        test_bond_params_are_not_interchangeable,
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
