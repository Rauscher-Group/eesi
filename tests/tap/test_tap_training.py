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

import numpy as np
import pytest
import torch

from eesi.ot import transport_cost
from eesi.systems.tap.data import sample_prior
from eesi.systems.tap.train import _hist_keys, make_si_model, tap_step, train_si

N, D = 6, 3
K, B_LEN, GAMMA, COS0 = 5.0, 1.0, 2.0, 0.5    # the prior
K1, B1, GAMMA1, COS1 = 1.0, 2.5, 0.0, 1.0     # stand-in reference data


def _model(seed: int = 0, **kw):
    torch.manual_seed(seed)
    return make_si_model(n_particles=N, n_dims=D, hidden_nf=8, n_layers=1, **kw).double()


def _data(B: int = 8, seed: int = 0) -> torch.Tensor:
    """Stand-in reference data: a longer, floppier chain, which is O(3)-invariant."""
    return sample_prior(B, K1, B1, GAMMA1, COS1, n_particles=N, n_dims=D,
                        generator=torch.Generator().manual_seed(seed))


def test_step_is_finite_and_backprops():
    model = _model()
    losses, _, _ = tap_step(model, _data(), K, B_LEN, GAMMA, COS0)
    total = losses["b"] + losses["s"]
    assert total.isfinite()
    total.backward()
    assert any(p.grad is not None for p in model.net_b.parameters())
    assert any(p.grad is not None for p in model.net_s.parameters())


def test_step_endpoints_stay_anchored():
    """Both endpoints handed to the loss are on the subspace."""
    model = _model()
    _, x0, x1 = tap_step(model, _data(), K, B_LEN, GAMMA, COS0)
    assert x0[:, 0].abs().max() < 1e-12
    assert x1[:, 0].abs().max() < 1e-12


def test_no_gradient_flows_through_the_coupling():
    model = _model()
    _, x0, x1 = tap_step(model, _data(), K, B_LEN, GAMMA, COS0)
    assert not x0.requires_grad and not x1.requires_grad


@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("batch", [True, False])
def test_all_ablation_arms_train(align, batch):
    model = _model()
    losses, _, _ = tap_step(model, _data(), K, B_LEN, GAMMA, COS0, align=align, batch=batch)
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
        _, x0, x1 = tap_step(model, data, K, B_LEN, GAMMA, COS0, align=align, batch=batch)
        costs[(align, batch)] = transport_cost(x0, x1).item()
    assert costs[(True, True)] < costs[(False, False)], costs


def _draw_size(model, data, k, b, gamma, cos_theta_0):
    """sum_j |x_j|^2 over a prior draw, under a fixed seed. Grows with the chain."""
    torch.manual_seed(1)
    _, x0, _ = tap_step(model, data, k, b, gamma, cos_theta_0, align=False, batch=False)
    return (x0 ** 2).sum().item()


def test_prior_params_reach_the_prior_draw():
    """All FOUR parameters thread through the step, rather than any being ignored.

    Each is moved on its own, against the analytic chain size at N=6 (the sum of
    E|x_j|^2, which is what the statistic measures):

        b       1 -> 3     45.1 -> 238.3      longer bonds
        k       5 -> 1     45.1 -> 117.7      floppier bonds
        gamma   2 -> 0     45.1 ->  29.0      bending switched off, chain coils up
        cos    0.5 -> -1   45.1 ->  13.5      bends reversed, chain folds back

    The last two are the ones this change adds, and they matter: a `tap_step` that
    accepted gamma and cos_theta_0 but forgot to forward them to `sample_prior` would
    pass every other test in this file.
    """
    model = _model()
    data = _data(B=64)
    base = _draw_size(model, data, K, B_LEN, GAMMA, COS0)

    assert _draw_size(model, data, K, 3.0, GAMMA, COS0) > 4 * base, "b is ignored"
    assert _draw_size(model, data, 1.0, B_LEN, GAMMA, COS0) > 2 * base, "k is ignored"
    assert _draw_size(model, data, K, B_LEN, 0.0, COS0) < 0.8 * base, "gamma is ignored"
    assert _draw_size(model, data, K, B_LEN, GAMMA, -1.0) < 0.5 * base, "cos is ignored"


def test_prior_params_are_not_interchangeable():
    """A transposed call site is caught -- by magnitude for (k, b), by range for the rest.

    Four adjacent positional floats make a transposition plausible, and no type checker
    would catch one. The two halves fail differently, which is worth knowing:

    (k, b) swapped produces a valid but wrong chain, so it has to be caught numerically:
    E[Q^2] is 7.24 for (5, 2.5) against 26.99 for (2.5, 5), far outside sampling noise.

    (gamma, cos_theta_0) swapped is usually caught by validation instead, because any
    gamma above 1 becomes an out-of-range cosine and `angle_moments` refuses it. That is
    the more robust failure of the two -- loud and immediate rather than a silently
    mis-specified prior -- and it is why cos_theta_0 keeping a hard [-1, 1] bound earns
    its keep.
    """
    model = _model()
    data = _data(B=64)

    assert (_draw_size(model, data, 2.5, 5.0, GAMMA, COS0)
            > 2 * _draw_size(model, data, 5.0, 2.5, GAMMA, COS0))

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        _draw_size(model, data, K, B_LEN, COS0, GAMMA)   # gamma and cosine transposed


def test_train_si_runs_and_reduces_nothing_catastrophically():
    """A short run completes, logs, and leaves finite parameters."""
    model = _model(seed=2)
    trained, hist = train_si(_data(B=32, seed=3), K, B_LEN, GAMMA, COS0, steps=5, batch=4,
                             log_every=0, model=model)
    assert len(hist) == 5
    assert all(torch.isfinite(p).all() for p in trained.parameters())


@pytest.mark.parametrize("entropy,width",
                         [(None, 2), ("dot", 3), ("zdot", 3), ("both", 4)])
def test_entropy_flag_widens_the_history(entropy, width):
    """`entropy` adds columns to `train_si`'s hist -- and only when asked.

    The default arity is load-bearing: the log line, `main`, the TAP notebook's
    `zip(*htot)` and `test_train_si_runs_and_reduces_nothing_catastrophically` above
    all unpack two. "dot" must work despite `TAPEESI` forcing `learn_score=False`;
    with gamma != "none" the denoising objective trains net_s anyway (see `EESI.loss`),
    which the sibling test below pins down.
    """
    model = _model(seed=2)
    _, hist = train_si(_data(B=32, seed=3), K, B_LEN, GAMMA, COS0, steps=5, batch=4,
                       log_every=0, model=model, entropy=entropy)
    assert all(len(row) == width for row in hist), (entropy, len(hist[0]))
    assert np.isfinite(np.asarray(hist)).all(), entropy


def test_entropy_channel_costs_no_extra_net_evaluations():
    """The channel must reuse the loss's own draw, not take a second one.

    Counting forward calls is the direct check: `entropy='both'` runs the two
    antithetic passes the loss needs and nothing more.
    """
    calls = {"b": 0, "s": 0}
    model = _model()
    for name in ("b", "s"):
        net = getattr(model, f"net_{name}")
        fwd = net.forward
        net.forward = (lambda *a, _f=fwd, _n=name, **kw:
                       (calls.__setitem__(_n, calls[_n] + 1), _f(*a, **kw))[1])

    tap_step(model, _data(), K, B_LEN, GAMMA, COS0, entropy="both")
    assert calls == {"b": 2, "s": 2}, calls


def test_score_is_trained_despite_learn_score_off():
    """Why entropy='dot' is legal for TAPEESI: net_s gets a gradient regardless.

    `learn_score` is forced off to disable ISM (the subspace divergence runs under
    no_grad and is not differentiable), not because the score is frozen -- the
    denoising objective still trains it whenever gamma != "none".
    """
    model = _model()
    assert model.learn_score is False and model.gamma != "none"
    losses, _, _ = tap_step(model, _data(), K, B_LEN, GAMMA, COS0)
    (losses["b"] + losses["s"]).backward()
    g = [p.grad for p in model.net_s.parameters() if p.grad is not None]
    assert g and sum(x.abs().sum() for x in g) > 0


def test_bad_entropy_settings_are_rejected():
    """Both rejection paths are errors, not silent defaults.

    An unknown name never reaches `model.loss` -- `_hist_keys` validates it before the
    loop, so a typo fails on step 0 rather than after a long run. "zdot" on a
    gamma="none" model is rejected inside `EESI._check_entropy_channel`: with no latent
    z the conditional score -z/gamma does not exist. Only reachable here by asking for
    it explicitly, since `make_si_model` defaults to gamma="quad".
    """
    with pytest.raises(ValueError, match="entropy must be one of"):
        _hist_keys("bogus")

    model = _model(gamma="none")
    with pytest.raises(ValueError, match="latent"):
        train_si(_data(B=8), K, B_LEN, GAMMA, COS0, steps=1, batch=4,
                 log_every=0, model=model, entropy="zdot")


@pytest.mark.parametrize("time_order,index_order", [(0, 0), (1, 3), (4, 4)])
def test_feature_settings_reach_the_nets(time_order, index_order):
    """The flags and the orders decide the embedding width, on BOTH nets.

    `make_si_model` builds two independent `TAPDynamics`, so a kwarg dropped from
    `net_kw` would leave one of them silently on the defaults -- a drift and a score
    field with different architectures, which nothing else here would notice.
    """
    for flag in (True, False):
        model = _model(index_feature=flag, time_order=time_order,
                       index_order=index_order)
        expected_in = (1 + 2 * time_order) + ((1 + 2 * index_order) if flag else 0)
        for net in (model.net_b, model.net_s):
            assert net.index_feature is flag
            assert (net.time_order, net.index_order) == (time_order, index_order)
            assert net.egnn.embedding.in_features == expected_in


def test_bond_feature_flag_reaches_the_nets():
    """Both arms keep in_edge_nf = 1, so they differ only in what that channel means.

    That is the point of falling back to the squared distance rather than to no
    channel at all: identical parameter counts make a training comparison between the
    arms a statement about the feature.
    """
    widths = set()
    for flag in (True, False):
        model = _model(bond_feature=flag)
        assert model.net_b.bond_feature is flag
        assert model.net_s.bond_feature is flag
        widths.add(sum(p.numel() for p in model.parameters()))
    assert len(widths) == 1


def test_the_resumption_keywords_change_nothing_at_their_defaults():
    """`opt`, `callback`, `start_step` and `seed=None` are for `eesi.systems.tap.run`.

    They exist so a long run can be checkpointed and resumed without a second copy of
    this loop living in the runner. The contract that makes that safe is that the loop
    is byte-identical when they are left alone -- so the same seed still gives the same
    history, and passing an equivalent optimizer explicitly gives the same history too.
    """
    data = _data(B=32, seed=3)
    _, base = train_si(data, K, B_LEN, GAMMA, COS0, steps=4, batch=4, log_every=0,
                       model=_model(seed=2), seed=11)

    model = _model(seed=2)
    _, given_opt = train_si(data, K, B_LEN, GAMMA, COS0, steps=4, batch=4, log_every=0,
                            model=model, seed=11,
                            opt=torch.optim.Adam(model.parameters(), lr=1e-3))
    assert given_opt == base, "supplying the optimizer changed the arithmetic"

    seen = []
    _, with_cb = train_si(data, K, B_LEN, GAMMA, COS0, steps=4, batch=4, log_every=0,
                          model=_model(seed=2), seed=11,
                          callback=lambda step, row, x0, x1: seen.append((step, row)))
    assert with_cb == base, "the callback changed the arithmetic"
    assert [s for s, _ in seen] == [0, 1, 2, 3]
    assert [r for _, r in seen] == base, "the callback saw a different history"


def test_a_callback_returning_false_stops_the_loop_cleanly():
    """How a SIGTERM gets a checkpoint written instead of losing the stage."""
    model = _model(seed=2)
    _, hist = train_si(_data(B=32, seed=3), K, B_LEN, GAMMA, COS0, steps=100, batch=4,
                       log_every=0, model=model,
                       callback=lambda step, row, x0, x1: step < 2)
    assert len(hist) == 3, "the stopping step's own row must still be recorded"


def test_start_step_and_seed_none_resume_mid_stage():
    """Four steps, then four more from a restored RNG state, equals eight straight.

    This is the property the whole resumable-run design rests on, checked here on the
    loop itself before any of the checkpoint plumbing is involved.
    """
    data = _data(B=32, seed=3)
    straight_model = _model(seed=2)
    _, straight = train_si(data, K, B_LEN, GAMMA, COS0, steps=8, batch=4, log_every=0,
                           model=straight_model, seed=11)

    model = _model(seed=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _, first = train_si(data, K, B_LEN, GAMMA, COS0, steps=8, batch=4, log_every=0,
                        model=model, seed=11, opt=opt,
                        callback=lambda step, row, x0, x1: step < 3)
    rng = torch.get_rng_state()          # what a checkpoint stores

    torch.manual_seed(999)               # something else happens in between
    torch.randn(64)
    torch.set_rng_state(rng)
    _, second = train_si(data, K, B_LEN, GAMMA, COS0, steps=8, batch=4, log_every=0,
                         model=model, seed=None, opt=opt, start_step=4,
                         callback=None)

    assert first + second == straight, "a resumed run diverged from an uninterrupted one"
    for a, b in zip(straight_model.parameters(), model.parameters()):
        assert torch.equal(a, b), "resumed parameters differ"


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
        test_prior_params_reach_the_prior_draw,
        test_prior_params_are_not_interchangeable,
        test_train_si_runs_and_reduces_nothing_catastrophically,
        test_entropy_channel_costs_no_extra_net_evaluations,
        test_score_is_trained_despite_learn_score_off,
        test_bad_entropy_settings_are_rejected,
        test_the_resumption_keywords_change_nothing_at_their_defaults,
        test_a_callback_returning_false_stops_the_loop_cleanly,
        test_start_step_and_seed_none_resume_mid_stage,
        test_bond_feature_flag_reaches_the_nets,
        test_drift_and_score_nets_are_independent,
    ]
    for _a in (True, False):
        for _b in (True, False):
            tests.append(lambda a=_a, b=_b: test_all_ablation_arms_train(a, b))
    for _e, _w in ((None, 2), ("dot", 3), ("zdot", 3), ("both", 4)):
        tests.append(lambda e=_e, w=_w: test_entropy_flag_widens_the_history(e, w))
    for _to, _io in ((0, 0), (1, 3), (4, 4)):
        tests.append(lambda t=_to, i=_io: test_feature_settings_reach_the_nets(t, i))
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
