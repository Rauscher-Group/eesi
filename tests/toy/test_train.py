"""Tests for `eesi.systems.toy.train`: the loop the toy notebooks used to inline.

These check the invariants, not just that the loop executes: both OT arms actually
train, both target forms (a fixed tensor and a live sampler) are accepted, the
learn_vel / learn_score flags reach the optimizer, and the coupling flag reaches the
coupling.

Runs as either pytest or a plain script:

    pytest tests/toy/test_train.py
    python tests/toy/test_train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch

from eesi.systems.toy.train import make_model, sample_base, toy_step, train

DT = torch.float32
SEED = 0


# ---- fixtures --------------------------------------------------------------


def _mixture(dim=4, n_mixes=3, loc_scaling=2.0, **kw):
    """Build a CPU mixture; `device` defaults to "cuda" in the constructor."""
    from eesi.systems.toy.data import GaussianMixture
    return GaussianMixture(dim=dim, n_mixes=n_mixes, loc_scaling=loc_scaling,
                           log_var_scaling=-1.0, seed=SEED, device="cpu", **kw)


def _small_model(d=4):
    """A model small enough that 60 steps visibly move the loss."""
    return make_model(d, hidden=32, n_layers=2)


def _total(hist):
    """History is [(loss_b, loss_s)]; the trained objective is their sum."""
    return np.asarray(hist).sum(1)


# ---- the loop --------------------------------------------------------------


def test_both_ot_arms_train():
    """Coupled and independent pairings both drive the loss down. Loss going down is
    the assertion; which arm wins is a research question, not a test."""
    target = _mixture()
    for batch_ot in (False, True):
        _, hist = train(target, steps=60, batch=32, lr=3e-3, batch_ot=batch_ot,
                        log_every=0, seed=SEED, model=_small_model())
        tot = _total(hist)
        assert np.isfinite(tot).all(), batch_ot
        assert tot[-20:].mean() < tot[:20].mean(), batch_ot


def test_train_accepts_a_fixed_tensor_target():
    """The 'limited data' regime: one dataset drawn up front, minibatched with
    replacement, rather than a fresh draw per step."""
    data = _mixture().sample((512,))
    model, hist = train(data, steps=60, batch=32, lr=3e-3, log_every=0, seed=SEED,
                        model=_small_model())
    tot = _total(hist)
    assert len(hist) == 60
    assert np.isfinite(tot).all()
    assert tot[-20:].mean() < tot[:20].mean()


def test_train_rejects_a_target_it_cannot_sample():
    target = object()
    with pytest.raises(TypeError, match="sample"):
        train(target, steps=1, batch=4, log_every=0)


def test_train_rejects_a_non_2d_tensor_target():
    with pytest.raises(ValueError, match=r"\(n_data, d\)"):
        train(torch.randn(8, 4, 3), steps=1, batch=4, log_every=0)


def test_dimension_is_inferred_without_perturbing_the_target():
    """`GaussianMixture.dim` states d, so resolving the sampler must not spend a draw
    -- `call_time` is how the notebooks account for target evaluations."""
    target = _mixture()
    assert target.call_time == 0
    train(target, steps=2, batch=8, log_every=0, seed=SEED, model=_small_model())
    assert target.call_time == 2 * 8, "an extra probe draw crept into the setup"


# ---- flags reach the code they name ----------------------------------------


def test_learn_flags_gate_the_loss():
    """`learn_vel=False` must leave every drift parameter bitwise unchanged, and
    vice versa -- the score-only mode of experiments/Toy/40D_GMM.ipynb."""
    target = _mixture()
    for learn_vel, learn_score, frozen in ((False, True, "net_b"), (True, False, "net_s")):
        model = _small_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        train(target, steps=10, batch=16, lr=3e-3, learn_vel=learn_vel,
              learn_score=learn_score, log_every=0, seed=SEED, model=model)
        after = model.state_dict()
        moved = [k for k in before if not torch.equal(before[k], after[k])]
        assert all(not k.startswith(frozen) for k in moved), (frozen, moved)
        assert moved, "nothing trained at all -- the test is blind"


def test_learn_flags_cannot_both_be_off():
    with pytest.raises(ValueError, match="cannot both be False"):
        train(_mixture(), steps=1, batch=4, learn_vel=False, learn_score=False,
              log_every=0)


def test_ot_flag_reaches_the_coupling():
    """The transport cost of the pairs `toy_step` hands to the loss drops when the
    coupling is on. If it did not, `batch_ot` would be decorative."""
    from eesi.systems.toy.ot import toy_transport_cost
    torch.manual_seed(SEED)
    model = _small_model()
    x1 = _mixture().sample((64,))
    costs = {}
    for batch_ot in (False, True):
        g = torch.Generator().manual_seed(1)
        _, a, b = toy_step(model, x1, batch_ot=batch_ot, generator=g)
        costs[batch_ot] = toy_transport_cost(a, b).item()
    assert costs[True] < costs[False]


# ---- shapes and gradients --------------------------------------------------


def test_toy_step_shapes_and_dtype():
    """The step preserves shape/dtype and returns a finite, backpropagatable loss."""
    torch.manual_seed(SEED)
    model = _small_model()
    x1 = _mixture().sample((16,)).to(DT)
    losses, a, b = toy_step(model, x1)
    assert a.shape == (16, 4) and b.shape == (16, 4)
    assert a.dtype == DT and not a.requires_grad
    loss = losses["b"] + losses["s"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_sample_base_is_reproducible_and_standard_normal():
    g = torch.Generator().manual_seed(2)
    x = sample_base(4096, 3, generator=g)
    assert x.shape == (4096, 3) and x.dtype == torch.float32
    assert x.mean().abs() < 0.05 and abs(x.std().item() - 1.0) < 0.05
    assert torch.equal(x, sample_base(4096, 3, generator=torch.Generator().manual_seed(2)))


def test_make_model_widths():
    """`hidden_s` widens the score net alone; unset, both nets match."""
    m = make_model(4, hidden=32, hidden_s=48, n_layers=2)
    assert m.net_b.hidden == 32 and m.net_s.hidden == 48
    assert m.net_b is not m.net_s
    assert make_model(4, hidden=32, n_layers=2).net_s.hidden == 32


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
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
