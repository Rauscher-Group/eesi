"""Tests for the coupled training loops (plans/EQOT_PLAN.md Phases B4 / C4).

    eesi/train/lj13.py   flow matching, S(13) x SO(3) coupling
    eesi/train/xy.py     xyEESI interpolant, Z2 x U(1) coupling

These check the invariants the plan calls out -- mean-freeness, no gradient through the
coupling, all four ablation arms wired -- not just that the loops execute.

Runs as either pytest or a plain script:

    pytest tests/test_training.py
    python tests/test_training.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

import numpy as np
import torch

from eesi.config import AveragingConfig
from eesi.systems.lj13.train import flow_matching_loss, train as train_lj13
from eesi.systems.xy.train import make_model, sample_base, train as train_xy, xy_step

DT = torch.float64


# ---- fixtures --------------------------------------------------------------


def _lj13_data(n: int = 256, seed: int = 0) -> torch.Tensor:
    """Stand-in for the OSF samples: a G-invariant cluster distribution. Keeps the
    tests free of the 1.6 GB .npy, which is not in the repo."""
    from scipy.spatial.transform import Rotation
    from eesi.systems.lj13.data import sample_prior
    base = sample_prior(1, generator=torch.Generator().manual_seed(99))
    x = base.repeat(n, 1, 1) * 1.5
    R = torch.as_tensor(Rotation.random(n, random_state=seed).as_matrix())
    x = torch.einsum('bij,bnj->bni', R, x)
    return x - x.mean(1, keepdim=True)


def _xy_data(n: int = 256, N: int = 8, seed: int = 0) -> torch.Tensor:
    """Chain-correlated angles with a uniform global phase: Z2 x U(1)-invariant,
    not S(N)-invariant. Same construction as tests/test_ot.py."""
    from eesi.systems.xy.gnn import angle_wrap
    g = torch.Generator().manual_seed(seed)
    step = 0.4 * torch.randn(n, N, generator=g, dtype=DT)
    return angle_wrap(torch.cumsum(step, 1)
                      + 2 * np.pi * torch.rand(n, 1, generator=g, dtype=DT))


# ---- LJ13 (Phase B4) -------------------------------------------------------


# Faithful to en_flows and present in the released checkpoint: LJ13Dynamics reads only
# the coordinate output, so the node-feature readout and the last layer's node MLP feed
# nothing. See the `eesi.systems.lj13.dynamics` module docstring.
_LJ13_DEAD = ("egnn.embedding_out.", "egnn.gcl_2.node_mlp.")


def test_lj13_loss_is_finite_and_backprops():
    from eesi.systems.lj13.data import sample_prior
    from eesi.systems.lj13.dynamics import LJ13Dynamics
    torch.manual_seed(0)
    net = LJ13Dynamics().to(DT)
    x1 = _lj13_data(16)
    x0 = sample_prior(16, generator=torch.Generator().manual_seed(1))
    loss, _, _ = flow_matching_loss(net, x0, x1)
    assert torch.isfinite(loss)
    loss.backward()
    live = [(n, p) for n, p in net.named_parameters()
            if not n.startswith(_LJ13_DEAD)]
    assert live
    for n, p in live:
        assert p.grad is not None, f"{n} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{n} has non-finite gradient"


def test_lj13_unused_parameters():
    """Pins the dead weight so it is a documented property, not a surprise.

    If this fails, the architecture changed: either something started using the node
    readout (good -- update _LJ13_DEAD), or a live path went dead (bad).
    """
    from eesi.systems.lj13.data import sample_prior
    from eesi.systems.lj13.dynamics import LJ13Dynamics
    torch.manual_seed(0)
    net = LJ13Dynamics().to(DT)
    x0 = sample_prior(8, generator=torch.Generator().manual_seed(1))
    flow_matching_loss(net, x0, _lj13_data(8))[0].backward()
    dead = {n for n, p in net.named_parameters() if p.grad is None}
    assert dead == {n for n, _ in net.named_parameters() if n.startswith(_LJ13_DEAD)}
    n_dead = sum(p.numel() for n, p in net.named_parameters() if n in dead)
    assert (n_dead, sum(p.numel() for p in net.parameters())) == (3169, 22468)


def test_lj13_coupling_output_is_mean_free():
    """The plan requires mean-freeness after coupling AND after interpolation: the
    prior lives on the 36-dim mean-zero subspace, not 39."""
    from eesi.systems.lj13.data import sample_prior
    from eesi.systems.lj13.dynamics import LJ13Dynamics
    torch.manual_seed(0)
    net = LJ13Dynamics().to(DT)
    x0 = sample_prior(16, generator=torch.Generator().manual_seed(2))
    _, a, b = flow_matching_loss(net, x0, _lj13_data(16))
    assert a.mean(1).abs().max() < 1e-12
    assert b.mean(1).abs().max() < 1e-12


def test_lj13_accepts_per_sample_time():
    """Regression: `h = ones(B*13, 1) * t` silently broadcasts a [B] time vector to
    (B*13, B) instead of (B*13, 1). Sampling only ever passes a scalar, so this never
    surfaced -- but flow-matching training needs a per-sample t.

    Row-by-row agreement is `allclose`, not `equal`: batched and single-row ops
    reassociate differently in fp (~1.8e-15 either way, t branch or not).
    """
    from eesi.systems.lj13.data import sample_prior
    from eesi.systems.lj13.dynamics import LJ13Dynamics
    torch.manual_seed(0)
    net = LJ13Dynamics().to(DT)
    x = sample_prior(6, generator=torch.Generator().manual_seed(8))
    ts = torch.tensor([0.1, 0.25, 0.4, 0.55, 0.7, 0.85], dtype=DT)
    with torch.no_grad():
        v = net(ts, x)
        rows = torch.stack([net(ts[i].item(), x[i:i+1])[0] for i in range(6)])
        assert v.shape == x.shape
        assert torch.allclose(v, rows, rtol=1e-12, atol=1e-12)
        # a constant vector t must reproduce the scalar path
        assert torch.allclose(net(torch.full((6,), 0.3, dtype=DT), x), net(0.3, x))
    try:
        net(torch.rand(3, dtype=DT), x)
        raise AssertionError("mismatched t length was accepted")
    except ValueError:
        pass


def test_lj13_velocity_output_is_mean_free():
    """v must map the mean-zero subspace to itself, or training leaks into the 3
    translational directions the prior does not support."""
    from eesi.systems.lj13.data import sample_prior
    from eesi.systems.lj13.dynamics import LJ13Dynamics
    torch.manual_seed(0)
    net = LJ13Dynamics().to(DT)
    v = net(0.3, sample_prior(8, generator=torch.Generator().manual_seed(3)))
    assert v.mean(1).abs().max() < 1e-12


def test_lj13_no_gradient_flows_through_the_coupling():
    """The coupling is a data-pairing step. If x1 (a leaf) picks up a gradient via the
    coupling rather than only via the loss target, no_grad has been lost somewhere."""
    from eesi.systems.lj13.data import sample_prior
    from eesi.systems.lj13.dynamics import LJ13Dynamics
    from eesi.systems.lj13.ot import equivariant_ot_couple
    torch.manual_seed(0)
    x1 = _lj13_data(8).requires_grad_(True)
    x0 = sample_prior(8, generator=torch.Generator().manual_seed(4)).requires_grad_(True)
    a, b = equivariant_ot_couple(x0, x1)
    assert not a.requires_grad and not b.requires_grad


def test_lj13_all_ablation_arms_train():
    """All four align/batch arms run and reduce the loss."""
    data = _lj13_data(128)
    for align in (False, True):
        for batch in (False, True):
            _, hist = train_lj13(data, steps=60, batch=8, lr=3e-3, align=align,
                                 batch_ot=batch, log_every=0, seed=0)
            assert np.isfinite(hist).all()
            assert np.mean(hist[-20:]) < np.mean(hist[:20]), (align, batch)


def test_lj13_averaging_populates_avg_without_leaking_into_the_state_dict():
    """`net.avg` must be set with `object.__setattr__`, not plain assignment --
    plain assignment would register the shadow as a submodule and put "avg.*" keys
    into `net.state_dict()`, breaking every `torch.save(net.state_dict(), ...)`."""
    data = _lj13_data(32)
    net, _ = train_lj13(data, steps=5, batch=8, lr=3e-3, log_every=0, seed=0,
                        averaging=AveragingConfig(kind="linear"))
    assert net.avg is not None and net.avg is not net
    assert not any(k.startswith("avg.") for k in net.state_dict())

    net_off, _ = train_lj13(data, steps=5, batch=8, lr=3e-3, log_every=0, seed=0)
    assert net_off.avg is None


def test_lj13_coupling_lowers_the_regression_target():
    """The point of the coupling: ||x1-x0||^2 is the regression target, so a lower
    transport cost is a lower-variance target. This is why it helps at all."""
    from eesi.systems.lj13.data import sample_prior
    from eesi.ot import transport_cost
    from eesi.systems.lj13.ot import equivariant_ot_couple
    x1, x0 = _lj13_data(32), sample_prior(32, generator=torch.Generator().manual_seed(5))
    off = transport_cost(*equivariant_ot_couple(x0, x1, align=False, batch=False))
    on = transport_cost(*equivariant_ot_couple(x0, x1, align=True, batch=True))
    assert on < off


def test_lj13_si_entropy_flag_widens_the_history():
    """`entropy` adds columns to `train_si`'s hist -- and only when asked.

    The default arity is load-bearing: the log line, `main` and the LJ notebook's
    unpacking all expect two. "dot" must work despite `LJ13EESI` forcing
    `learn_score=False`; with gamma != "none" the denoising objective trains net_s
    anyway (see `EESI.loss`), which the sibling test below pins down.
    """
    from eesi.systems.lj13.train import make_si_model, train_si
    data = _lj13_data(64)
    widths = {None: 2, "dot": 3, "zdot": 3, "both": 4}
    for entropy, width in widths.items():
        model = make_si_model(hidden_nf=8, n_layers=2)
        _, hist = train_si(data, steps=5, batch=8, lr=3e-3, log_every=0, seed=0,
                           model=model, entropy=entropy)
        assert all(len(row) == width for row in hist), (entropy, len(hist[0]))
        assert np.isfinite(np.asarray(hist)).all(), entropy


def test_lj13_si_score_is_trained_despite_learn_score_off():
    """Why entropy='dot' is legal for LJ13EESI: net_s gets a gradient regardless.

    `learn_score` is forced off to disable ISM (its subspace divergence is not
    differentiable), not because the score is frozen -- the denoising objective
    still trains it whenever gamma != "none".
    """
    from eesi.systems.lj13.train import make_si_model, si_step
    torch.manual_seed(0)
    model = make_si_model(hidden_nf=8, n_layers=2).to(DT)
    assert model.learn_score is False and model.gamma != "none"
    losses, _, _ = si_step(model, _lj13_data(8))
    (losses["b"] + losses["s"]).backward()
    g = [p.grad for p in model.net_s.parameters() if p.grad is not None]
    assert g and sum(x.abs().sum() for x in g) > 0


def test_lj13_si_entropy_channel_costs_no_extra_net_evaluations():
    """The channel must reuse the loss's own draw, not take a second one."""
    from eesi.systems.lj13.train import make_si_model, si_step
    calls = {"b": 0, "s": 0}
    model = make_si_model(hidden_nf=8, n_layers=2).to(DT)
    for name in ("b", "s"):
        net = getattr(model, f"net_{name}")
        fwd = net.forward
        net.forward = (lambda *a, _f=fwd, _n=name, **k:
                       (calls.__setitem__(_n, calls[_n] + 1), _f(*a, **k))[1])

    si_step(model, _lj13_data(8), entropy="both")
    assert calls == {"b": 2, "s": 2}, calls


# ---- XY (Phase C4) ---------------------------------------------------------


def test_xy_step_is_finite_and_backprops():
    torch.manual_seed(0)
    model = make_model(hidden=16, n_layers=2).to(DT)
    losses, _, _ = xy_step(model, _xy_data(16))
    total = losses["b"] + losses["s"]
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_xy_no_gradient_flows_through_the_coupling():
    from eesi.systems.xy.ot import xy_ot_couple
    x1 = _xy_data(8).requires_grad_(True)
    x0 = sample_base(8, 8, dtype=DT).requires_grad_(True)
    a, b = xy_ot_couple(x0, x1)
    assert not a.requires_grad


def test_xy_coupled_noise_is_still_a_valid_prior_sample():
    """The aligned noise must stay on the manifold: angles in (-pi, pi]."""
    from eesi.systems.xy.ot import xy_ot_couple
    a, _ = xy_ot_couple(sample_base(32, 8, dtype=DT,
                                    generator=torch.Generator().manual_seed(6)),
                        _xy_data(32))
    assert a.max() <= np.pi + 1e-9 and a.min() > -np.pi - 1e-9


def test_xy_all_ablation_arms_train():
    data = _xy_data(128)
    for align in (False, True):
        for batch in (False, True):
            model = make_model(hidden=16, n_layers=2)
            _, hist = train_xy(data, steps=60, batch=16, lr=3e-3, align=align,
                               batch_ot=batch, log_every=0, seed=0, model=model)
            h = np.asarray(hist).sum(1)
            assert np.isfinite(h).all()
            assert np.mean(h[-20:]) < np.mean(h[:20]), (align, batch)


def test_xy_averaging_populates_avg_without_leaking_into_the_state_dict():
    data = _xy_data(64)
    model = make_model(hidden=16, n_layers=2)
    model, _ = train_xy(data, steps=5, batch=16, lr=3e-3, log_every=0, seed=0,
                        model=model, averaging=AveragingConfig(kind="ema", decay=0.9))
    assert model.avg is not None and model.avg is not model
    assert not any(k.startswith("avg.") for k in model.state_dict())


def test_xy_entropy_flag_widens_the_history():
    """`entropy` adds columns to `hist` -- and only when asked.

    The default arity is load-bearing: the log line, `main`, the notebook's
    `zip(*total_hist)` and `test_xy_all_ablation_arms_train` above all unpack two.
    """
    data = _xy_data(64)
    widths = {None: 2, "dot": 3, "zdot": 3, "both": 4}
    for entropy, width in widths.items():
        model = make_model(hidden=16, n_layers=2)
        _, hist = train_xy(data, steps=5, batch=8, lr=3e-3, log_every=0, seed=0,
                           model=model, entropy=entropy)
        assert all(len(row) == width for row in hist), (entropy, len(hist[0]))
        assert np.isfinite(np.asarray(hist)).all(), entropy


def test_xy_entropy_channel_costs_no_extra_net_evaluations():
    """The channel must reuse the loss's own draw, not take a second one.

    Counting forward calls is the direct check: `entropy='both'` runs the two
    antithetic passes the loss needs and nothing more.
    """
    calls = {"b": 0, "s": 0}
    model = make_model(hidden=16, n_layers=2).to(DT)
    for name in ("b", "s"):
        net = getattr(model, f"net_{name}")
        fwd = net.forward
        net.forward = (lambda *a, _f=fwd, _n=name, **k:
                       (calls.__setitem__(_n, calls[_n] + 1), _f(*a, **k))[1])

    xy_step(model, _xy_data(8), entropy="both")
    assert calls == {"b": 2, "s": 2}, calls


def test_xy_reflect_flag_reaches_the_coupling():
    """--no-reflect must actually change the coupling, not silently no-op."""
    from eesi.systems.xy.ot import xy_ot_couple, xy_transport_cost
    x1 = _xy_data(32)
    x0 = sample_base(32, 8, dtype=DT, generator=torch.Generator().manual_seed(7))
    with_z2 = xy_transport_cost(*xy_ot_couple(x0, x1, batch=False, reflect=True))
    without = xy_transport_cost(*xy_ot_couple(x0, x1, batch=False, reflect=False))
    assert with_z2 <= without + 1e-9
    assert abs((with_z2 - without).item()) > 1e-6, "Z2 branch is a no-op"


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
