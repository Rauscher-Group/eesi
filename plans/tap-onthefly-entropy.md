# TAP: on-the-fly entropy tracking during training

**Status: implemented 2026-08-09.** Package code, tests and the notebook wiring are all in;
the only thing deliberately not done is the `eps=1e-3` bump in the notebook's
`make_si_model` call, which changes the sampled-`t` range and so is left for the user to
decide against a real run's S_zdot trace.

## Context

`eesi.systems.lj13.train` and `eesi.systems.xy.train` both log the interpolant entropy
estimators per batch while training (commit `e8dbee0` for XY, the same shape in
`lj13/train.py`). `eesi.systems.tap.train` does not: `tap_step` calls `model.loss(x1, x0)`
with no `entropy` keyword, and `train_si`'s history is a hard-coded 2-tuple.

Everything substantive already exists. `EESI.loss(x1, x0, entropy=...)` builds the
`"ent_dot"` / `"ent_zdot"` channels from the antithetic draw the loss already made — no
extra network evaluation, no autograd, detached — and `TAPEESI` inherits it unchanged.
`_check_entropy_channel` permits both channels here: `TAPEESI.__init__` forces
`learn_score = False`, but that only disables ISM, and with `gamma != "none"` the denoising
objective trains `net_s` regardless, so `"dot"` reads a genuinely trained score. This is the
identical situation to `LJ13EESI`.

So the work is **plumbing in `tap/train.py` plus tests**, mirroring `lj13/train.py`
line for line. Notebook wiring is a separate, optional section at the end.

Standing caveat, carried over verbatim from the LJ13 and XY docstrings: these numbers are a
progress diagnostic watched DURING a run, **not** a pass/fail criterion for a coupling or
architecture change — they run through the trained networks. The model-free arbiters stay
`transport_cost` for the coupling and generated-sample statistics (bond moments, tangent
correlation, E[Re²]) for the network.

---

## Decision to record

`_ENTROPY_KEYS` / `_HIST_LABELS` / `_hist_keys` will exist in **three** copies (xy, lj13,
tap) after this change. Plan is to **duplicate**, not to factor out:

- it keeps this diff identical in shape to the LJ13 one, which is what was asked for;
- the systems subpackages already mirror rather than share (`si_step` ↔ `tap_step`,
  `make_si_model` ×2), so a shared helper would be the first exception to that;
- promoting the three to something like `eesi/train_utils.py` is a clean, independent
  follow-up that can be done later without touching behaviour.

---

## Changes

### 1. `eesi/systems/tap/train.py` — the whole of the code work

**1a. Module docstring.** Add a paragraph after the "There is no plain flow-matching path
here" one, adapted from the LJ13 wording:

> `--entropy` / `train_si(entropy=...)` logs the b·s and b·z estimators per batch, reusing
> the interpolant draw `model.loss` already made (see `EESI.loss`), so it costs no extra
> network evaluation. It is there to watch dS converge DURING a run — not a pass/fail
> criterion for a coupling or architecture change, since it runs through the trained
> networks. The model-free arbiters stay `transport_cost` for the coupling and
> generated-sample statistics for the network. TAP has no free-energy cross-check at all
> (no target potential exists), which makes the entropy channel the only quantity of
> interest that is visible mid-run.

**1b. `tap_step`** — one new keyword, threaded straight through:

```python
def tap_step(model, x1, k, b, gamma, cos_theta_0, align=True, batch=True,
             generator=None, entropy: str | None = None):
    ...
    return model.loss(x1, x0, entropy=entropy), x0, x1
```

Docstring note (mirrors `si_step`): *`entropy` ("dot", "zdot", "both") is passed straight to
`model.loss`, which adds the matching detached "ent_dot"/"ent_zdot" keys to the returned
dict.* Keep the existing warning about the two different `gamma`s — it now has a third
neighbour in `entropy`, and the paragraph is the only thing keeping them straight.

**1c. History-key tables**, copied from `lj13/train.py` (module level, above `train_si`):

```python
#: History columns contributed by each `entropy` setting, and how they are labelled
#: in the log line. "b"/"s" always come first, so the default history stays the
#: 2-tuple (loss_b, loss_s) that the notebooks and tests unpack. Mirrors
#: eesi.systems.lj13.train / eesi.systems.xy.train.
_ENTROPY_KEYS = {None: (), "dot": ("ent_dot",), "zdot": ("ent_zdot",),
                 "both": ("ent_dot", "ent_zdot")}
_HIST_LABELS = {"b": "loss_b", "s": "loss_s", "ent_dot": "S_dot", "ent_zdot": "S_zdot"}


def _hist_keys(entropy: str | None) -> tuple[str, ...]:
    """The ordered `model.loss` keys recorded per step, given the `entropy` setting."""
    if entropy not in _ENTROPY_KEYS:
        raise ValueError(f"entropy must be one of {sorted(map(str, _ENTROPY_KEYS))}, got {entropy!r}")
    return ("b", "s") + _ENTROPY_KEYS[entropy]
```

**1d. `train_si`** — new `entropy: str | None = None` parameter (last, after `model=`), and
three edited lines in the loop:

```python
    keys = _hist_keys(entropy)          # before the loop; validates early
    ...
        losses, x0, x1 = tap_step(model, data[idx], k, b, gamma, cos_theta_0,
                                  align=align, batch=batch_ot, entropy=entropy)
    ...
        hist.append(tuple(losses[key].item() for key in keys))
    ...
        if log_every and (step % log_every == 0 or step == steps - 1):
            means = np.mean(hist[-log_every:], axis=0)
            cols = "  ".join(f"{_HIST_LABELS[key]} {m:9.4f}" for key, m in zip(keys, means))
            print(f"  step {step:5d}  {cols}  "
                  f"transport {transport_cost(x0, x1).item():7.3f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
```

Note the local-name hazard this file already documents: `b` is the prior's equilibrium bond
length in this scope, so the comprehension variable must be `key`, not the `k` used in the
LJ13 copy (where `k` is free) — `k` is the bond stiffness here. Same reason `x0, x1` are not
`a, b`.

Docstring addition, adapted from LJ13 (keep the existing `k`/`b`/`gamma`/`cos_theta_0`
paragraph above it):

> `history` is a list of per-step tuples, `(loss_b, loss_s)` by default. `entropy` ("dot",
> "zdot" or "both") appends the matching per-batch entropy estimates as extra columns —
> `(loss_b, loss_s, S_dot, S_zdot)` for "both" — and prints them in the log line. They are
> computed inside `model.loss` from the draw it already made, so they add no network
> evaluations; see `EESI.loss`.
>
> The estimates inherit the model's `eps`, which floors the 1/gamma in "zdot". If that
> channel looks noisy, build the model with a looser floor — `make_si_model(..., eps=1e-3)`
> — as `EESI.entropy_estimate` documents.

**1e. `main()`** — the `--entropy` flag, verbatim from LJ13's help text minus its `--si`
sentence (TAP has no flow-matching path, so there is no `--entropy needs --si` guard to
port):

```python
    p.add_argument("--entropy", choices=("dot", "zdot", "both"), default=None,
                   help="log the per-batch entropy estimators alongside the losses: "
                        "'dot' is -b.s with the learned score, 'zdot' is -b.z with the "
                        "exact conditional score. Free (reuses the loss's own draw). A "
                        "progress diagnostic, not a validation metric. 'zdot' inherits "
                        "the model's eps, which floors its 1/gamma; if it looks noisy, "
                        "build the model with eps~1e-3.")
```

pass `entropy=a.entropy` to `train_si`, and replace the fixed two-column final line with:

```python
    means = np.mean(hist[-100:], axis=0)
    cols = "  ".join(f"{_HIST_LABELS[key]} {m:.4f}"
                     for key, m in zip(_hist_keys(a.entropy), means))
    print(f"final (last 100): {cols}")
```

Nothing else in the module changes. `make_si_model` already forwards `**kw` to `TAPEESI`,
so `eps=1e-3` is reachable without an edit.

### 2. `eesi/systems/tap/interpolant.py`, `dynamics.py`, `data.py`, `ot.py` — no changes

`TAPEESI` needs nothing: `loss`, `_check_entropy_channel` and both accumulators are
inherited, and the only override that matters (`_noise_like`) already anchors the latent `z`
the "zdot" channel reads. Worth one sentence in the `TAPEESI` class docstring's
"Divergence/entropy" paragraph, pointing at the new training-time channel as the cheap
counterpart of `entropy_estimate`.

---

## Tests (`tests/tap/test_tap_training.py`)

Three new tests, mirroring `tests/core/test_training.py`'s LJ13 trio. Note this file keeps
an explicit list in its `__main__` block — **each new test must be appended there too**, or
the script mode silently skips it.

1. **`test_entropy_flag_widens_the_history`** — the LJ13/XY test, adapted:

```python
@pytest.mark.parametrize("entropy,width",
                         [(None, 2), ("dot", 3), ("zdot", 3), ("both", 4)])
def test_entropy_flag_widens_the_history(entropy, width):
    model = _model(seed=2)
    _, hist = train_si(_data(B=32, seed=3), K, B_LEN, GAMMA, COS0, steps=5, batch=4,
                       log_every=0, model=model, entropy=entropy)
    assert all(len(row) == width for row in hist)
    assert np.isfinite(np.asarray(hist)).all()
```

The default arity of 2 is load-bearing — the log line, `main`, the notebook's
`zip(*htot)` and `test_train_si_runs_and_reduces_nothing_catastrophically` all unpack two.
(This file does not currently import numpy; add it.)

2. **`test_entropy_channel_costs_no_extra_net_evaluations`** — the direct check that the
channel reuses the loss's own draw. Wrap both nets' `forward` with a counter and assert
`calls == {"b": 2, "s": 2}` after `tap_step(..., entropy="both")` — two antithetic branches
and nothing more.

3. **`test_score_is_trained_despite_learn_score_off`** — why `"dot"` is legal for `TAPEESI`
at all. Assert `model.learn_score is False and model.gamma != "none"`, then that
`(losses["b"] + losses["s"]).backward()` puts non-zero gradient on `net_s` parameters. TAP
has no equivalent of this test today, and it is the invariant that would break if someone
ever "fixed" `TAPEESI.__init__`.

4. **Rejection path** (one extra assert, cheap): `_hist_keys("bogus")` raises `ValueError`,
and `train_si(..., entropy="zdot")` on a `gamma="none"` model raises from
`_check_entropy_channel`. The second one is the interesting half — it is the only place a
TAP user can hit that error, since `make_si_model`'s default is `gamma="quad"`.

`tests/core/test_training.py` is untouched (it covers lj13 and xy only).

---

## Notebook wiring — `experiments/TAP/TAP.ipynb` (optional, do after the tests pass)

Small and self-contained, matching what `experiments/LJ/LJ13.ipynb` cells 11–12 do:

- **Model cell (12)**: the model is built `gamma="sqrt", gamma_scale=0.2, learn_score=False`
  with the default `eps=1e-6`. That is the floor `EESI.entropy_estimate` warns about for
  "zdot". Add `eps=1e-3` to the `make_si_model` call if the S_zdot trace looks noisy — it
  affects only the sampled-`t` range and the gamma floor, so it is safe to leave in.
- **Training cell (14)**: `entropy="both"` on the `train_si` call, then
  `hist_b, hist_s, ent_dot, ent_zdot = map(np.asarray, zip(*htot))`. **`htot` accumulates
  across repeated runs of that cell** — mixing a 2-column run with a 4-column one makes the
  unpack fail. Reset `htot = []` (cell 12) when switching the flag on.
- **Plot cell (15)**: third panel, `smooth(ent_dot, ...)` and `smooth(ent_zdot, ...)` on the
  same axes; their gap is how far `net_s` is from the true score. No horizontal reference
  line here, unlike LJ13's `-34.17` — TAP has no ΔF cross-check. The available comparison is
  §5's batch `entropy_estimate` values (`means_dot` / `means_zdot` / `means_div`, already
  saved to `means_*.npy`), which the running trace should converge toward; the "div" channel
  remains estimator-only and cannot appear in the training trace.
- Optionally `np.save("tap_train_hist.npy", np.array(htot))` alongside the existing
  `tap_train_{b,s}.npy`, matching the LJ13 notebook's single-array save.

---

## Verification

1. `pytest tests/tap -q` — new tests pass, existing ones untouched.
2. `pytest tests/core/test_training.py -q` — the shared `EESI.loss` path is unchanged.
3. `python -m eesi.systems.tap.train --k 100 --b 1.0 --gamma 2.33 --cos-theta-0 0.5
   --n-data 2000 --steps 40 --batch 16 --entropy both` — log line shows four columns; the
   same command without `--entropy` still shows two.
4. Sanity on the numbers, not just the plumbing: over a few hundred steps of a real run,
   `S_zdot` should track `S_dot` closely once the score has trained, and both should land in
   the neighbourhood of §5's `entropy_estimate(method="dot"/"zdot")` batch means. A large,
   persistent `S_dot` − `S_zdot` gap means `net_s` is behind, not that the channel is wrong.
