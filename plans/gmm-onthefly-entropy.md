# GMM: on-the-fly entropy tracking during training

**Status: planned, not started.**

## Context

`eesi.systems.xy.train`, `eesi.systems.lj13.train` and `eesi.systems.tap.train` all log the
interpolant entropy estimators (`ent_dot` = -b.s with the learned score, `ent_zdot` = -b.z/gamma
with the exact conditional score) per batch while training, via `EESI.loss(x1, x0,
entropy=...)`. `eesi.systems.gmm.train` does not: `gmm_step` calls `model.loss(x1, x0)` with no
`entropy` keyword, and `train`'s history is a hard-coded 2-tuple `(loss_b, loss_s)`.

Everything substantive already exists in `EESI.loss` / `_check_entropy_channel`
(`eesi/interpolant.py`) — GMM uses the base `EESI` directly (no subclass), so nothing there
needs to change. The channel is free: it reuses the antithetic `+z`/`-z` draw the loss already
made, under `no_grad`, so it costs no extra network evaluation.

GMM is actually the **best-instrumented** of the four systems for this: `experiments/GMM/GMM.ipynb`
already computes an *exact* reference value, `delta_pred = -E[log p_target(x1)] + E[log
p_base(x0)]`, from the closed-form mixture density (§4, cell `a8ccf09d`) — not a TI estimate like
LJ13's -34.17, and not absent like TAP's. The on-the-fly trace has a real target to converge
toward from step 1 of the work, not just after post-hoc `entropy_estimate` calls.

So the work is **plumbing in `gmm/train.py` only** — no new tests, no notebook changes (both
explicitly out of scope for this pass).

Standing caveat, carried over verbatim from the other three docstrings: these numbers are a
progress diagnostic watched DURING a run, **not** a pass/fail criterion for a coupling or
architecture change — they run through the trained networks. The model-free arbiters stay
`gmm_transport_cost` for the coupling and generated-sample log-prob statistics (`delta_pred` vs.
`target.log_prob(gen_ode/sde)`) for the network. [[no-entropy-as-validation]]

---

## The one GMM-specific wrinkle

`gmm.train.train` is the only one of the four systems whose loop gates gradient with its own
local booleans instead of unconditionally summing `losses["b"] + losses["s"]`:

```python
loss = losses["b"].new_zeros(())
if learn_vel:
    loss = loss + losses["b"]
if learn_score:
    loss = loss + losses["s"]
```

`model.learn_score` (the `EESI` attribute `_check_entropy_channel` inspects) stays `True` by
default here and is irrelevant to this gate — it only controls whether `EESI.loss` takes the ISM
path when `gamma == "none"`, and `make_model`'s default is `gamma="sqrt"`, so that path is never
taken. This means **`_check_entropy_channel` will happily allow `entropy="dot"` even when the
caller also passed `learn_score=False` to `train()`** — `net_s` never receives gradient in that
run, so `"dot"` (which reads `net_s`'s live output) reports a meaningless number that looks like
a normal float.

Decision: add an explicit guard in `train()`, not just a docstring warning — this is silent and
easy to hit (e.g. `--no-score --entropy dot` for a "watch the drift-only run's b.z" experiment
that doesn't realize "dot" needs the score net too):

```python
if entropy in ("dot", "both") and not learn_score:
    raise ValueError("entropy='dot'/'both' reads net_s's output, which needs learn_score=True")
```

`"zdot"` alone stays legal with `learn_score=False`, same as `"zdot"` staying legal with
`learn_vel=False` in principle (though `entropy="zdot"` with `learn_vel=False` has the mirror
problem — `net_b` untrained — deliberately left unguarded, since `entropy="zdot"` conventionally
implies interest in `net_b`'s quality and no one has hit this arm in practice; note it in the
docstring instead of guarding it, to keep the check to the one hazard that's actually easy to hit).

---

## Changes

### 1. `eesi/systems/gmm/train.py` — the whole of the code work

**1a. Module docstring.** Add a paragraph after the existing usage-examples block:

> `--entropy` / `train(entropy=...)` logs the b.s and b.z/gamma estimators per batch, reusing the
> interpolant draw `model.loss` already made (see `EESI.loss`), so it costs no extra network
> evaluation. It is there to watch dS converge DURING a run — not a pass/fail criterion for a
> coupling or architecture change, since it runs through the trained networks. The model-free
> arbiters stay `gmm_transport_cost` for the coupling and generated-sample log-prob statistics for
> the network. Unlike LJ13 (TI-estimated) or TAP (no cross-check at all), GMM has an *exact*
> entropy-change reference — `delta_pred` in `experiments/GMM/GMM.ipynb` §4, from the closed-form
> mixture density — which the on-the-fly trace should converge toward.

**1b. `gmm_step`** — one new keyword, threaded straight through:

```python
def gmm_step(model: EESI, x1: torch.Tensor, batch_ot: bool = True, generator=None,
            entropy: str | None = None):
    ...
    return model.loss(x1, x0, entropy=entropy), x0, x1
```

Docstring note (mirrors `si_step`/`tap_step`): *`entropy` ("dot", "zdot", "both") is passed
straight to `model.loss`, which adds the matching detached "ent_dot"/"ent_zdot" keys to the
returned dict.*

**1c. History-key tables**, the fourth copy of the pattern (module level, above `train`):

```python
#: History columns contributed by each `entropy` setting, and how they are labelled
#: in the log line. "b"/"s" always come first, so the default history stays the
#: 2-tuple (loss_b, loss_s) that the notebook and tests unpack. Mirrors
#: eesi.systems.{xy,lj13,tap}.train.
_ENTROPY_KEYS = {None: (), "dot": ("ent_dot",), "zdot": ("ent_zdot",),
                 "both": ("ent_dot", "ent_zdot")}
_HIST_LABELS = {"b": "loss_b", "s": "loss_s", "ent_dot": "S_dot", "ent_zdot": "S_zdot"}


def _hist_keys(entropy: str | None) -> tuple[str, ...]:
    """The ordered `model.loss` keys recorded per step, given the `entropy` setting."""
    if entropy not in _ENTROPY_KEYS:
        raise ValueError(f"entropy must be one of {sorted(map(str, _ENTROPY_KEYS))}, got {entropy!r}")
    return ("b", "s") + _ENTROPY_KEYS[entropy]
```

**1d. `train`** — new `entropy: str | None = None` parameter (after `learn_score=True`, before
`device=`), the guard from the wrinkle above, and edits to the loop:

```python
    if not (learn_vel or learn_score):
        raise ValueError("learn_vel and learn_score cannot both be False: no loss to train")
    if entropy in ("dot", "both") and not learn_score:
        raise ValueError("entropy='dot'/'both' reads net_s's output, which needs learn_score=True")
    ...
    keys = _hist_keys(entropy)          # before the loop; validates early
    ...
    for step in range(steps):
        losses, a, b = gmm_step(model, draw_x1(), batch_ot=batch_ot, entropy=entropy)
        ...
        hist.append(tuple(losses[key].item() for key in keys))

        if log_every and (step % log_every == 0 or step == steps - 1):
            means = np.mean(hist[-log_every:], axis=0)
            cols = "  ".join(f"{_HIST_LABELS[key]} {m:9.4f}" for key, m in zip(keys, means))
            print(f"  step {step:5d}  {cols}  "
                  f"transport {gmm_transport_cost(a, b).item():7.3f}  "
                  f"({time.perf_counter()-t0:5.1f}s)")
```

Docstring addition (extend the existing `history` sentence):

> `history` is a list of per-step tuples, `(loss_b, loss_s)` by default. `entropy` ("dot", "zdot"
> or "both") appends the matching per-batch entropy estimates as extra columns — `(loss_b, loss_s,
> S_dot, S_zdot)` for "both" — and prints them in the log line. They are computed inside
> `model.loss` from the draw it already made, so they add no network evaluations; see `EESI.loss`.
> `entropy="dot"`/`"both"` requires `learn_score=True` (raises otherwise) since it reads `net_s`'s
> live output.

**1e. `main()`** — the `--entropy` flag, adapted from LJ13/TAP's help text:

```python
    p.add_argument("--entropy", choices=("dot", "zdot", "both"), default=None,
                   help="log the per-batch entropy estimators alongside the losses: "
                        "'dot' is -b.s with the learned score (needs --no-score absent), "
                        "'zdot' is -b.z/gamma with the exact conditional score. Free "
                        "(reuses the loss's own draw). A progress diagnostic, not a "
                        "validation metric.")
```

pass `entropy=a.entropy` to `train`, and replace the fixed two-column final line with:

```python
    means = np.mean(hist[-100:], axis=0)
    cols = "  ".join(f"{_HIST_LABELS[key]} {m:.4f}"
                     for key, m in zip(_hist_keys(a.entropy), means))
    print(f"final (last 100): {cols}")
```

`p.error` guard mirroring LJ13's `--entropy needs --si`: here it's `--entropy {dot,both} needs
--no-score absent` — i.e. reject `a.entropy in ("dot", "both") and a.no_score` at parse time
(cheaper feedback than waiting for `train`'s `ValueError`).

### 2. `eesi/interpolant.py`, `eesi/systems/gmm/{ot,data,mlp}.py` — no changes

`EESI.loss`, `_check_entropy_channel` and both accumulators are used as-is. `make_model`'s
`gamma="sqrt"` default already satisfies `_check_entropy_channel`'s `zdot`/`both` requirement
(`gamma != "none"`).

---

## Out of scope for this pass

- **No new tests.** `tests/gmm/` currently has only `test_gmm_ot.py`; this change adds no
  `tests/gmm/test_train.py`. The default history arity of 2 (unaffected when `entropy=None`) is
  what keeps this a safe no-test change for existing callers.
- **No notebook changes.** `experiments/GMM/GMM.ipynb`'s training/plot cells keep unpacking
  `hist_b, hist_s = np.asarray(htot).T` and are not touched. `entropy=` stays available to call
  from the notebook manually if wanted, but no cell is edited to use it.

## Verification

1. `python -m eesi.systems.gmm.train --d 8 --n-mixes 4 --steps 40 --batch 32 --entropy both` —
   log line shows four columns; the same command without `--entropy` still shows two;
   `--no-score --entropy dot` errors at parse time.
2. `python -m eesi.systems.gmm.train --d 8 --n-mixes 4 --steps 40 --batch 32` (no `--entropy`) —
   confirms the default path is byte-for-byte unchanged in shape (2-column history, same log line).
3. Sanity on the numbers, not just the plumbing: over a few hundred steps of a real run, `S_dot`
   and `S_zdot` should both trend toward `delta_pred` (§4 of the notebook, computed manually if
   cross-checking) as training progresses, with `S_zdot` converging faster (it uses the exact
   conditional score, not the learned one). A large, persistent `S_dot` − `S_zdot` gap means
   `net_s` is behind, not that the channel is wrong. [[no-entropy-as-validation]]
