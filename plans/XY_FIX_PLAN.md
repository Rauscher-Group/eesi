# XY chain fix plan

Implementation plan for the changes in `plans/XY_MODEL_REVIEW.md`.

**Hard constraint: no changes to any loss function.** Nothing in this plan touches
`EESI.loss`, `_denoising_loss`, `score_loss`, `_div_exact`, `_div_hutchinson`, or
`xyEESI._interpolant_sample`. Every change is confined to:

```
eesi/models/xygnn.py      the network
eesi/ot.py                the coupling group
eesi/train/xy.py          construction + CLI flags
tests/test_xygnn.py       equivariance tests
tests/test_ot.py          coupling tests
tests/ot_reference.py     the CPU oracle
experiments/XY_chain_eqOT.ipynb   config only
```

Five phases, ordered so each is independently testable and independently revertible.
Phases 0–4 are small. Phase 5 is the only one that changes the layer's shape.

> **Revised 2026-07-24 after the diagnostics** (results in `XY_MODEL_REVIEW.md` §0.5):
> Phase 6 (score preconditioning) is **deleted** — F2 was refuted; the true score is
> bounded by ~5 everywhere, so there is nothing to precondition. `eps=1e-3` is dropped
> from Phase 0 for the same reason. Phase 1 is promoted to the top on strong evidence.
> `n_layers` should drop 4 → 2 (Phase 0.4), and `mcxy` should be replaced by the exact
> sampler (Phase 0.5).

---

## Status (2026-07-24)

| phase | state |
|---|---|
| 0 — free wins | **done** — graph cache, exact sampler, docstrings |
| 1 — non-periodic time embedding | **done** — `time_features` + learned `time_mlp`, `time_order=0` default |
| 2 — O(2) equivariance in the network | **done** — `cos_expand` edge attrs, oddness test passes exactly |
| 3 — O(2) × Z₂ in the coupling | **done** — `negate` threaded through, 16-cell oracle agreement |
| 4 — zero-sum projection on `net_s` | **skipped by decision** — see below |
| 5 — node features `h` | **on hold** — priorities under review |

All of the above is uncommitted on `main`. Full suite green at 134 tests.

The notebook `experiments/XY_chain_eqOT.ipynb` is still deliberately untouched, so
Phase 0.4 (`n_layers` 4 → 2) and the switch to the exact sampler are pending there.

### Validation rule (supersedes the "what to measure" column below)

**`ΔU` and `ΔS` are not validation metrics and must not be used to score a phase.**
They are computed *through the trained networks*, so they confound model quality,
training noise, and the change under test — and they are the scientific result this
project exists to produce, which makes scoring code changes against them circular.
`eesi/train/xy.py`'s own module docstring already said this.

Validate with model-free quantities instead: `xy_transport_cost` for coupling
changes; exact symmetry/equivariance assertions and oracle agreement against
`tests/ot_reference.py` for network and group changes; wall-clock for performance.
Report `ΔU`/`ΔS` only as a *result*, in a notebook.

This invalidates the "measure after" column of the table at the bottom of this file
for phases 1, 2, 4 and 5. Phase 5 in particular now has **no agreed acceptance
criterion** — that has to be settled before it starts.

---

## Phase 0 — free wins (no design decisions)

**0.1 Cache the chain graph.** `chain_edge_index` currently runs a Python triple loop
and builds CPU tensors on *every* forward (4× per training step, more during Heun
sampling). Wrap it:

```python
@lru_cache(maxsize=None)
def _chain_edge_index_cached(N, n_neighbors, device_str): ...
def chain_edge_index(N, n_neighbors, device=None):   # keep the public signature
    return _chain_edge_index_cached(N, n_neighbors, str(device))
```

Cache `_batch_graph`'s output too, keyed `(B, N, n_neighbors, device)` — `B` is
constant during training. The cached tensors are read-only downstream (`inv_dist` is
only ever `cat`-ed, `edge_index` only indexed), so sharing them is safe. Add a test
asserting a second call returns the *same* tensor object.

**0.2 ~~`eps = 1e-3`~~.** Dropped. The diagnostics show the true drift and score are
bounded at both endpoints (`‖∇log p_t‖ → 0` at `t→0`, `→ 4.90` at `t→1`), so there is
no endpoint region to avoid. Leave `eps` at its default.

**0.4 `n_layers`: 4 → 2.** Measured: layers 3–4 change `ΔS` error by 6.6 % → 6.7 % and
make `loss_s` marginally worse, for 2× parameters and 2× wall-clock. Config-only change
in the notebook. Re-check after Phase 5, which changes what a layer costs.

**0.5 Exact data sampler.** Replace `load_mc_data`'s 10⁶-step Python Metropolis loop
with the exact `O(BN)` sampler implied by §1 of the review:

```python
def sample_p1_exact(B, N, J, rng):
    """theta_1 ~ U(-pi,pi], Delta_i ~ vonMises(0, J) i.i.d., theta = cumsum."""
    d  = scipy.stats.vonmises.rvs(kappa=J, size=(B, N-1), random_state=rng)
    th = np.concatenate([u := rng.uniform(-np.pi, np.pi, (B,1)),
                         u + np.cumsum(d, axis=1)], axis=1)
    return (th + np.pi) % (2*np.pi) - np.pi
```

Exact, uncorrelated, instant, and `N`-agnostic — which also unblocks the mixed-`N`
training suggested in Phase 5.3. Keep `mcxy` for validation (a test that the two agree
in distribution is a good use of it), but stop training on it.

**0.3 Docstring corrections.** Delete the "invariant to shifting `t` by an integer"
claim (`xygnn.py:36-37`) — Phase 1 removes the property, and it was never desirable.
Delete the `tanh` / `coords_range` argument docs (`xygnn.py:198-200`) or implement them
(Phase 5.2); right now they document parameters `__init__` does not accept.

---

## Phase 1 — F1: learned, non-periodic time embedding

The bug: `fourier_expand(2π·t, K)` is exactly 1-periodic, so `t=0` and `t=1` map to the
identical vector.

**1.1** New helper beside `fourier_expand` (keep `fourier_expand`; Phase 2 still needs
an angle expansion):

```python
def time_features(t, order, w_max=30.0):
    """Non-periodic features of t in [0,1] -> [..., 1 + 2*order].

    Log-spaced frequencies, deliberately NOT harmonics of 2*pi, so no two distinct
    t in [0,1] share an embedding -- in particular t=0 and t=1 do not.
    """
    w = torch.logspace(0.0, math.log10(w_max), order, device=t.device, dtype=t.dtype)
    a = t.unsqueeze(-1) * w
    return torch.cat([t.unsqueeze(-1), a.cos(), a.sin()], dim=-1)
```

**1.2** Learned projection on the backbone (computed once per forward, as now):

```python
self.time_mlp = nn.Sequential(
    nn.Linear(1 + 2 * time_order, time_dim), act_fn, nn.Linear(time_dim, time_dim))
```

New ctor arg `time_dim: int = 32`. `time_order` keeps its name but now means "number of
log-spaced frequency pairs"; `time_order=8` is plenty.

**1.3** In `forward`: `g_t = self.time_mlp(time_features(t_b, self.time_order))`, then
`time_edges = g_t[edge_batch]` unchanged. `in_dim = edge_attr_dim + time_dim`.

**Tests.** `time_features(0) != time_features(1)` by a wide margin; and end-to-end,
`net(zeros, x)` differs from `net(ones, x)` for a randomly initialised net.

**Note.** This is the one phase where I'd also try the trivial ablation — replace
`time_features` with raw `t` alone into the same MLP. If it matches, keep the simpler
version.

---

## Phase 2 — F4a: O(2) equivariance in the network

`trans = Δθ · φ` is odd **iff `φ` is even**. `sin(kΔθ)` in `edge_attr` is what breaks it.

**2.1** Add the even-only expansion:

```python
def cos_expand(x, order):
    """[cos(x), ..., cos(order*x)] -- EVEN under x -> -x. [...] -> [..., order]"""
```

**2.2** In `forward`, `edge_attr = cat([inv_dist, cos_expand(dθ, edge_order)])`, so
`edge_attr_dim = 1 + edge_order` (was `1 + 2*edge_order`).

That is the whole change. Oddness then holds inductively: `Δθ` odd × `φ` even ⟹ `trans`
odd ⟹ `θ^(ℓ)` odd at every layer ⟹ output `θ^(L) − θ^(0)` odd. The time embedding is
invariant under negation, so it needs nothing.

**Parity discipline to maintain from here on (matters for Phase 5):** every scalar
channel in the network is **even**; the *only* odd quantity is `Δθ`, and it appears
exactly once, as the direction carried by `trans`. This is EGNN's invariant-scalar ×
equivariant-direction structure.

**Tests.** In `tests/test_xygnn.py`, next to the existing rotation/reversal tests:

```python
def test_spin_reflection_equivariance():
    net, x, t = _make_model(2), _random_angles(4, 8), torch.rand(4)
    assert torch.allclose(net(t, -x), -net(t, x), atol=1e-10)
```

This should **fail on `main`** and pass after 2.2 — worth confirming in that order.

---

## Phase 3 — F4b: O(2) × Z₂ in the coupling

Group goes from 2 to 4 discrete elements. Single point of definition, as designed.

**3.1** `eesi/ot.py`:

```python
def _xy_group_elements(x, reflect=True, negate=True):
    """Discrete orbit under Z2^site x Z2^spin. Yields (B, N) tensors."""
    g = [x]
    if reflect:            g.append(x.flip(-1))
    if negate:             g.append(-x)
    if reflect and negate: g.append(-x.flip(-1))
    return tuple(g)
```

**3.2** Thread `negate: bool = True` through `xy_cost_matrix` and `xy_ot_couple`. Both
call `_xy_group_elements` with the same flags, so the `argmin` index and the survivor
`pick` stay consistent automatically — no other logic changes. `_xy_apply_alignment`
(the circular mean) is untouched: the closed form applies to any group element.

Cost: 2 extra `(B,N)@(N,B)` matmul pairs, ~2.6 MFLOP at `B=256, N=10`.

**3.3** `tests/ot_reference.py::xy_match_pair` — the oracle loops `for r in ((0,1) if
reflect else (0,))`. Make it a nested loop over `(reflect, negate)` so the
oracle-comparison tests stay meaningful.

**3.4** `eesi/train/xy.py` — `negate=True` on `xy_step`/`train`, `--no-negate` CLI flag
alongside `--no-reflect`.

**3.5** Docstrings: the module table in `ot.py` says `XY ... Z2 x U(1)`; it is now
`O(2) x Z2`. Same for the group description block above `_xy_group_elements`.

**Tests.**
- `test_xy_negation_branch_is_live` — mirror `test_xy_reflection_branch_is_live`.
- Extend `test_xy_all_ablation_cells_match_oracle` with the `negate` axis (2→4 cells
  per existing cell).
- Extend `test_xy_marginal_preserved_over_z2_u1` to the 4-element group (and rename).
  The proof only needs `p₀`, `p₁` to be `G`-invariant, which holds.

**Expected effect:** a few percent off the transport cost, no more — your own numbers
have `batch` (−32 %) dominating `align` (−20 %) at `B=256`. The payoff is consistency
with Phase 2, not the coupling metric.

**Measured (done).** Chordal transport cost, mean over 20 batches, `J=2`:

| group | N=10, B=256 | N=32, B=256 | N=10, B=32 |
|---|---|---|---|
| SO(2) | 3.798 | 19.485 | 4.741 |
| SO(2) × Z₂^site | 3.582 | 18.873 | 4.378 |
| O(2) (spin flip, no reversal) | 3.577 | 18.866 | 4.390 |
| **O(2) × Z₂^site (full)** | **3.392** | **18.290** | **4.102** |
| spin flip buys | −5.3 % | −3.1 % | −6.3 % |

As predicted: a few percent. Worth noting the two Z₂s contribute almost identically
on their own and their gains are near-additive, so they are *not* redundant despite
both acting as `Δθ → −Δθ` on bond differences — the reversal also reorders sites.

---

## ~~Phase 4 — zero-sum projection on `net_s` only~~ (skipped 2026-07-24)

**Skipped by decision, not refuted.** The physics below is correct and was checked
carefully; the change is simply not worth making now. Recorded in full so the
reasoning isn't re-derived later.

*Why it is sound.* Rigid rotation `θ_j → θ_j + φ` costs zero energy, so it is the
Goldstone mode of the chain's U(1) symmetry. Every `p_t` is exactly invariant under
it — `p_1` because the energy sees only bonds, `p_0` because it is uniform, and the
intermediates because the coupling (cost sees only relative angles), the geodesic
(displacements are unchanged by a common shift) and the isotropic noise all commute
with the rotation. Differentiating `log p_t(θ + φ1) = log p_t(θ)` at `φ = 0` gives
`Σⱼ ∂ⱼ log p_t = 0` — pointwise, every `t`, every configuration. The score is the
thermodynamic force, and there is no restoring force against a free rotation.
Independently confirmed at both ends: the score vanishes identically at `t→0`, and
at `t=1` the site force `J[sin(D_j) − sin(D_{j−1})]` is the lattice divergence of a
bond current, which telescopes to zero (Kirchhoff, with no flux through the open
ends — and it would also close around a ring, so this is the symmetry talking, not
the boundary condition).

*Why the drift is genuinely different.* Invariance of a vector *field* says nothing
about its component along `1`. Only gradients of invariant *scalars* are forced
orthogonal to the symmetry direction. The drift must carry net rotation: `Σⱼ dⱼ` is
the total twist needed to carry a prior sample onto its partner, and the alignment
forces `Σⱼ sin(dⱼ) = 0`, not `Σⱼ dⱼ = 0`. Flatness of the density along the zero mode
and absence of *transport* along it are different things.

*Why it was skipped anyway.* The gain is `‖Qv‖²` — whatever spurious mean the trained
net happens to carry — not the `1/N` of target variance that drives it. And the
antithetic ±z pairing already suppresses that driving noise: the two branches enter
the rotational component with opposite signs, and they fail to cancel only because
the two evaluations sit at slightly different configurations, making the residual
second-order. Small, and shrinking as `1/N`. If it returns, it returns as cleanup,
not as a win.

*The condition that would invalidate it.* The argument needs the coupling to stay
rotation-equivariant. A coupling that fixed a gauge, pinned a spin, or conditioned on
the total angle would leave `p_t` non-flat along the rotation mode, and the projection
would become a bias.

<details><summary>Original Phase 4 text</summary>

### Phase 4 — zero-sum projection on `net_s` only

`p_t` is U(1)-invariant, so `log p_t(x + φ·1) = log p_t(x)` for all `φ`, hence
`Σⱼ ∂ⱼ log p_t = 0` **exactly**. The score's rotational component is known to be zero,
so projecting it out is free accuracy — the `S¹` analogue of `LJ13Dynamics`'s
`vel - vel.mean(1)`.

This does **not** hold for the drift: `Σⱼ b_target,j = β̇·Σd + γ̇·Σz ≠ 0`. (This
corrects §7.1 of the review, which overstated it.)

**4.1** `XYChainGNN(..., zero_sum: bool = False)`; in `forward`, before the reshape:
`if self.zero_sum: v = v - v.mean(-1, keepdim=True)`.

**4.2** `make_model` builds the two nets with different kwargs —
`XYChainGNN(**net_kw, zero_sum=False)` for `net_b`, `zero_sum=True` for `net_s`.

Mean subtraction commutes with negation and with site permutation, so Phase 2 and the
reversal equivariance survive.

**Test.** `net_s(t, x).sum(-1)` is zero to `1e-12`; `net_b(t, x).sum(-1)` is not.

</details>

---

## Phase 5 — node features `h` (the structural change) — **ON HOLD**

> On hold as of 2026-07-24: the user is reading the code to reconsider priorities.
> Do not assume the phase order above still holds — confirm before starting.
>
> **Blocking issue:** the "measure after" line for this phase was `ΔU`, `ΔS` and the
> N-sweep, all of which the validation rule above rules out. The symmetry and oracle
> tests certify *correctness* but say nothing about whether the architecture is
> better, and 5.3's N-generalisation claim has no model-free proxy at all. An
> acceptance criterion has to be agreed before this starts.

Restores the second state channel, so information propagates without moving spins.
Do this last, and measure against Phase 0–3 as the baseline.

**5.1** Backbone: `self.h_init = nn.Linear(time_dim, hidden)`, giving `h [N_tot, hidden]`
identical across nodes at layer 0 (exactly what EGNN does with `h = t·ones` →
`embedding`). `XYChainConv` becomes EGNN-shaped:

```python
edge_feat = edge_mlp([h[src], h[dst], edge_attr, time_edges])   # [E, hidden]
trans     = d_theta * coord_mlp(edge_feat)                      # [E, 1]
theta     = theta + scatter_add(trans, dst, N_tot)
h         = h + node_mlp([h, scatter_add(edge_feat, dst, N_tot)])
```

**Parity, per the rule in Phase 2:** `h` is built from `edge_attr` (even) and the time
embedding (invariant), so `h` is even; `trans` stays odd; the Phase 2 test must still
pass. That test is the guard for this phase — run it first, not last.

**5.2** While here, add the `tanh` + `coords_range` step bound the docstring already
advertises: `coord_mlp` ends in `Tanh()`, and `trans` is scaled by
`coords_range / n_layers`. EGNN uses a total budget of 15 for LJ13; for angles a total
of ~`2π` is the natural starting point.

**5.3** An explicit boundary feature is cheap once `h` exists — concat a per-node
`[is_left_end, is_right_end]` (or the node degree) into `h_init`. This addresses the
`N=10 → N=64` generalisation gap in notebook cell 19, where 20 % of training sites are
ends versus 3 % at evaluation.

---

## ~~Phase 6 — score output preconditioning~~ (deleted)

Removed after measurement. Both halves of the argument for it were wrong:

1. The antithetic ±z pairing reduces the objective to implicit score matching, so the
   loss never sees `1/γ` (measured `loss_s(t=1e-6) = +8.4`, bounded).
2. More decisively, the *minimizer* is bounded: the net regresses onto
   `E[-z/γ | x_t] = ∇log p_t`, and `p_t → uniform` as `t→0` so `∇log p_t → 0`, while at
   `t→1` it is `∇log p₁` with norm 4.90. **The true score never exceeds ~5.** There is
   nothing to precondition.

Recorded here rather than silently dropped, so the reasoning isn't re-derived later.

---

## Order, and what to measure

> ⚠️ The "measure after" column below is the ORIGINAL plan and is partly superseded by
> the validation rule in the Status section: `ΔU`/`ΔS` are not acceptance criteria.
> Kept for the record; phases 0 and 3 were measured as written, 1–2 were validated by
> their symmetry/injectivity tests instead, 5 has no criterion yet.

| phase | files | risk | measure after |
|---|---|---|---|
| 0 | `xygnn.py`, `datasets/xy.py`, notebook | none | wall-clock/step |
| 1 | `xygnn.py` | low | ~~`ΔU`, `ΔS`~~ → endpoint-injectivity tests |
| 2 | `xygnn.py` | low | exact oddness under `θ → −θ` |
| 3 | `ot.py`, `train/xy.py`, tests | low | `xy_transport_cost` vs. `negate=False` ✔ |
| 4 | `xygnn.py`, `train/xy.py` | low | *skipped* |
| 5 | `xygnn.py` | moderate | ~~`ΔU`, `ΔS`, N-sweep~~ → **undecided** |

**Fixed benchmark for every phase**, so numbers stay comparable: `N=10`, `J=2`,
`B=256`, exact-sampled data (Phase 0.5), 700 steps at `lr=1e-3`, seed 0. That is the
harness the diagnostics already used, so there is a baseline to beat:

| config | `ΔU` err | `ΔS` err |
|---|---|---|
| `main`, `n_layers=4` (700 steps) | 5.0 % | 6.7 % |
| `main`, `n_layers=2` (700 steps) | 6.7 % | 6.6 % |
| `main`, `n_layers=4`, full 3 500-step notebook schedule | 1.9 % | 3.0 % |

Exact targets: `ΔU = -12.5599`, `ΔS = -5.1440` (`ΔU/N = -1.2560`, `-ΔS/N = +0.5145`).

The two pre-Phase-1 diagnostics are **done**; see `XY_MODEL_REVIEW.md` §0.5. Outcome:
F1 confirmed (endpoints bitwise identical), F2 refuted, depth useful to 2 layers and
flat after.

---

## Explicitly out of scope

- The bond-CNN and bond-MLP rebuilds (review §7.1, §7.2). They are a replacement for
  `XYChainGNN`, not a patch to it; revisit after Phase 5 shows what the repaired GNN
  can do.
- Any change to `eesi/interpolant.py`. Phase 4 and Phase 6 are deliberately designed as
  network-side changes for this reason.
- The `mcxy` proposal asymmetry (review §5.5) — the data is correct; it needs a comment,
  not a fix.
