# TAP prior: finite-equilibrium-length harmonic bonds

## Context

The TAP experiments currently start the interpolant from an **ideal (Gaussian) bead-spring
chain**: N−1 iid isotropic Gaussian bond vectors whose scale is set by one parameter,
`re_sqr` (the mean squared end-to-end distance). That prior is a poor structural match to
the target — the real TAP polymer's bonds are harmonic springs with a *finite* rest length
`b`, so its bond-length distribution is concentrated near `b` while the Gaussian prior's is
peaked at a non-physical short length.

The change: make the prior the **same harmonic-bond chain as the real model, minus excluded
volume and minus activity**. Bond orientations stay isotropic; bond magnitudes Q follow

    p(Q) ∝ Q² exp(−(k/2)(Q − b)²),   Q > 0

This shortens the transport the flow has to learn and makes the prior a physically
meaningful reference state (it is the b→finite generalization of what is there now; **b = 0
recovers today's prior exactly**, with k = 1/σ²).

Scope: **package code + tests only.** The notebook `experiments/TAP/TAP.ipynb` is explicitly
out of scope, though it will break — see "Notebook fallout" at the end.

Decisions already made with the user:
- Parameterize by two plain floats `(k, b)`, both required; retire `re_sqr` and `bond_sigma`.
- Require `n_dims == 3`; raise for anything else.
- Full rework of the TAP test suite's statistical tests.

---

## The math (verified numerically against `scipy.integrate.quad` before writing this plan)

Let σ = 1/√k and a = b/σ = b√k. Define the truncated Gaussian moments

    G_j = ∫_{−a}^{∞} u^j e^{−u²/2} du,
    G_0 = √(π/2)·(1 + erf(a/√2)),  G_1 = e^{−a²/2},  G_j = (j−1)·G_{j−2} + (−a)^{j−1} e^{−a²/2}

All G_j (j ≥ 0, a ≥ 0) are positive sums — no cancellation, numerically clean. Then with
the substitution Q = b + σu,

    Z₁ ≡ ∫₀^∞ Q² e^{−(k/2)(Q−b)²} dQ = σ · Σ_{j=0..2} C(2,j) b^{2−j} σ^j G_j
    E[Q^m]                            = σ · Σ_{j=0..m+2} C(m+2,j) b^{m+2−j} σ^j G_j  /  Z₁

so one recursion gives the normalizer and every moment. (Equivalent closed form for Z₁:
`σ(b²+σ²)√(π/2)(1+erf(b/(σ√2))) + b σ² e^{−b²/(2σ²)}` — but the recursion is the better
implementation because the moments come free.)

Useful identities, both verified — good test material:
- **E[Q²] = b·E[Q] + 3/k**  (virial / integration by parts)
- Per-bond entropy: **S₁ = log(4π Z₁) + 3/2 − (k b/2)(E[Q] − b)**, from
  S₁ = log(4πZ₁) + (k/2)E[(Q−b)²] and E[(Q−b)²] = 3/k − b(E[Q] − b).
  At b = 0 this reduces to (3/2)(1 + log 2πσ²), i.e. today's formula.

3D bond density and the chain log-density (the cumsum bond→position map is unit lower
triangular, so **|det J| = 1 still holds and there is still no Jacobian correction**):

    p(**Q**) = exp(−(k/2)(|**Q**| − b)²) / (4π Z₁)
    log p₀(x) = − Σ_k (k/2)(|x_k − x_{k−1}| − b)² − (N−1)·log(4π Z₁)

**Sampling.** Rejection on the non-dimensional radius y = Q√k, target ∝ y² e^{−(y−a)²/2},
proposal N(y*, 1) truncated to y > 0, where y* = (a + √(a²+8))/2 is the target's mode
(root of y² − ay − 2 = 0). Log-acceptance `2·log(y/y*) − (2/y*)(y − y*)`, which is exactly
log(target/proposal) minus its maximum — verified algebraically and by matching sampled
E[Q], E[Q²] to quadrature to 3–4 digits. Measured overall acceptance (including the y > 0
truncation loss) is **0.68 at a = 0 rising to >0.99 for a ≳ 10** — so use a single
oversample of `int(needed / 0.65) + 32` rather than the snippet's 1.2/1.5 split, which
under-shoots at a ≈ 1 and forces extra loop passes.

---

## Changes

### 1. `eesi/systems/tap/data.py` — the substantive work

**Delete** `bond_sigma`. **Add / rewrite**, in the `--- the prior ---` section:

- `_bond_moment_sums(k, b) -> list[float]` (private): the G_j recursion, returning the
  σ·Σ C(m+2,j)… sums for m = 0,1,2. Validates `k > 0`, `b >= 0`.
- `bond_moments(k: float, b: float) -> tuple[float, float, float]` → `(z1, mean_q,
  mean_q_sq)`. **This is the new single source of truth**, replacing `bond_sigma`'s role:
  `log_prior`, `prior_entropy` and `end_to_end_mean_sq` all route through it, so sampler
  and density cannot disagree. Docstring should state the convention the way
  `eesi/systems/lj13/data.py:68–91` states its energy convention (that comment block is the
  repo's house style for a pinned convention).
- `sample_bond_lengths(n, k, b, dtype=torch.float64, device="cpu", generator=None) ->
  (n,)`: the rejection loop above. Public because the tests want to hit the radial law
  directly. Must thread `generator` into **both** `torch.randn` and `torch.rand` (the repo
  threads generators everywhere for reproducibility; note in the docstring that the RNG
  stream advances a data-dependent number of draws, so results are reproducible for a fixed
  generator but not stream-aligned with other samplers). Keep the **exact b == 0 shortcut**
  (draw the 3-vector as `randn(n,3)/√k` — the Maxwell case) both for speed and because it
  makes "b = 0 is exactly the old prior" a testable statement.
- `sample_prior(n_batch, k, b, n_particles=N_DEFAULT, n_dims=N_DIMS, dtype, device,
  generator)`: raise `ValueError` unless `n_dims == 3`; draw `n_batch*(N−1)` magnitudes,
  draw directions as `randn(...,3)` normalized by `torch.linalg.vector_norm`, multiply,
  reshape to `(n_batch, N−1, 3)`, `cumsum` and prepend the literal-zero head row exactly as
  today (lines 155–159). Row 0 must stay an exact zero, not a rounded one.
- `log_prior(x, k, b)`: `q = bond_vectors(x).norm(dim=-1)`, then
  `-(k/2)*((q-b)**2).sum(-1) - (n-1)*log(4π z1)`. Raise for `d != 3`. Keep the existing
  "assumes anchored" paragraph — it is still true and still the right warning.
- `prior_entropy(k, b, n_particles=N_DEFAULT, n_dims=N_DIMS) -> float`: `(N−1)·S₁` with S₁
  as above. **New public function**: the Gaussian version of this formula is currently
  duplicated inline in the notebook *and* in `tests/tap/test_tap_data.py:159`, and it is the
  one closed form that has no drop-in analogue. Giving it a home kills the duplication.
- `end_to_end_mean_sq(k, b, n_particles=N_DEFAULT) -> float`: `(N−1)·E[Q²]`. The
  calibration diagnostic that `re_sqr` used to be, now derived instead of imposed. Name
  mirrors the existing `end_to_end_sq` observable.

**Rewrite the module docstring**, lines 1–11 (the function table: `bond_sigma` → `bond_moments`)
and especially lines 45–66 ("The prior: an ideal (Gaussian) chain"). The new section should
state: the bond law, that orientations are isotropic and magnitudes carry the Q² surface
Jacobian, the Z₁ closed form, that |det J| = 1 is unchanged, that O(3)-invariance is
**preserved** (this is what keeps `tap/ot.py` marginal-preserving — say so explicitly), and
that b = 0 recovers the ideal chain with k = 1/σ². Also update lines 22–24 ("If a
conservative piece is ever wanted as a diagnostic (harmonic bonds…)") — harmonic bonds are
now *in* the prior, so the sentence needs to distinguish "prior bond law" from "a target
density", which the module correctly insists TAP does not have.

### 2. `eesi/systems/tap/train.py` — parameter threading

- `tap_step(model, x1, k, b, align=True, batch=True, generator=None)` (line 49) and
  `train_si(data, k, b, steps=…, …)` (line 64) — `re_sqr` → two positional floats; update
  the `sample_prior` call at lines 57–59 and the `tap_step` call at line 80.
- CLI: replace `--re-sqr` (lines 98–99) with `--k` (required, "bond spring constant") and
  `--b` (required, "bond equilibrium length; 0 gives the ideal chain"); update the
  `print` at line 117 and the `train_si` call at line 122.
- Docstring usage lines 9–11 (`--re-sqr 4.0` → `--k 5.0 --b 1.0` or whatever pair matches
  the reference simulation).

### 3. `eesi/systems/tap/__init__.py`

Swap `bond_sigma` for `bond_moments`, `prior_entropy`, `end_to_end_mean_sq`,
`sample_bond_lengths` in both the import block (lines 25–39) and `__all__` (lines 44–58).
Update the docstring line 3 ("closed-form facts -- the ideal-chain prior").
No top-level `eesi/__init__.py` churn: TAP's prior is deliberately not re-exported there.

### 4. `eesi/systems/tap/dynamics.py` — docstrings only, no code

- Line 193: `log_prior(x0, re_sqr)` → `log_prior(x0, k, b)`.
- Lines 41–42: the rationale for `index_feature=True` says "an ideal chain has no excluded
  volume, so bonded neighbours are not reliably the nearest ones". With finite b the prior's
  bonded neighbour *is* much more nearly the nearest one — but the target still has no
  excluded volume in the prior's sense and the feature must stay. Reword so the
  justification doesn't rest on a premise that just changed.

### 5. `eesi/systems/tap/ot.py` and `eesi/systems/tap/interpolant.py` — no code changes

Both are already prior-agnostic in the ways that matter, and this is worth *not* disturbing:
- `ot.py` needs only O(3)-invariance, which the new prior keeps. Update the two docstring
  mentions of "the ideal (Gaussian) chain" (lines 8–11 and the table at 50–51).
- `interpolant.py` deliberately does **not** draw its latent `z` from the prior (see its
  lines 31–36). That design note goes from "a real difference from LJ13" to load-bearing —
  with a non-Gaussian prior, routing `_noise_like` through `sample_prior` would now be
  outright wrong, not merely different. Strengthen that comment; change no code.
- `eesi/interpolant.py` entropy estimators (`dot`/`div`/`zdot`/`bdot`) estimate ΔS only and
  never touch `log_prior`. **No changes.**

---

## Tests (`tests/tap/`) — full rework

Pick two parameter pairs at the top of each file, e.g. `K, B = 5.0, 1.0` for the prior and a
distinct `K1, B1 = 1.0, 2.5` wherever a fake "target" is needed.

**`test_tap_data.py`** — the file that carries the convention. Keep unchanged (they are the
best regression anchors and all still hold): `test_bonds_are_iid_and_isotropic`,
`test_tail_is_exactly_zero`, `test_anchor_is_idempotent_and_fixes_the_tail`,
`test_dof_and_basis`, `test_log_prior_is_o3_invariant`,
`test_log_prior_reads_bonds_not_positions`, `test_load_ref_data_*`. Rework:
- `bond_moments` vs `scipy.integrate.quad` for Z₁, E[Q], E[Q²] over several (k, b) — the
  direct check that the closed form is right.
- `bond_moments` satisfies **E[Q²] = b·E[Q] + 3/k**.
- Sampled bond lengths match `bond_moments` (mean and mean-square, ~1%) — and are all > 0.
- `E[Re²] ≈ end_to_end_mean_sq(k, b, N)` and `E|**Q**|² ≈ E[Q²]` (replaces
  `test_end_to_end_matches_re_sqr` / `test_bond_mean_square_is_re_sqr_over_n_bonds`).
- **b = 0 reproduces the old prior**: `bond_moments(k, 0)` gives E[Q²] = 3/k, and
  `log_prior(x, k, 0)` equals the Gaussian bond log-density with σ = 1/√k computed
  independently (this is the migration-safety test; it also preserves the old
  scipy-multivariate-normal comparison at line 133 in a form that still applies).
- `−E[log p₀] ≈ prior_entropy(k, b, N)` on a large sample (replaces the Gaussian-entropy
  test at line 150), plus a normalization check: `4π Z₁` from `quad` matches the constant
  `log_prior` actually subtracts.
- `test_re_sqr_scaling_is_linear` (line 69) has no analogue — replace with monotonicity:
  increasing b at fixed k increases `end_to_end_mean_sq` and the sampled E[Re²] together.
- `test_gyration_smaller_than_end_to_end` (line 194): the Re²/Rg² = 6 identity is
  ideal-chain-only. Assert Rg² < Re², and keep the ratio-6 check as a b = 0 case.
- **Update the hand-rolled `__main__` runner list (lines 223–259) in lockstep** — it names
  every test explicitly and will silently skip anything renamed.

**`test_tap_ot.py`** — `_pair` (lines 46–47) and line 299 build the fake target as
`sample_prior(B, RE_SQR*2.5, …)`, which only works because the old prior was a scale family.
Use the second (k, b) pair instead — still O(3)-invariant, which is all these tests need.
`test_coupled_noise_is_still_a_valid_prior_sample` (289–308) retargets to
`end_to_end_mean_sq`.

**`test_tap_training.py`** — `_data()` (32–35) same fix; `test_re_sqr_reaches_the_prior_draw`
(84–92) becomes `test_bond_params_reach_the_prior_draw`: doubling `b` at fixed `k` must grow
`(x0**2).sum()` substantially. Update its `__main__` runner list too.

**`test_tap_eesi.py`** — mechanical signature updates at `_anchored` (46–48). Keep
`test_noise_like_is_isotropic_not_a_chain` (66–79) verbatim; it now guards something
sharper.

---

## Verification

1. `python -m pytest tests/tap -q` — all green. This is the primary gate; the reworked
   `test_tap_data.py` is where a convention error would show up.
2. `python -m pytest tests -q` — nothing outside `tests/tap/` should move.
3. Independent normalization spot-check (not a test, a sanity run):
   `-log_prior(sample_prior(200_000, k, b, n_particles=20), k, b).mean()` must land on
   `prior_entropy(k, b, 20)` to ~3 decimals — this catches a wrong Z₁ or a missing 4π, the
   two errors the closed form is most likely to hide.
4. Smoke the CLI end-to-end on the reference data:
   `python -m eesi.systems.tap.train --k <k> --b <b> --steps 20 --n-data 2000` — confirm the
   loss is finite, the transport cost printed at step 0 is **lower** than the same run with
   `--b 0` (the whole point of the change: a closer prior means less transport).
5. `python -c "import eesi.systems.tap as t; print(t.__all__)"` — exports resolve, no stale
   `bond_sigma`.

## Notebook fallout (out of scope, listed so it isn't a surprise)

`experiments/TAP/TAP.ipynb` breaks at every `sample_prior(..., RE_SQR, ...)` call (cells
around lines 97, 158, 245, 310, 549, 635, 668), and cell 6's inline
`S_prior = 0.5*DOF*(1 + log(2π σ²))` plus cell 19's `S[p1] = S_prior + ΔS` must become
`prior_entropy(k, b, N)`. The new function exists precisely so that fix is one line.
