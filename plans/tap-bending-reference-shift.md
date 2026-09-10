# TAP at Pe = 0: reading the angle potential back in, to compare with thermodynamic integration

**Status: implemented 2026-08-09.** Package code, tests and notebook section 6 are all in.
Confirmed with the user: TI's ideal reference uses k = 100, b = 1, matching the notebook's
hard-coded `K_BOND` / `B_BOND`, so the bond offset in 6.1 comes out exactly zero. The only
thing outstanding is the TI numbers themselves — `TI = {"dS": None, "dU": None, "dF": None}`
in cell 6.2, to be filled in by hand.

## Context

For the passive chain (Pe = 0) there is an external thermodynamic-integration result: ΔF and
ΔS between the **full system** (harmonic bonds + excluded volume) and an **ideal reference**
that has the same harmonic bonds, no excluded volume, and **no bending potential**.

The interpolant measures something else. Its ΔS runs from the flow's prior — the same
harmonic-bond chain **plus** the fitted bending potential γ(cos θ − cos θ₀)² — to the same
target. The two reference states differ by exactly one thing, the angle potential, so

    dS_TI(ideal -> full)  =  dS_interpolant(prior -> full)  +  dS(ideal -> prior)

and the last term is a closed form: it is purely angular, and — this is the useful part —
**independent of k and b**, because the bond law p(Q) ∝ Q² e^{−(k/2)(Q−b)²} is identical in
both states and cancels term by term. The same bridge exists for U and F, so the whole TI
triple can be re-referenced to the prior and compared against the flow.

Everything needed is already in `eesi/systems/tap/data.py` (`angle_moments`,
`bond_moments`, `prior_entropy`); this plan adds the two missing thermodynamic potentials
alongside the entropy, the bending bridge itself, and a notebook section that does the
reconciliation.

**Scope note.** `data.py`'s docstring says, in capitals, that no energy function lives there,
and that is about the *target*: an active polymer's steady state is not Boltzmann and has no
U(x). Nothing here contradicts it. `prior_energy` / `prior_free_energy` are statements about
**p0**, whose Boltzmann factor the module already writes down in `log_prior`, and they must be
documented as such — facts about the base distribution, never about p1. At Pe = 0 the target
does happen to be Boltzmann, but we still never evaluate its potential (see "Out of scope").

---

## The math (verified numerically before writing this plan; numbers below are real)

Write the prior's configurational partition function straight off `log_prior`'s constant:

    Z = (4 pi Z1) . (2 pi Z1 Z_ang)^(N-2)        <- N-1 radial factors, N-2 angular ones
    F = -log Z
      = -[ log(4 pi Z1) + (N-2) log(2 pi Z1 Z_ang) ]
    U = <U>_p0
      = (N-1) (k/2) E[(Q-b)^2]  +  (N-2) gamma E[(u-u_0)^2]
      = (N-1) (k/2) (3/k - b (E[Q] - b))  +  (N-2) gamma E[(u-u_0)^2]

with Z1, E[Q] from `bond_moments` and Z_ang, E[(u−u₀)²] from `angle_moments` (the latter's
third return is the moment about **u₀**, the potential's centre, which is exactly the one U
needs). Comparing with `prior_entropy` term for term shows the identity it already encodes:

    S = U - F        (kT = 1, nats, per chain, on the tail-anchored subspace)

**The bending bridge.** Setting γ = 0 gives the freely-jointed chain with Z_ang = 2 and no
angular energy. Differencing the two states, every radial term cancels:

    dU_bend = +(N-2) gamma E[(u-u_0)^2]
    dF_bend = -(N-2) log(Z_ang / 2)
    dS_bend =  dU_bend - dF_bend = (N-2) [ gamma E[(u-u_0)^2] + log(Z_ang / 2) ]

all three functions of (γ, u₀, N) alone. Sign convention throughout: **ideal → ideal+angle**,
i.e. `X_prior − X_ideal`.

Checked numerically at the notebook's own calibration (N = 20, k = 100, b = 1, and
γ = 2.330904, cos θ₀ = 0.534582 fitted from `data/tap_N20_Pe0.npy`):

| quantity | value (nats / kT) | check |
|---|---|---|
| S_ideal = `prior_entropy(k, b, 0, ·)` | +31.677351 | — |
| S_prior = `prior_entropy(k, b, γ, u₀)` | +25.148311 | — |
| dS_bend | **−6.529040** | equals the difference above to machine precision |
| dU_bend | **+6.355509** | MC over 2·10⁵ prior draws: +6.3576 |
| dF_bend | **+12.884549** | per-angle FEP over uniform u, ×(N−2): +12.8846 |
| U − F | 25.148311 | equals `prior_entropy` exactly |

Also verified: dS_bend is unchanged at (k, b) = (3.7, 2.9) — the k/b-independence is real, not
approximate — and at γ = 0 the value of cos θ₀ is completely inert.

The signs are physically the right way round: the bending potential *constrains* the chain, so
it lowers the entropy (−6.53) and costs free energy (+12.88).

---

## Changes

### 1. `eesi/systems/tap/data.py` — three additions

**`prior_free_energy(k, b, gamma, cos_theta_0, n_particles=N_DEFAULT, n_dims=N_DIMS)`**
→ `-log Z` for the prior, from the same `bond_moments` / `angle_moments` values that
`log_prior` uses for its normalising constant. One line of arithmetic. Docstring must say:
this is p0's free energy, on the tail-anchored subspace, in kT with the configurational
integral only — no kinetic/momentum factor, no translational volume — and that those omitted
pieces cancel in any difference between two states of the same chain, which is the only way
this is ever used.

**`prior_energy(k, b, gamma, cos_theta_0, n_particles=N_DEFAULT, n_dims=N_DIMS)`**
→ `<U>_p0`, the two-term expression above. Docstring must carry the module's own warning
forward: this is the mean potential energy **of the base distribution**, not of the target;
the target has no potential at Pe > 0, and even at Pe = 0 nothing here evaluates it.

**`bending_deltas(gamma, cos_theta_0, n_particles=N_DEFAULT)`** → `(dS, dU, dF)` for
freely-jointed → semiflexible. Implemented **directly** from `angle_moments` (three lines,
manifestly k/b-free), not as a difference of the two functions above — the independence from
the bond law is the theorem worth stating in code, and the test then pins it by checking the
direct form against the difference. Docstring gives the sign convention explicitly
(`X_semiflexible − X_freely_jointed`), notes that `cos_theta_0` is inert at γ = 0, and states
the intended use: shifting an externally computed ideal-referenced ΔS onto the prior's
reference state, since the interpolant's ΔS starts from the prior.

`prior_entropy` gains one cross-reference line: it now equals
`prior_energy(...) - prior_free_energy(...)`, which is the cheapest available check that all
three agree on the 4π/2π convention.

Export all three from `eesi/systems/tap/__init__.py` if that module re-exports names (check;
match whatever `prior_entropy` does).

### 2. Tests — `tests/tap/test_tap_data.py`

Grouped with the existing `prior_entropy` tests:

1. **`test_entropy_is_energy_minus_free_energy`** — the identity, over a grid of
   (k, b, γ, u₀, N) including γ = 0 and b = 0. This is the one that catches a 4π/2π slip.
2. **`test_bending_deltas_match_the_prior_differences`** — `bending_deltas(γ, u₀, N)` equals
   `(prior_entropy, prior_energy, prior_free_energy)` at (k, b, γ, u₀) minus the same at
   (k, b, 0, ·), for several (k, b) — which simultaneously pins the k/b-independence.
3. **`test_bending_deltas_vanish_at_zero_gamma`** — all three exactly 0.0 at γ = 0, for any
   `cos_theta_0`, and `cos_theta_0` inert there.
4. **`test_bending_free_energy_matches_quadrature`** — dF_bend against a deterministic
   trapezoid/Simpson integral of `exp(-gamma (u-u_0)^2)` over [−1, 1]: −(N−2)·log(⟨e^{−U}⟩)
   under uniform u. Deterministic, no seeds, agrees to ~1e-6 with a few thousand points; the
   per-angle route is the one to check, because a whole-chain FEP is variance-dominated
   (measured: 12.78 vs 12.88 at 2·10⁵ chains, useless as a test).
5. **`test_bending_energy_matches_prior_samples`** — dU_bend against
   `gamma * ((bond_cosines(x0) - u0)**2).sum(-1).mean()` on a prior draw, loose tolerance
   (~1%). Ties the analytic moment to the sampler that actually produces the training data.
6. **`test_bending_deltas_signs`** — for γ > 0: dS < 0, dF > 0, dU > 0. Cheap, and it is the
   statement a reader will want to trust without re-deriving anything.

### 3. `experiments/TAP/TAP.ipynb` — a new section 6

Placed after the existing section 5 (entropy estimates), before the current trailing cells.
Add the three new names to the section-0 import cell.

**6.0 markdown — the bookkeeping.** The two reference states, the bridge identity, and the
diagram of what is being compared:

    ideal (bonds only) --[ analytic, dS_bend ]--> prior (bonds+angle) --[ flow, dS_interp ]--> full
    ideal (bonds only) ------------------[ TI, dS_TI ]---------------------------------------> full

so `dS_TI == dS_bend + dS_interp` is the check, and equivalently
`S_full = S_ideal + dS_TI = S_prior + dS_interp` reads it as two routes to one absolute
entropy. State plainly that this section applies to **Pe = 0 only**: TI needs an equilibrium
target, so at Pe > 0 there is nothing to reconcile against and the interpolant stands alone.

**6.1 code — the analytic bridge.** Guard on the dataset (`IS_PASSIVE = "Pe0" in
str(DATA)`; print a skip notice otherwise), then

```python
dS_bend, dU_bend, dF_bend = bending_deltas(GAMMA_BEND, COS_THETA_0, N)
S_ideal = prior_entropy(K_BOND, B_BOND, 0.0, 0.0, N)
assert abs((S_prior - S_ideal) - dS_bend) < 1e-9      # the two routes agree
```

and a printed table of (S, U, F) for the ideal state and the prior, with the three deltas.
The assert is worth keeping in the notebook, not just in the tests: it is what catches a
`GAMMA_BEND` / `COS_THETA_0` that has drifted out of sync with the trained model.

**6.2 code — the reconciliation.** The TI numbers are external, so they enter as a dict at
the top of the cell, filled in by hand, with a comment recording their provenance and units:

```python
# Thermodynamic integration, ideal (harmonic bonds, no EV, no bending) -> full passive
# chain, per chain, in kT / nats. Sign convention: X_full - X_ideal.
TI = {"dF": None, "dU": None, "dS": None}    # <- fill in
```

Then two printed blocks:

- **Prior-referenced TI triple**: `dX_TI - dX_bend` for each of S, U, F — the TI result moved
  onto the flow's own reference state.
- **Residual**: `dS_interp - (TI["dS"] - dS_bend)`, next to the interpolant's own error bar
  from section 5's bootstrap. This single number is the whole point of the section.

If `TI["dU"]` is available, add the derived line `dF_interp = dU_TI_prior - dS_interp` — the
flow supplies the entropy TI would otherwise have to integrate for, so a free energy comes out
of a quantity the flow measures directly. Guard the whole block on the dict being filled in
(`if all(v is not None for v in TI.values())`) so the notebook still runs top to bottom
without the numbers.

**6.3 code — the bond-parameter guard.** The reconciliation is only valid if the ideal state
TI integrated from is the *same* ideal state, which means the same (k, b). Section 1 currently
**overrides** the measured values with hard-coded ones (`K_BOND = 100.0` after
`1/var(Q) = 91.32`, `B_BOND = 1.0` after `E[Q] = 1.0241`), presumably because those are the
simulation's actual Hamiltonian parameters — which is exactly right for this comparison and
exactly wrong to leave implicit. Add a cell that states the requirement and, if they differ,
computes the extra bridge term generally rather than assuming it away:

```python
K_SIM, B_SIM = 100.0, 1.0          # the simulation's bond Hamiltonian; MUST match TI's ideal
dF_bond = -(prior_free_energy(K_SIM, B_SIM, 0.0, 0.0, N) - prior_free_energy(K_BOND, B_BOND, 0.0, 0.0, N))
```

— zero when they agree, and the correct offset when they do not. Same for U and S. Fold it
into the bridge before the residual is printed.

**6.4 markdown — reading the residual.** A short diagnostic ladder, so a non-zero answer is
actionable rather than alarming:

- a residual much larger than the flow's error bar, and *insensitive* to more training →
  a convention mismatch (see the checklist below), not a model error;
- a residual that shrinks as training continues → the flow, not the bridge;
- a residual that changes when γ / cos θ₀ are recalibrated but the model is not retrained →
  the prior and the trained model have gone out of sync.

**Also update the section 5 markdown.** It currently says flatly "There is **no free-energy
cross-check here**, unlike the LJ13 notebook". That stays true for Pe > 0 and is now false at
Pe = 0; qualify it and point forward to section 6.

---

## Convention checklist (the offsets that would break the comparison)

Worth writing into 6.0 as a list, because every one of these is a silent constant:

1. **Translational gauge.** Our S and F are on the tail-anchored subspace (dim 3(N−1)); a TI
   calculation in a box carries an extra log V. It **cancels** in ideal → full and in
   ideal → prior alike, so no correction is needed — but only because it is the same factor
   in both, which is worth stating rather than assuming.
2. **Kinetic terms.** Momentum integrals cancel in every difference here. Only the
   configurational part is ever compared.
3. **Units.** kT = 1, nats, **per chain** (not per monomer, not per bond, not in k_B with
   log₁₀). N−1 = 19 bonds and N−2 = 18 angles is the counting; a per-monomer TI number is a
   factor of 20 away from this.
4. **Sign.** Everything here is (final − initial) with initial = the *less* constrained state.
5. **Bond parameters.** TI's ideal must use the same (k, b) as the prior — see 6.3.
6. **Excluded volume in the ideal state.** TI's reference must have EV fully off, matching the
   prior, which has none.

---

## Out of scope (named so it is a decision, not an oversight)

- **Any evaluation of the target's potential.** At Pe = 0 the full chain is Boltzmann with
  U = bonds + excluded volume, so an importance-weighted ΔF like LJ13's is possible *in
  principle*. It is not possible *here*, because the excluded-volume parameters are not in
  this repo and no `tap_energy` exists. If those parameters ever arrive, that is a separate
  plan, and it would give a second, independent cross-check at Pe = 0.
- **Any change to training, the interpolant, or the entropy estimators.** This is
  post-analysis arithmetic on top of a ΔS that is already produced; nothing upstream moves.
- **The active datasets.** Pe > 0 gets nothing from this section by construction.

---

## Verification

1. `pytest tests/tap -q`.
2. Numbers already confirmed while writing this plan and expected to reproduce exactly:
   at N = 20, k = 100, b = 1, γ = 2.330904, cos θ₀ = 0.534582 →
   `bending_deltas` = (−6.529040, +6.355509, +12.884549), `prior_entropy` = 25.148311,
   ideal = 31.677351.
3. Run section 6 of the notebook top to bottom with `TI = {...: None}` — it must print the
   analytic bridge and skip the reconciliation cleanly.
4. Fill in the TI numbers and read the residual against section 5's bootstrap error bar.
