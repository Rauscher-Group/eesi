# LJ13 Free Energy, Entropy, and Divergence Diagnostics — Implementation Plan

**Context.** A trained equivariant OT flow-matching model (Klein, Krämer & Noé 2023) on LJ13,
checkpoints from `https://osf.io/srqg7/` ("Equivariant flow matching – data and models").

**Deliverable.** A single Jupyter notebook, `lj13_thermo.ipynb`. Everything runs and plots inline;
no CLI scripts, no separate modules unless a helper grows past ~50 lines (then put it in
`lj13_utils.py` and import it, keeping the notebook readable). Cache expensive artifacts
(`samples.pt`, `log_q.pt`, `div_traj.pt`) to `cache/` so plotting cells re-run cheaply.

**Three things to produce:**
1. Free energy `F = -log Z` via importance-sampled reweighting of the flow.
2. Mean energy `⟨U⟩`, hence entropy `S = ⟨U⟩ - F`.
3. Divergence `∇·v` of the velocity field along sampled trajectories.

---

## Physics (reduced units, k_B = 1, T = 1)

Target: `p(x) = exp(-U(x)) / Z` on the **center-of-mass-free subspace**.

| Quantity | Estimator |
|---|---|
| log Z | `logsumexp_i(log w_i) - log N`, with `log w = -U(x) - log q(x)`, `x ~ q` |
| F | `-log Z` |
| ⟨U⟩ | `Σ_i w̃_i U(x_i)`, self-normalized `w̃ = softmax(log w)` |
| S | `⟨U⟩ - F = ⟨U⟩ + log Z` (nats) |
| ESS | `exp(2·logsumexp(lw) - logsumexp(2·lw))` |

The flow gives exact `log q`, so no MCMC is needed — all four numbers come from one batch of
samples plus their log-densities. Include the identity `S = -⟨log p⟩_p` as a cross-check: it is
*not* an independent estimate (it's algebraically the same thing), but it catches sign and
normalization bugs immediately.

**Note the synergy:** `∇·v` is the integrand of the log-det. Deliverable 3 is computed for free
while doing Deliverable 1. Compute once, store the trajectory, use twice.

---

## Phase 0 — Assets

1. Clone `https://github.com/olsson-group/hollowflow` (MIT): `eq_ot_flow/` holds the equivariant OT
   FM implementation derived from Klein et al.; `run/trainer.py` is the entry point. Also check
   `https://github.com/noegroup/bgflow` (branches `factory`, `flowmatching`) — Klein's paper says
   the code "will be integrated" there.
2. Inventory the OSF archive in a notebook cell: which checkpoints exist, for which system, and for
   which objective (likelihood / OT-FM / **equivariant OT-FM**). Record exact filenames in markdown.
3. LJ13 data into `data/` per HollowFlow's README.

**Exit criterion:** checkpoint loads; `v(t, x)` returns finite output of the right shape on a
random mean-zero input.

---

## Phase 1 — Energy function (gates everything; this is where the bugs are)

Implement `lj_energy(x)` and validate it against the shipped dataset *before touching the model*.

### ⚠️ Blocker: the harmonic confinement term. Resolve first.

Köhler et al. (2020) and the DEM lineage use `U = U_LJ + c·U_osc` with
`U_osc = ½ Σ_i ||x_i - x_COM||²`. HollowFlow's appendix writes LJ13 with **no** oscillator term.

This is decisive for a free-energy calculation:

- **Without confinement, Z diverges.** As particles separate `U_LJ → 0`, so `exp(-U) → 1` over
  infinite volume. The 13-particle cluster is only *metastable*. "Free energy relative to an ideal
  gas" is then undefined without an added constraint — a container volume or a cluster definition
  (exactly the density axis in the nested-sampling phase diagrams).
- **With confinement, Z is finite** and everything below is well-posed.

**Action:** read the actual energy implementation in the repo you use and in Klein's released
config. Determine `c` (oscillator scale / `κ`). If `c = 0`, **stop** and pick a confinement
convention before computing any free energy. Record the resolved answer (`c`, `ε`, `r_m`, `τ`) in a
markdown cell at the top of the notebook.

### ⚠️ Trap: the factor of 2

Published form is `U_LJ = (ε/2τ) Σ_{i,j} [(r_m/d_ij)^12 - 2(r_m/d_ij)^6]` with `Σ_{i,j}` over
**ordered** pairs. Summing unique pairs `i<j` instead requires prefactor `ε/τ`, not `ε/2τ`. Getting
this wrong scales `F` and `⟨U⟩` by two while leaving W2 and energy histograms looking perfect.

### ⚠️ Trap: repo-specific regularization

Klein-lineage code often clamps or smooths the `r^-12` singularity. If a cutoff was applied at
training time but not in your evaluator, `log w` is wrong. Match the training-time energy exactly.
Also compute pairwise distances manually (`diff → d² → sqrt` with an epsilon) rather than `cdist`,
which misbehaves under autograd near zero separation.

### Validation gates (all must pass)
- Recompute energies on the OSF **test set**; histogram must match the published LJ13 figure.
- Invariance to random rotation, translation, permutation to ~1e-6.
- Relax random configs with L-BFGS at `c=0`; global minimum should land near the known bare-LJ13
  icosahedral value ≈ **-44.327 ε**. This single number confirms prefactor *and* sign.

---

## Phase 2 — Subspace, prior, sampling

LJ13 is 39 ambient dims; the model lives on the **36-dim mean-zero subspace**. Densities must be
defined there.

```
D_ambient = 39           # 13 × 3
D_eff     = 36           # (13-1) × 3
log p_prior(x) = -0.5*||x||² - 0.5*D_eff*log(2π)     # D_eff, NOT D_ambient
```

Using 39 in the normalizer silently offsets `log Z` by `1.5·log(2π) ≈ 2.76` nats and changes
nothing else — no sample metric will catch it. Assert `x.view(-1,13,3).mean(1) ≈ 0` after every op.

### Model adapter
Write one thin adapter cell exposing `v(t, x) -> (B, 39)`, mean-free-projected for safety. Verify:
- **Time convention.** Assume `t=0` → prior, `t=1` → target. If the checkpoint is trained the other
  way, wrap as `lambda t, x: -v_raw(1-t, x)`. A flipped convention flips the log-det sign and
  yields a plausible-looking but wrong `log Z`.
- **Mean-free output.** `v` must map the mean-zero subspace to itself.

Use `float64` — float32 log-dets accumulate visible error over the ODE.

### Augmented ODE (t: 0 → 1)
```
dx/dt = v(t, x)
dℓ/dt = -∇·v(t, x)
log q(x_1) = log p_prior(x_0) - ∫₀¹ ∇·v(t, x_t) dt
```
Use a **fixed-step RK4 grid** (not adaptive) so `∇·v` lands on a regular time axis for plotting;
trapezoid the divergence integrand. Validate once against `torchdiffeq` `dopri5` at `rtol=atol=1e-5`.

Also consider a **development placeholder**: a linear velocity field with analytic divergence, so
the whole pipeline can be unit-tested end-to-end before the checkpoint is wired in. Guard it with an
explicit `USING_DUMMY` flag that prints a loud warning.

---

## Phase 3 — Divergence of the velocity field

### ⚠️ Trap: divergence must be taken on the 36-dim subspace

Do **not** trace the 39×39 ambient Jacobian — it includes 3 translational directions outside the
model's support. Two routes, cross-checked:

- **Projector (simpler):** `P = I - (1/n)(1·1ᵀ ⊗ I₃)`, rank 36. Subspace divergence is `tr(PJP)`.
- **Explicit basis (rigorous):** orthonormal `B ∈ R^{39×36}` from QR of `P`. Define
  `ṽ(y) = Bᵀ v(By)`, take the plain 36×36 trace of `∂ṽ/∂y`. Unambiguous — use as ground truth to
  validate the projector route.

### Estimators — implement both
- **Exact:** `D_eff = 36` is small enough to afford the full Jacobian. `vmap(jacrev(...))` over the
  batch, then trace. This is the luxury of LJ13; use it.
- **Hutchinson:** Rademacher probes via `autograd.grad(v, x, grad_outputs=ε)`. Project probes into
  the subspace. Calibrate probe count against the exact value — a rare chance to measure Hutchinson
  variance against ground truth rather than guessing.

### Outputs
- `div_traj`: `(n_timesteps+1, batch)`.
- **Plots:** mean ∇·v vs. t with 10/50/90 percentile band; spaghetti overlay of ~50 trajectories;
  scatter of `∫∇·v dt` against final `log w`.
- **Interpretation:** `∇·v < 0` means local volume compression (density rising along the
  trajectory). Sharp features in `t` mark where transport strains; expect these to coincide with the
  high-weight outliers dominating `log Z`.

### ⚠️ Gate: log-det consistency
`-∫₀¹ ∇·v dt` accumulated along a trajectory **must** equal `log q(x_1) - log p_prior(x_0)` to
solver tolerance. If this fails, the ODE augmentation or the subspace projection is broken — stop
and fix before computing any thermodynamics. Also check step-size convergence: halve `n_steps`,
confirm `log q` stable to <0.01 nats.

---

## Phase 4 — log Z, F, ⟨U⟩, S

1. Draw N samples with log-densities. Start N=10⁵; Klein-lineage papers use 5×10⁵ for LJ13 metrics.
2. `log w = -U(x) - log q(x)`. Log space throughout; never exponentiate raw weights.
3. Report `log Z`, `F`, `⟨U⟩`, `S`, ESS, ESS%, and `max(w̃)`.
4. **Two kinds of uncertainty, both required** — they measure different things:
   - **Bootstrap** (~200 resamples) → Monte Carlo error given the samples you drew.
   - **Seed-to-seed spread** across ≥3 independent batches → whether the flow explores
     consistently.
   A gap between them means mode-finding is unreliable and the bootstrap CI is falsely reassuring.

### Be paranoid here

`log Ẑ` is **biased low** (Jensen), and the bias is governed by weight variance. On a rugged 36-dim
target the estimate can look rock-steady across seeds and still be badly wrong. Guardrails:

- Plot the `log w` histogram. A long left tail is fine; a **spiky right tail is fatal** — it means a
  handful of samples carry the mass and `log Z` is being decided by ~5 points.
- Report ESS on a batch of **≥1000**, never 16. (The DEM authors flagged exactly this: their
  original ESS used batch size 16; they now recommend 1000.)
- Plot `log Ẑ` vs. N on a log x-axis. Still climbing at N_max ⇒ report as a **lower bound**.
- If `max(w̃) > ~0.01`, treat the number as a lower bound regardless of what the CI says.

---

## Phase 5 — Independent reference (the actual validation)

The IS estimate grades the **model**. To know the **truth**, build an independent `log Z` on the
*identical* energy:

- **AIS / SMC** from the harmonic reference (analytically integrable, so the `β=0` end is exact —
  no reference-state ambiguity) to the full target. Geometric β-ladder, start ~1000 rungs, MALA or
  HMC transitions per rung, Jarzynski weights. Converge by increasing rungs until `log Z` plateaus.
- Or **nested sampling** if you want Z across temperatures for free.

Agreement between AIS and flow-IS within CIs is the validation. **Disagreement is the interesting
result** — it means the flow is missing basins (icosahedral vs. other minima), and the sign of the
discrepancy tells you which way.

---

## Notebook layout

1. Setup; checkpoint inventory; **resolved energy convention** written out explicitly
2. Energy validation gates (invariance, test-set histogram, global minimum)
3. Model adapter + time-convention check
4. Divergence estimators + projector-vs-basis cross-check + Hutchinson calibration
5. Sampling + `log q`, with the **log-det consistency gate**
6. Distributional sanity (energy and interatomic-distance histograms, model vs. test set) — if
   these are off, the thermodynamics is meaningless regardless of how clean the arithmetic is
7. **log Z / F / ⟨U⟩ / S** table, bootstrap CIs, convergence-vs-N plot, `log w` histogram
8. **Divergence** plots
9. AIS reference comparison
10. Summary table

---

## Order of operations

Phase 1 gates everything; the log-det check in Phase 3 gates Phase 4. Do not compute a free energy
until (a) the harmonic-term question is resolved, (b) the energy passes invariance and
minimum-energy tests, and (c) log-det consistency passes.

The failure mode of this project is a number that is precise, reproducible, and wrong. Every
distributional metric in the literature — W2, MMD, energy histograms — is blind to the three bugs
that matter most here: the factor of 2, the 36-vs-39 normalizer, and the time-convention sign.
