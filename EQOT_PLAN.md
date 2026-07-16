# Equivariant OT Coupling for EESI — LJ13 and the XY Chain

**Scope.** Three deliverables, in dependency order:

- **Part A** — move `experiments/lj13_egnn.py` into the `eesi` package as `eesi/lj13.py`. ✅ **done**
- **Part B** — equivariant-OT coupling for LJ13 (`S(13) ⋊ SO(3)`), reproducing Klein, Krämer & Noé
  (arXiv:2306.15030). ✅ **done** (B1–B5).
- **Part C** — the analogous coupling for the 1D XY chain over `Z₂ ⋉ U(1)` (reflection + global
  rotation). ✅ **done** (C1–C4).

**Modules:** `eesi/lj13.py` (system + net), `eesi/ot.py` (both couplings, vectorized),
`eesi/ot_reference.py` (scipy oracles), `experiments/train_lj13.py`, `experiments/train_xy.py`,
`experiments/bench_ot.py` (B5), `tests/test_ot.py`, `tests/test_training.py`. 111 tests pass.

**Measured so far** (LJ13, fp64, CPU, mean transport cost as % of random pairing):

| B | coupling time | `(B,B,13,13)` | cost vs random |
|---|---|---|---|
| 32 | 18 ms | 1.3 MB | 14.8% |
| 64 | 21 ms | 5.3 MB | 15.0% |
| 128 | 77 ms | 21.1 MB | 13.7% |
| 256 | 296 ms | 84.5 MB | 12.3% |

The `B²` memory is the binding constraint, not time: `(B,B,13,13)` fp64 is 84.5 MB at B=256 and
1.4 GB at B=1024. Equivariant OT is a small-batch method, so this is not the limitation it looks
like — but any test that needs many samples must **accumulate over small batches**, not raise B.

> Renamed from `EQOT_LJ13_PLAN.md`: the scope is now two systems.

**Strategy, unchanged from the original plan.** SemlaFlow's `semlaflow/data/interpolate.py` is the
faithful reference implementation of Klein's algorithm, but it runs scipy in a Python double loop
over molecule objects and hides the cost in dataloader workers. Our systems are homogeneous — no
atom types, bonds, padding, or variable sizes — so the whole thing vectorizes. We port SemlaFlow's
algorithm to a scipy **oracle**, then write a GPU path validated against it.

> This is data-driven flow matching / interpolant training. No energy function is required by the
> coupling itself. The energies in `eesi/lj13.py` exist for the thermodynamics work
> (`experiments/LJ13_PLAN.md`), not for this.

---

## The invariance condition (read this first — it is the whole design)

Both parts transform the **noise** `x0`, never the data. The convention is SemlaFlow's, and the
original plan justified it as "the prior is exactly invariant, the data only approximately." That
justification is incomplete, and the complete version is what determines the XY group:

> **Transforming the noise preserves the noise marginal iff BOTH `p₀` and `p₁` are G-invariant.**

Why: for fixed `x1`, the alignment map `T(x0) = ρ(g*(x0,x1))·x0` satisfies
`g*(ρ(h)x0, x1) = g*(x0,x1)h⁻¹`, hence `T(ρ(h)x0) = T(x0)`. `T` is *invariant* under the group
acting on `x0` — it collapses each orbit onto the single representative best matching `x1`. What
undoes the collapse is averaging over `x1`: if `p₁` is G-invariant then `x1`'s own orientation is
uniformly spread over its orbit, and the marginal is restored. If `p₁` is **not** G-invariant, it
never is, and the training coupling has an `x0`-marginal that is not the prior you sample from at
inference.

Counterexample to keep in mind: `p₀ = N(0,I)` on `R²` (exchangeable), `p₁ = δ(1,0)` (not). Aligning
the noise over `S(2)` sorts every `x0` descending. The marginal is the sorted-Gaussian law, not
`N(0,I)`. Exchangeability of the prior is **necessary but not sufficient.**

Consequences, per system:

| System | `p₀` | `p₁` | Admissible G |
|---|---|---|---|
| LJ13 | COM-free isotropic Gaussian — invariant under `S(13) ⋊ O(3)` | LJ13 Boltzmann — invariant under `S(13) ⋊ O(3)` | `S(13) ⋊ SO(3)` ✓ |
| XY chain | i.i.d. uniform on `(−π,π]` — invariant under `S(L) ⋊ U(1)` | open-chain NN Boltzmann — invariant under `Z₂ ⋉ U(1)` only | `Z₂ ⋉ U(1)` ✓, `S(L)` ✗ |

**`S(L)` is not admissible for the XY chain.** The open chain's NN coupling
(`E = J Σᵢ cos(θᵢ₊₁ − θᵢ)`, `classicalXY.py`) fixes who neighbours whom, so a general permutation
changes the energy: `p₁` is not `S(L)`-invariant, and permuting the noise would leak chain structure
into the prior marginal. `XYChainGNN` is not permutation-equivariant either — it is built on a
static chain graph. Both facts point the same way. Do not add `S(L)` to the XY cost; if you want to
measure the damage, Test C4 below is the one that catches it.

---

## Part A — Move `experiments/lj13_egnn.py` to `eesi/lj13.py`

**A single module**, moved essentially whole: the Satorras E(n)-GNN reimplementation
(`unsorted_segment_sum`, `E_GCL`, `EGNN`, `LJ13Dynamics`, `from_checkpoint`), the LJ13 physics
(`sample_prior`, `lj_energy`, `oscillator_energy`, `target_energy`, `delta_energy`), and the
thermodynamics (`DOF`, `_subspace_dirs`, `divergence`, `log_prior`, `rk4_sample`,
`integrate_with_logdet`, `free_energy`). Rename and tidy; do not restructure.

The Satorras reimplementation is the point, not an embarrassment to be hidden: it is the
architecture the surrounding literature uses, and keeping it readable next to the system it models
is worth more than factoring it out. `eesi/egnn.py` (the cutoff-EGNN with global linear attention)
stays where it is — it is currently unused by any experiment, so the two nets' conflicting edge
conventions (`src/dst` + scatter-by-`dst` vs `row/col` + scatter-by-`row`) never meet. Leave
`egnn.py` and `tests/test_egnn.py` alone; retiring dead code is a separate decision from this one.

### Refactor gates
1. **Checkpoint still loads.** `LJ13Dynamics.from_checkpoint(...)` with `strict=True` — the existing
   call already errors loudly if the key mapping drifts. Keep it strict.
2. **Bit-identical velocity.** Fixed seed, fixed `x`: `v_new(t,x) == v_old(t,x)` exactly (not
   `allclose` — this is a pure move, any difference is a bug).
3. **`experiments/lj13_sampling.ipynb` still runs.** Update its imports; re-run top to bottom.
   `lj13_sampling.ipynb` and `LJ13_PLAN.md` both reference the old path.
4. Export from `eesi/__init__.py`; delete `experiments/lj13_egnn.py` only once 1–3 pass.

### Noted, not scheduled
`lj13.py::divergence` (forward-mode `jvp` over 36 fixed basis directions) overlaps
`model.py::_divergence` / `_div_exact` / `_div_hutchinson`. They solve the same problem with
different tradeoffs and different subspace handling. **Leave both.** Unifying them is a separate
change with its own correctness gates, and Part B does not need it.

---

## Part B — LJ13 equivariant OT (`S(13) ⋊ SO(3)`)

Unchanged from the original plan except for module paths. Deliverables:

- `eesi/ot_reference.py` — scipy oracle, ported from SemlaFlow. Slow, obviously correct.
- `eesi/ot.py` — vectorized GPU implementation. The real thing. Also hosts Part C.
- `tests/test_ot.py` — oracle-equivalence + invariance tests for both systems.

(Flat modules, matching the existing package layout — `egnn.py`, `mlp.py`, `model.py`, `xygnn.py`.)

### What Klein's algorithm actually is

Cost `c̃(x0, x1) = min_{g∈G} ‖x0 − ρ(g)x1‖²`, `G = S(N) ⋊ SO(3)`, approximated **sequentially**:

1. `s̃ = argmin_{s∈S(N)} ‖x0 − ρ(s)x1‖²` — Hungarian on the 13×13 inter-particle cost.
2. `R* = argmin_{R∈SO(3)} ‖x0 − R ρ(s̃) x1‖²` — Kabsch.
3. Fill `M[i,j]` with the resulting aligned cost; run **mini-batch OT on M**.

Not alternating, not iterated. Klein's Appendix A.9 tested fancier approximations and found no
significant improvement at additional cost. Don't re-litigate this.

Why it matters: permutations scale as `N! = 13! ≈ 6.2e9`, sample pairs only as `B²`. Naive minibatch
OT never finds a matching permutation. The flip side is that equivariant OT works at **small batch
sizes**, which caps the `B²` blowup.

### Phase B0 — Dependencies ✅ done

`torch-linear-assignment`, `scipy`, and `POT` all import cleanly in this environment. The original
plan budgeted an hour for the custom-CUDA build tax and sketched a dataloader fallback; neither is
needed. `all_data_LJ13-1000.npy` (1.6 GB) and `LJ13_eq_OT_flow_matching` are already in
`experiments/`.

Still worth reading: clone `https://github.com/rssrwn/semla-flow` for reference only —
`semlaflow/data/interpolate.py`, functions `_ot_map`, `_match_mols`, `_match_cost`.

**Do not use `olsson-group/hollowflow` as the equivariant-OT reference.** Despite its paper's claim,
its trainer does plain minibatch OT (`torchcfm.OTPlanSampler`) on the *unaligned* Euclidean cost and
then Kabsch-aligns the already-selected pairs — the order is inverted from Klein and there is no
permutation search at all (`linear_sum_assignment` appears nowhere in the repo). Its Kabsch also
omits the determinant sign correction, making it O(3) rather than Klein's SO(3). Useful for the LJ
energy (`eq_ot_flow/LJ.py`) and bgflow wiring — just not for this.

### Phase B1 — The scipy oracle (`eesi/ot_reference.py`)

Port SemlaFlow's three functions to plain `(B, 13, 3)` tensors, dropping `GeometricMol`, padding,
and truncation:

```
match_pair(x0_i, x1_j) -> (x0_aligned, cost)      # Hungarian on 13x13, then Kabsch
ot_map(x0, x1)         -> (x0_aligned_reordered)  # B*B match_pair calls, then Hungarian on M
```

Fidelity requirements, all directly from SemlaFlow's code:
- **Permutation first, then rotation.** Not the reverse.
- Rotation via `scipy.spatial.transform.Rotation.align_vectors` → a **proper** rotation (SO(3)).
  FLOWR's parallel helper is explicitly documented "no reflection allowed". Matches Klein Eq. 16.
- **Return the aligned configurations, not just the costs.** SemlaFlow returns `mol_matrix[r][c]` —
  the already-transformed objects. Computing the cost with alignment and then interpolating the
  *unaligned* pair is the classic bug here.
- **Transform the noise, not the data.** See the invariance condition above.
- SemlaFlow's `_match_cost` uses `.mean()` (MSE) not `.sum()`. For LJ13 all configurations are the
  same size, so this is a uniform rescaling of `M` and cannot change the assignment. Pick one, be
  consistent, note the divergence from Klein's `‖·‖²` if comparing published numbers.

This module is never used in training. It exists to be right.

### Phase B2 — Vectorized GPU path (`eesi/ot.py`)

All under `torch.no_grad()`. **You never backprop through the coupling** — it is a data-pairing step,
and the regression loss sees the aligned pair as fixed targets. A hard solver is correct here; a
Sinkhorn relaxation would be strictly worse.

**Step 1 — Center.** Remove CoM from `x0` and `x1`. Assert `x.mean(dim=1) ≈ 0`. Kabsch and the cost
both assume it.

**Step 2 — Pairwise particle cost `(B, B, 13, 13)`.** Do **not** materialize the `(B,B,13,13,3)`
difference tensor — 3× the memory for nothing. Use the expansion:

```
sq0   = (x0**2).sum(-1)                          # (B, 13)
sq1   = (x1**2).sum(-1)                          # (B, 13)
inner = torch.einsum('iad,jbd->ijab', x0, x1)    # (B, B, 13, 13)
D     = sq0[:, None, :, None] + sq1[None, :, None, :] - 2*inner
D     = D.clamp_min(0)                           # the expansion can go slightly negative
```

At B=256, fp32: ~44 MB. The clamp matters because `‖a‖²+‖b‖²−2⟨a,b⟩` loses precision near zero.

**Step 3 — Batched Hungarian.** `batch_linear_assignment(D.reshape(B*B, 13, 13))`, then
`assignment_to_indices`. **Verify the dtype it accepts** — the README example uses an integer tensor;
confirm fp32 works. If you run fp64 elsewhere, cast down: the assignment is a discrete argmin and
fp32 is ample except in exact ties, which don't matter for a training-time coupling.

**Step 4 — Apply permutations.** Gather into `(B*B, 13, 3)` permuted form.

**Step 5 — Batched Kabsch, singular values only.** `H = Yᵀ X` → `(B*B, 3, 3)`. The aligned cost needs
**only** the singular values:

```
c̃ = ‖X‖² + ‖Y‖² − 2(σ₁ + σ₂ + d·σ₃),    d = sign(det H)
```

- `sign(det(V Uᵀ)) = sign(det H)` (since `det H = det(U)det(V)·Πσ` and `Πσ ≥ 0`), so the SO(3) sign
  correction comes from a cheap `torch.linalg.det` — no need for `U`, `V`.
- Singular values only — never the full `svd` — for the `B²` cost entries. **Only materialize `R` for
  the `B` pairs the outer OT actually selects.** cuSOLVER's batched SVD has real per-matrix overhead
  at 3×3; this cuts the work substantially. ⚠️ **It does not cut it enough, and `torch.linalg.svdvals`
  is the wrong call** — B5 measured it at 84–90% of the whole coupling, slower on GPU than on CPU.
  Use `ot._svdvals_3x3` (eigenvalues of `HᵀH`), ~20× faster with no accuracy cost. See Phase B5.
- `‖Y‖² = ‖x1_j‖²` — permutation preserves norm, don't recompute it.
- `d = +1` unconditionally gives O(3) instead of SO(3). Klein specifies SO(3); expose as a flag,
  default SO(3).

**Step 6 — Outer OT.** `linear_sum_assignment` on the `(B,B)` matrix — one scipy call, ~1 ms, not
worth moving to GPU. (Or `batch_linear_assignment` with a leading singleton dim, or `ot.emd`.)

**Step 7 — Materialize.** For the `B` selected `(i,j)` pairs: full SVD → `R`, apply permutation +
rotation, return the aligned pair. The only place `R` is ever constructed.

### Phase B3 — Validation (`tests/test_ot.py::TestLJ13`)

1. **Oracle equivalence.** B=32, GPU path vs `ot_reference`. **Compare total transport cost, not the
   permutations** — ties break differently between solvers. Agreement to ~1e-5 (fp32).
2. **Cost floor.** `c̃(x,x) == 0` for `x0 = x1` (identity permutation, identity rotation).
3. **Cost monotonicity.** `c̃(x0,x1) ≤ ‖x0 − x1‖²` always. A violation means a sign or transpose
   error in Kabsch.
4. **Invariance — but only the invariances that actually hold.** ⚠️ **The original plan got this
   wrong, and the wrong version fails on correct code.** It asked for `c̃` to be unchanged under a
   random rotation *and* permutation of `x1`. The **exact** minimum over `S(N) ⋊ SO(3)` is
   rotation-invariant, but Klein's **sequential approximation** to it is not: the Hungarian runs on
   *unrotated* coordinates, so rotating `x1` changes which permutation it selects. Measured: rotating
   `x1` alone moves the sequential cost by `−1.9 … +10.1` on a baseline of `11.4` — and **both signs
   occur**, which is the signature of an approximation gap rather than a bug.

   Only transformations that leave the Hungarian's cost matrix invariant are exact symmetries. Test
   these, all to ~1e-12:
   - **Permute `x1` alone** → exact (the cost matrix is column-permuted; same optimum). Verified 1.8e-15.
   - **Permute `x0` alone** → exact, same reason (row permutation).
   - **Rotate `x0` and `x1` simultaneously** → exact. A frame change leaves every
     `‖x0[a] − x1[b]‖²` unchanged, so the Hungarian picks the same permutation and Kabsch is
     equivariant. Verified 7.1e-15. **This is the test that catches wrong-side and transpose bugs** —
     it is what the original plan was reaching for.
   - **Rotate `x1` alone** → *assert nothing about equality*. Optionally record the gap as a
     diagnostic of approximation quality; do not gate on it.
5. **SO(3) vs O(3) — isolate the Kabsch step.** Same trap: through the *full* pipeline, reflecting
   `x1` changes the Hungarian's permutation, so the O(3) branch does not recover the unreflected
   cost and the test is meaningless. Hold the permutation at identity and test Kabsch alone. With
   `x1 = R·mirror(x0)`: `d = sign(det H)` must give a strictly positive cost (the mirror is
   unreachable in SO(3)), `d = +1` must give ~0 (the mirror is free in O(3)). Verified: 11.64 vs
   0.0000. If both branches agree, the sign correction is dead code.
6. **Cost reduction — the 2×2 ablation.** See "Attributing the win" below. Klein's whole claim is
   that `align + batch` drops substantially below `batch` alone at small B. The end-to-end smoke
   test.
7. **Marginal preservation.** The LJ13 twin of Test C6, and it is **not** redundant with it — see
   the note below on what Test 4 implies. Couple `x0 ~ p₀` against real MCMC `x1`, then check the
   aligned noise still looks like the prior. Statistic must be **non-G-invariant**, or it is
   preserved trivially and proves nothing: use a raw coordinate, `x0_aligned[:, 0, 0]`, which is
   `N(0, 12/13)` under the COM-free prior (`Var(xᵢ − x̄) = 1 − 1/13`). Check mean, variance, and a KS
   test; also check exchangeability across particles (`Cov(xᵢ, xⱼ) = −1/13` for `i ≠ j`).

### ⚠️ What Test 4 implies for the marginal — an open question, not a settled one

The marginal-preservation argument at the top of this plan needs the alignment map to be
**equivariant**: `g*(ρ(h)x0, x1) = g*(x0,x1)h⁻¹`, so that `T(ρ(h)x0) = T(x0)` and `T` collapses each
orbit onto one representative, which averaging over a G-invariant `p₁` then restores.

Test 4 shows Klein's sequential approximation is **not exactly equivariant** under rotations — the
Hungarian runs on unrotated coordinates. So for LJ13 the orbit-collapse is approximate, and the
marginal-preservation argument inherits that approximation. This is a property of the **published
method**, not of our implementation, and Klein's results suggest the perturbation is small in
practice. But "suggests" is not "measures", and Test 7 is cheap.

Note the asymmetry with Part C, and that it runs the *other* way from what you'd expect:

| | alignment | exactly equivariant? | marginal argument |
|---|---|---|---|
| LJ13 | sequential **approximation** to `S(13) ⋊ SO(3)` | **no** (Test 4) | approximate |
| XY | **exact** enumeration of `Z₂ ⋉ U(1)` | **yes** | exact |

The small-group system has the rigorous guarantee; the big-group system does not. If Test 7 shows a
measurable marginal shift on LJ13, that is a **finding worth reporting**, not a bug to fix — and it
would make the XY path the theoretically cleaner of the two despite buying less transport cost.

### Phase B4 — Training loop

Linear interpolant, `t=0` → prior, `t=1` → data (HollowFlow's convention, and it matches the
checkpoints you'll compare against):

```
x0 = sample_prior(B)                       # mean-zero isotropic Gaussian on the 36-dim subspace
x0, x1 = equivariant_ot_couple(x0, x1)     # Phase B2, under no_grad
mu_t   = x0*(1-t) + x1*t
x      = mu_t + sigma * sample_prior(B)    # HollowFlow uses sigma = 0.01
ut     = x1 - x0
loss   = ((v_theta(t, x) - ut)**2).mean()
```

- `v_theta` is the EGNN, output projected mean-free.
- Assert mean-free after coupling *and* after interpolation.
- The prior lives on the `(13−1)×3 = 36`-dim mean-zero subspace, not 39. Doesn't affect training
  loss but bites later if you compute likelihoods.

### Phase B5 — Benchmark ✅ done

`experiments/bench_ot.py` — three sections (`step`, `breakdown`, `overlap`). Measured on an RTX PRO
2000 Blackwell, fp32, real OSF MCMC `x1`.

The benchmark found a **20× constant** in the cost path and it has been fixed (`ot._svdvals_3x3`);
the numbers below are post-fix, with the pre-fix column kept for the record.

**The coupling still dominates the step at large B, but no longer at the sizes that matter.**

| B | couple ms | *(was)* | EGNN fwd+bwd ms | coupling % of step | *(was)* |
|---|---|---|---|---|---|
| 64 | 2.1 | *5.6* | 5.8 | 26.9% | *44.3%* |
| 128 | 3.6 | *14.9* | 6.8 | 34.5% | *70.7%* |
| 256 | 11.0 | *58.1* | 6.3 | 63.7% | *89.3%* |
| 512 | 42.9 | *227.9* | 9.4 | 82.1% | *96.1%* |

The EGNN is flat in B (13 particles — the GPU is idle at any of these sizes); the coupling is the
only thing that scales. **Two of this plan's assumptions about *where* that time goes were wrong:**

| B | D build | hungarian | gather | eigvals | det | **outer LSA** | *svdvals (rejected)* |
|---|---|---|---|---|---|---|---|
| 64 | 0.09 | 0.27 | 0.02 | 0.39 | 0.15 | 0.27 | *3.3* |
| 128 | 0.26 | 0.78 | 0.02 | 0.74 | 0.16 | 1.06 | *12.4* |
| 256 | 1.55 | 3.98 | 0.03 | 2.29 | 0.11 | 6.03 | *48.8* |
| 512 | 6.18 | 17.35 | 0.28 | 9.34 | 0.25 | **38.85** | *193.4* |

1. **It was not the Hungarian — it was `svdvals`, at 84–90% of the coupling.** ✅ **fixed.** B2 Step 5
   anticipated cuSOLVER's per-matrix overhead at 3×3 and prescribed `svdvals` over full `svd` to
   dodge it. Not enough: batched `svdvals` on `(B,B,3,3)` is slower **on GPU than on CPU** (51 ms vs
   33 ms at B=256). The fix keeps the `svdvals` *trick* and drops the *call* — singular values of `H`
   are `sqrt(eigvals(HᵀH))`, swapping cuSOLVER's SVD for a batched symmetric eigensolve: **~20×
   faster** (48.8 → 2.29 ms at B=256), which is 5× off the whole coupling. Accuracy is a non-issue
   despite squaring the condition number: against an fp64 `svdvals` reference on real data,
   `eigvalsh` errs 5.44e-05 vs `svdvals`' own 5.35e-05 in fp32 (1.2e-13 vs 1.1e-13 in fp64), and the
   outer assignment is **identical**. LJ13's `H` is well conditioned — the point clouds are not
   degenerate. All 111 tests pass unchanged, including B3's oracle equivalence.
2. **The outer LSA is not "~1 ms".** B2 Step 6 called it "one scipy call, ~1 ms, not worth moving to
   GPU". True at B≤128; it is 6.0 ms at B=256 and **38.9 ms at B=512** — now the **largest single
   cost** in the coupling, having inherited the crown from `svdvals`. Revisit only if B>256 ever
   matters; it should not, since equivariant OT is a small-batch method. **Left as-is.**

**Inline wins — but only because of finding 1.** Inline coupling vs SemlaFlow's pattern (couple on
CPU in dataloader workers, overlapped with GPU compute), 8 workers, s/step:

| B | inline | *(was)* | 8 CPU workers | winner | *(was)* |
|---|---|---|---|---|---|
| 64 | 0.0093 | *0.0106* | 0.0107 | inline | *tie* |
| 128 | 0.0101 | *0.0249* | 0.0110 | inline | *workers 2.2×* |
| 256 | 0.0200 | *0.0683* | 0.0315 | **inline 1.6×** | *workers 2.0×* |
| 512 | 0.0542 | *0.2403* | 0.1244 | **inline 2.3×** | *workers 1.8×* |

So "keep whichever wins" says **inline** — no dataloader plumbing, no worker RAM, single process.
**Note the verdict inverted.** Pre-fix, the workers arm beat the GPU path ~2× and the honest reading
was "adopt SemlaFlow's architecture"; that was never an architectural fact, only 8 CPUs in parallel
out-running one GPU doing `B²` 3×3 SVDs badly. Landing a 20× constant reversed it. Worth remembering
before the next such comparison: **profile before you architect.**

---

## Part C — XY chain OT (`Z₂ ⋉ U(1)`)

Same shape as Part B — align the noise, fill `M`, run minibatch OT on `M` — with two differences
that make it *simpler*, not harder:

- **No Hungarian.** The group is `Z₂ ⋉ U(1)`: two reflections, each with a closed-form optimal
  rotation. It is enumerated **exactly**, not approximated. Klein's sequential Hungarian→Kabsch
  approximation exists because `S(N) ⋊ SO(3)` is too big to enumerate; `Z₂ ⋉ U(1)` is not.
- **No SVD.** The `U(1)` alignment is the circular mean, which has a closed form — the `S¹` analogue
  of Kabsch.

### The group

| Element | Action on `θ ∈ (−π,π]^L` | Symmetry of `E = J Σᵢ cos(θᵢ₊₁−θᵢ)`? |
|---|---|---|
| global rotation `φ ∈ U(1)` | `θᵢ → θᵢ + φ` | ✓ energy depends only on differences |
| reflection `r ∈ Z₂` | `θᵢ → θ_{L−1−i}` | ✓ `θ'ᵢ₊₁−θ'ᵢ = −(θ_{L−1−i}−θ_{L−2−i})`, `cos` even |

Both verified numerically against `classicalXY.py::energy`. Both are respected by `XYChainGNN`: it
consumes only wrapped differences (⇒ `U(1)`-invariant, and its output is a tangent field ⇒
`U(1)`-equivariant), and reversal is an automorphism of the open-chain graph with its `inv_dist`
weights (⇒ reflection-equivariant, as for any GNN on a static graph). Prior and target are both
invariant under both — see the invariance table. This is the coherent Klein setup.

> **If the chain ever becomes periodic**, the group grows to `D_L ⋉ U(1)` — `L` cyclic translations
> × reflection, `2L` elements. Still exact enumeration, still no Hungarian; the `for r in (0,1)` loop
> below becomes `for r in range(2L)`. Worth structuring the code so this is a one-line change.

### The cost — closed form

Chordal (cosine) distance rather than wrapped-geodesic `Σ wrap(Δ)²`. **This is a deliberate choice
and the reason the rotation is closed-form.** Per pair, `1−cos Δ = 2 sin²(Δ/2)` is a monotone
function of `|wrap(Δ)|` on `[0,π]`, so the two costs agree on *per-pair* ordering; they differ on
sums, and only the chordal one admits a closed-form minimizer. Note this in the docstring — it is a
divergence from the geodesic cost `xyEESI._interpolant_sample` uses for the interpolant itself.

For `d = x1 − ρ(r)x0` and `S = Σᵢ sin dᵢ`, `C = Σᵢ cos dᵢ`:

```
min_φ Σᵢ (1 − cos(dᵢ − φ))  =  L − √(S² + C²)          φ* = atan2(S, C)
```

Verified against a 2×10⁶-point brute-force grid: agreement to 9 decimals. **The cost needs no
`atan2` at all** — only the `B` pairs the outer OT selects need `φ*` materialized. This is exactly
the Part B `svdvals` trick: singular values for the `B²` costs, full factorization only for the `B`
survivors.

### Vectorization — 8 GEMMs, no `(B,B,L)` tensor

Expand `sin(x1ⱼ − x0ₖ)` and `cos(x1ⱼ − x0ₖ)` to turn the reduction over `L` into matmuls:

```
S = sin(x1) @ cos(x0).T - cos(x1) @ sin(x0).T        # (B, B)
C = cos(x1) @ cos(x0).T + sin(x1) @ sin(x0).T        # (B, B)
M_r = L - torch.hypot(S, C)                          # (B, B) cost for reflection r
```

Four matmuls per reflection, eight total; `M = min(M_e, M_R)`, with `argmin` retained to know which
reflection each pair chose. Verified against the naive `(B,B,L)` construction: max abs error
3.55e-15. The `(B,B,L)` tensor never exists — which matters, since `L` here (200 in `classicalXY`'s
default) is an order of magnitude past LJ13's 13.

Then: `linear_sum_assignment` on `M` (identical to Phase B6), and materialize the `B` winners —
apply the chosen reflection, compute `φ*`, shift. Wrap the result onto `(−π,π]` for cleanliness;
`xyEESI` is wrap-invariant so this is cosmetic, not load-bearing.

### Phase C1 — Oracle

`eesi/ot_reference.py::xy_ot_map` — the same `B²` Python double loop, with `min_φ` found by a **dense
grid search over φ** rather than the closed form. Deliberately not the formula under test: a closed
form validated against itself proves nothing. Grid of 10⁴ points gives ~1e-7 agreement, ample.

### Phase C2 — GPU path

`eesi/ot.py::xy_ot_couple(x0, x1, reflect=True)`. `reflect=False` restricts to `U(1)` only — the
ablation, and the thing to compare against.

### Phase C3 — Validation (`tests/test_ot.py::TestXY`)

1. **Oracle equivalence.** B=32 vs `xy_ot_map`. Compare total transport cost, not assignments.
2. **Cost floor.** `c̃(x,x) == 0`.
3. **Cost monotonicity.** `c̃(x0,x1) ≤ Σᵢ(1−cos(x0ᵢ−x1ᵢ))`.
4. **Invariance.** Apply a random global rotation and/or a reversal to `x1`; `c̃` unchanged to ~1e-6.
   With `reflect=False`, the reversal case **must** change the cost — otherwise the `Z₂` branch is
   dead code (the Part B Test 5 pattern).
5. **Closed form vs grid.** `L − √(S²+C²)` against a dense `φ` grid, on random pairs. Already
   verified by hand; make it a regression test.
6. **Marginal preservation.** ⚠️ **The gate that makes Part C legitimate.** Draw `x0 ~ p₀`,
   `x1 ~ p₁` (real MC data from `mcxy`), align the noise, and confirm the aligned `x0` is still
   distributed as the prior. Test statistic: `⟨cos(θᵢ₊₁ − θᵢ)⟩`, which is `0` for the i.i.d. uniform
   prior and strongly positive for `J=2` chain data. If chain structure leaks into the aligned noise,
   this catches it.
   Run the **same test with an `S(L)` Hungarian step added** and confirm it *fails* — that is the
   experiment behind the invariance table, and it documents why `S(L)` isn't in the group.
7. **Cost reduction — the 2×2 ablation.** As Part B Test 6, and the place the "where does the win
   come from" question gets answered. See "Attributing the win" below. Expect the `align` axis to
   contribute little — the group is `Z₂ ⋉ U(1)`, not `S(13) ⋊ SO(3)`, so there is far less symmetry
   to exploit. **A modest alignment gain here is the honest expected result, not a failure.**

### Phase C4 — Training

The coupling is **orthogonal to the model**: `EESI.loss` already takes both endpoints, so wiring it
in is one line and needs no API change.

```
x0 = sample_base(B, N)
x0, x1 = xy_ot_couple(x0, x1)          # no_grad
losses = model.loss(x1, x0)
```

### ⚠️ Entropy is NOT a gate. It is the experiment.

An earlier draft of this plan proposed gating C4 on `ΔS` matching the exact Bessel value
`ΔS/L = (N−1)/N · (−J·I₁(J)/I₀(J))`. **Do not.** It conflates two independent failures:

- `ΔS` is estimated *through the trained networks* (`entropy_estimate_div` / `entropy_estimate_dot`),
  so a model that trains poorly misses the Bessel value for reasons having nothing to do with the
  coupling. Trainability degrades at low temperature (large `J`) exactly where the interesting
  physics is, so the gate would fire hardest where it is least diagnostic.
- The comparison it was reaching for — "did the coupling perturb the marginal?" — is already
  answered **model-free** by Test C6, which measures `⟨cos Δθ⟩` on the coupling's output with no
  network involved. Entropy is a strictly noisier proxy for a check we already have.

Entropy is the *point of the project*, not a unit test for a data-pairing step. Run it in a notebook,
against a `J` chosen for trainability, and compare coupled vs uncoupled as a **result**.

> Note the tension when picking `J`: it sets how far `p₁` is from `p₀`. At `J → 0` the chain
> decouples to i.i.d. uniform — which *is* the prior — so `ΔS → 0` and the flow becomes trivial.
> Shallower helps trainability but shrinks the signal being measured. There is a floor.

---

## Attributing the win — the 2×2 ablation (Tests B6 / C7)

The coupling has **two independent layers**, and they are easy to conflate:

| Layer | What it optimizes | Where |
|---|---|---|
| **align** — OT over the group | per-pair `min_{g∈G}`, filling entry `M[i,j]` | B2 steps 3–5 / C's closed form |
| **batch** — OT over the minibatch | `argmin` over assignments of `M` — which `x0_i` goes with which `x1_j` | B2 step 6, shared by both systems |

Both are already in the plan; `batch` is Phase B6 and Part C reuses it verbatim. But the obvious
comparison — "equivariant OT vs naive minibatch OT" — **cannot see the batch layer at all**, because
both arms include it. It measures `align` and silently credits it with everything.

So measure the 2×2 instead. Same batch, same seed, mean transport cost under the *true* symmetry-aware
cost `c̃` (so the arms are comparable — scoring an un-aligned pairing with the naive cost measures
nothing):

| | no batch OT (identity pairing) | batch OT |
|---|---|---|
| **no align** | baseline — random pairing | plain minibatch OT |
| **align** | group alignment alone | full equivariant OT |

Read it as: row effect = what the symmetry buys, column effect = what the batch OT buys, and the
interaction = whether they compose or overlap. **The interaction is the interesting cell.** The two
layers are not obviously additive — aligning every pair compresses the spread of `M`, which can
leave the batch OT less to choose between. If `align + batch ≈ max(align, batch)` rather than
their sum, they are substitutes, not complements, and the cheaper one wins on a cost/benefit basis.

### ✅ Measured — the prediction held

Real data both sides (LJ13: OSF MCMC; XY: `mcxy`, L=32, J=2). Cost as % change vs random pairing;
LJ13 `‖x0−x1‖²`, XY chordal.

| | LJ13 B=32 | LJ13 B=256 | XY B=32 | XY B=256 |
|---|---|---|---|---|
| `batch` alone | −29.0% | −39.4% | −21.8% | −31.6% |
| `align` alone | **−74.6%** | **−74.4%** | −19.7% | −20.2% |
| both | −80.9% | −83.0% | −33.9% | −40.6% |
| **verdict** | **align dominates 2.6×** | align dominates | **batch dominates** | **batch dominates 1.6×** |

Exactly as predicted: `align` dominates LJ13 (`S(13) ⋊ SO(3)` is enormous — `13! ≈ 6.2e9` — so
minibatch OT never stumbles onto a matching permutation, which is Klein's thesis), and `batch`
dominates XY (`Z₂ ⋉ U(1)` is tiny). **Most of the XY benefit is ordinary minibatch OT wearing a
symmetry-aware cost.** That is a useful negative: the LJ13 result does not transfer, and not for the
reason one might assume.

They are **substitutes, not complements**, on both systems — the interaction cell is well short of
additive (XY B=256: additive would be 15.41, actual 19.01). Aligning every pair compresses the
spread of `M`, leaving the batch layer less to choose between.

> ⚠️ **Correction to an earlier claim in this plan.** It said "both layers should weaken as B grows."
> False, and the data is unambiguous: **`align` is flat in B** (−74.6→−74.4, −19.7→−20.2 — it is a
> per-pair quantity and cannot depend on B), while **`batch` strengthens with B** (−29.0→−39.4,
> −21.8→−31.6 — more candidates to choose among). What makes equivariant OT a small-batch method is
> not that alignment decays, but that batch OT *catches up*: align's **relative** advantage shrinks
> as B grows. For XY, batch has already overtaken align by B=32.

## Decisions to fix up front (write them into module docstrings)

| Choice | LJ13 | XY | Why |
|---|---|---|---|
| Group | `S(13) ⋊ SO(3)` | `Z₂ ⋉ U(1)` | Klein Eq. 16 / the invariance condition. `S(L)` inadmissible for XY. |
| Transform which side | noise `x0` | noise `x0` | SemlaFlow's convention; valid because both marginals are G-invariant. |
| Approximation | sequential Hungarian → Kabsch, once | **exact** — group is enumerable | Klein §5; A.9 says iterating doesn't help. XY needs no approximation. |
| Distance | Euclidean `‖·‖²` | chordal `1−cos Δ` | Chordal gives the closed-form rotation; per-pair order-equivalent to geodesic. |
| Cost trick | `svdvals`, full SVD only for winners | `L−√(S²+C²)`, `atan2` only for winners | Same idea, both systems. |
| Gradients | none, `no_grad` | none, `no_grad` | Coupling is data pairing, not part of the objective. |
| Reduction | `sum` or `mean`, consistently | `sum` | Uniform rescale at fixed `N`; matters only against published numbers. |

## Order of operations

```
Part A  ──►  Part B1 ──► B2 ──► B3 ──► B4 ──► B5
   │                        (B3 gates B4)
   └──────►  Part C1 ──► C2 ──► C3 ──► C4
                             (C3 gates C4)
```

Part A first: it is a pure move with an exact gate, and both other parts import from it. Parts B and
C are independent after that — C does not depend on B, though B1/B2 are worth reading first since C
mirrors their structure.

Within each part: **no oracle, no way to know the GPU path is right.** Do not tune any network until
the cost-reduction test (B6 / C7) shows equivariant OT beating naive OT — otherwise you'll debug the
network when the bug is in the coupling.

The failure mode of Part C specifically is a coupling that lowers transport cost by quietly
destroying the prior marginal. Test C6 is the one that catches it; it is not optional.
