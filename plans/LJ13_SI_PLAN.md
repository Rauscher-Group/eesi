# LJ13 Stochastic-Interpolant Model — Implementation Plan

**Context.** `eesi.interpolant.EESI` implements the general Euclidean stochastic interpolant

    x_t = alpha(t) x0 + beta(t) x1 + gamma(t) z,   z ~ N(0, Id)

with a latent `z` whose antithetic denoising score (`gamma != "none"`) is exact because,
conditional on the pair `(x0, x1)`, `x_t | (x0,x1) ~ N(alpha x0 + beta x1, gamma^2 Id)` and
`s_target = -z / gamma`. `xyEESI` specialises the *geometry* (periodic angle manifold) by
overriding the single hook `_interpolant_sample`. There is no LJ13 counterpart yet: LJ13
training (`eesi.train.lj13`) is plain flow matching with a fixed `sigma` smoothing, not a
proper stochastic interpolant, and has no score network or interpolant-based entropy.

**Goal.** A new `LJ13EESI(EESI)` in `eesi/interpolant.py` — the LJ13 analogue of `xyEESI` —
for point clouds of shape `[B, N, 3]` (N-agnostic: 13 today, larger clusters later).

## The physics

The one correctness requirement is that **every Gaussian draw lives on the mean-zero
(COM-free) subspace** `V = {x : sum_i x_i = 0}`, so the whole path stays on the space where
`p0` (COM-free Gaussian prior) and `p1` (LJ13 Boltzmann) live. An isotropic Gaussian is
already SO(3)-invariant with or without centering; centering is what enforces the
*translation* quotient and keeps the path on `V` (see the `eesi.ot` module docstring and the
`center` discussion). Because `x0` and `z` are then the same kind of object — centered
Gaussians in the *flat* subspace `V` — the two-sided interpolant is equivalent to a
one-sided one, which is exactly why this is simple: no tangent-space / exp-map machinery is
needed (contrast the general Riemannian case).

**Keep the latent independent.** The one-sided *equivalence* (folding `alpha x0 + gamma z`
into one Gaussian) holds only for an *uncoupled* base. With the equivariant-OT coupling `x0`
is aligned to `x1` and is no longer free noise given `x1`, so `z` must remain an
**independent** centered latent for the denoising score to stay exact. This model therefore
keeps `z`; it is not plain FM-with-fixed-sigma.

## What breaks in stock `EESI` for `[B, N, 3]`, and the fix

Stock `EESI` assumes flat `[B, d]` with uncentered Gaussians. Three issues:

1. **Uncentered noise draws (core).** `torch.randn_like` in `loss` (`z`),
   `entropy_estimate_div` / `entropy_estimate_dot` (`z`), the SDE diffusion term in all four
   `sample_sde*` methods, and the Hutchinson probes in `_div_hutchinson`. Each must be
   COM-free for LJ13.
   → **Fix:** add a `_noise_like(ref)` hook to `EESI` (default `torch.randn_like`) and route
   *all* raw-Gaussian draws through it. `LJ13EESI` overrides it to center over the particle
   axis: `g - g.mean(dim=-2, keepdim=True)`. Single-hook parallel of `xyEESI`.

2. **Time broadcasting.** `loss`/entropy build `t` as `[B,1]`, which mis-broadcasts against
   `[B,N,3]`.
   → **Fix:** a `_draw_time(x)` helper building `t` broadcastable to `x`
   (`shape = (B,) + (1,)*(x.dim()-1)`) plus `t_b = t.reshape(B)`. Backward-compatible with
   `[B,d]`.

3. **Feature reductions.** `_denoising_loss`, `score_loss`, `_div_hutchinson`, and the
   entropy dots use `.sum(dim=-1)`, which sums only the 3-axis for `[B,N,3]`.
   → **Fix:** reduce over all non-batch dims via `.flatten(1).sum(-1)`. Identical to the
   current path for `[B,d]`; `_div_exact` already `reshape(B,-1)`s and needs no change.

These are backward-compatible generalisations of the shared base (verified against the
`[B,d]` reductions and `test_loss_matches_antithetic_formula`, which replays `t=[B,1]` /
`dim=-1`). `_interpolant_sample` needs **no** LJ13 override — the Euclidean formula is
already correct on the flat subspace once `z` is centered and `t` broadcasts.

## Divergence / entropy correctness

`entropy_estimate_div` traces `net_b` by Hutchinson. With **centered** probes `v ~ N(0, P)`
and a mean-free field, `E[v^T J v] = tr(PJ)` = the subspace divergence — the estimator
analogue of `eesi.models.lj13_dynamics.divergence`. So the centered-probe hook makes entropy
correct with no change to `_div_*` internals. Cross-check `entropy_estimate_div` against the
exact `lj13_dynamics.divergence` on a fixed batch; assert `DOF = (N-1)*3` bookkeeping.

## Training wiring

Mirror `eesi.train.xy`: `make_si_model(...)` (two `LJ13Dynamics` nets, one drift one score),
`si_step(model, x1, align, batch)` that samples COM-free `x0` via `sample_prior`,
`equivariant_ot_couple`s it, and returns `model.loss(x1, x0)`, and a `train_si` loop. Keep
the existing plain-FM `train`/`main` intact; expose the SI path behind a flag or sibling
entry. OT coupling stays on.

## Tests (`tests/test_lj13_eesi.py`, mirroring `test_xy_eesi.py`)

- COM-free preservation of `x_t`, `b_target`, `s_target`; `s_target == -z/gamma` and centered.
- Endpoints recovered at `t=0, 1`; antithetic `±z` sign relations.
- Loss finite + backprops into both nets across `path`/`gamma`.
- `entropy_estimate_div` matches the exact subspace `lj13_dynamics.divergence` on a batch.
- Base-class parity: the rank-agnostic reductions leave `[B,d]` results unchanged.

## Alternative considered (rejected as default)

Run stock `EESI` in 36-dim orthonormal subspace coordinates (`subspace_dirs`): cleanest
mathematically (prior is genuine `N(0,I_36)`, `z` centered by construction, divergence a
plain 36-trace, zero base-class change), but forces a map back to `[N,3]` for every
energy / OT / plot call and diverges from all existing `lj13_dynamics` tooling and the
checkpoint conventions. Physical `[B,N,3]` chosen instead.
