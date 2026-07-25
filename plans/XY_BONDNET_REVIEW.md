# XY chain: second architecture review — why 16 k parameters, and what replaces them

*Follow-up to `plans/XY_MODEL_REVIEW.md` and `plans/XY_FIX_PLAN.md`, prompted by
"still 16 k parameters for underwhelming performance". Context has changed since those
documents: the notebook now runs **plain flow matching** (`path="linear"`,
`gamma="none"`, `gamma_scale=0.0`, `learn_score=False`), so only `net_b` is trained.
No repository code was modified for this review. Date: 2026-07-24.*

Prototype: `plans/xybond_prototype.py`. Benchmark harnesses were written in the session
scratchpad and are not in the repo; every number below is reproducible from the configs
stated in §8.

---

## 0. Summary

| # | Finding | Severity |
|---|---------|----------|
| **G1** | **The 0.39 loss plateau is the objective's noise floor, not model error.** The flow-matching minimum is `E[Var(d｜x_t)] > 0`, set by the OT coupling. It is not comparable across batch sizes and moves ~1 % where sample error moves 2×. | **Critical** — it invalidates the metric the architecture was being judged by |
| **G2** | The parameter count is genuinely ~4.5× too high, because the model works in **site** coordinates. A bond-space net with **3 724** parameters matches `XYChainGNN`'s 16 688 at 63 % of the wall-clock. | **High** |
| **G3** | **Review §7.1 as specified has a hard ceiling** (0.4555 vs the GNN's 0.377) at *every* size. Its odd content is an input feature, not a propagated stream. | **Corrects `XY_MODEL_REVIEW.md` §7.1** |
| **G4** | The zero-sum drift is **exact**, not an approximation. Freezing the U(1) zero mode is free accuracy for `net_b`, not only for `net_s`. | **Corrects `XY_FIX_PLAN.md` Phase 4** |
| **G5** | `entropy="dot"` in notebook cells 15 and 19 reads an **untrained** `net_s` under the current `learn_score=False` config. | **Bug** |
| **G6** | `_outer_assignment` forces a GPU→CPU sync every training step, while the CUDA path next to it is installed and unused. | Low, rising as the net shrinks |
| **G7** | The residual ~2.5–3 % bias in `<cos Δθ>` is a **floor of the method**, shared by every architecture and size tried. Integration, EMA, data volume, coupling bias and training length past ~12 k steps were each tested and ruled out. | Informational — stop spending parameters on it |

Recommendation: adopt the metric change in §1 first (it is free and it is the reason the
architecture looked worse than it is), then the bond net in §3. Treat §7 as the open
question.

---

## 1. G1 — the loss plateau is the floor of the objective

Under `gamma="none"`, `EESI.loss` computes `loss_b = (b - b_target).square().mean()`
with `b_target = d = min_image(x1 - x0)`. Its minimiser is `b* = E[d｜x_t]`, so

```
loss_b  =  E[ Var(d｜x_t) ]  +  E‖b - b*‖²
           \_____________/      \________/
            irreducible;         the only part
            set by the coupling  the model controls
```

The first term is not small here. Three independent measurements:

**(a) Capacity is saturated.** 12 k steps, `N=10, J=2, B=256`:

| model | params | final `loss_b` |
|---|---|---|
| bond net, small | 2 226 | 0.3806 * |
| bond net | 3 724 | 0.3701 |
| bond net, large | 53 832 | 0.3690 |
| `XYChainGNN` (notebook cfg) | 16 688 | 0.3727 |

\* 1 500-step figure; see §8 Run A.

15× the parameters buys 0.3 % of the loss.

**(b) The floor is a property of the coupling, not the model.** Same 3 724-parameter
net, same schedule, only the OT batch changed:

| `B` | `loss_b` | `<cos Δθ>` error, N=10 |
|---|---|---|
| 256 | 0.3701 | −2.55 % |
| 512 | **0.3188** | −2.39 % |

A 14 % drop in the loss with essentially unchanged sample quality. **Loss values are not
comparable across batch sizes**, because the minibatch-OT coupling gets tighter with `B`
and `Var(d｜x_t)` falls with it.

**(c) The useful signal is in the 4th significant figure.** Same net, longer schedule:

| steps | `loss_b` | `<cos Δθ>` error |
|---|---|---|
| 3 000 | 0.3766 | −5.60 % |
| 12 000 | 0.3720 | −2.57 % |

The loss moved 1.2 %; the sample error halved.

### The metric to use instead

`p₁` is exactly a product of independent von Mises bonds (`XY_MODEL_REVIEW.md` §1), so
generated configurations have analytic ground truth with no network in the loop:

```python
D  = wrap(x[:, 1:] - x[:, :-1])
m1 = cos(D).mean()          # exact  I1(J)/I0(J)   = 0.69777 at J=2
m2 = cos(2*D).mean()        # exact  I2(J)/I0(J)   = 0.30223 at J=2
corr(cos D_i, cos D_{i+1})  # exact  0   (the bonds are independent)
```

Two seconds to compute on 20 k samples, and it is not `ΔU`/`ΔS` — it does not run
through a trained score, it is not the scientific result of the project, and it is
therefore a legitimate acceptance criterion under the validation rule in
`XY_FIX_PLAN.md`. The bond correlation is the sharpest of the three: it is exactly zero
in the target, and it is where the two architectures differ most.

---

## 2. G2 — why 16 k parameters buy so little

`XYChainGNN` operates on sites and recovers rotation invariance through the network:
each layer's MLP runs over `E = 2·n_neighbors·B·N` edges, and rotation invariance is
paid for by only ever consuming `Δθ = wrap(θ_dst − θ_src)`. But
`Δ_i = wrap(θ_{i+1} − θ_i)` is a *complete and non-redundant* set of U(1) invariants
(`T^N/U(1)` has dimension `N−1`), so rotation invariance is a change of variables, done
once, in line one — not an architectural cost.

With no node features `h`, each layer's per-edge scalar is a function of
`(1/k, cos(kΔθ), t)` alone — three effective inputs — evaluated by a 4-hidden-layer
width-32 MLP at every one of the ~15 000 edges per batch. That is where the parameters
and the wall-clock go.

---

## 3. The proposal — a parity-tracking bond net

Full source in `plans/xybond_prototype.py`. The structure:

```
Delta_i = wrap(theta_{i+1} - theta_i)                    [B, M],  M = N-1

e <- cos(k*Delta)  , k = 1..K       EVEN stream, palindromic convs WITH bias
o <- sin(k*Delta)  , k = 1..K       ODD  stream, palindromic convs WITHOUT bias

per block:
    e <- e + conv_e( act( FiLM_t([ e , o * conv(o) ]) ) )   # odd*odd -> even
    o <- o + conv_o(o) * gate(e)                            # even*odd -> odd

f   <- sum_c  o_c * head(act(e))_c                          [B, M]   ODD
v_j <- f_j - f_{j-1}      (f_0 = f_M+1 = 0)                 [B, N]   ODD, zero-sum
```

The parity algebra is the whole design: `even*even -> even`, `odd*odd -> even`,
`even*odd -> odd`, and any **bias-free** linear map preserves parity. Biases are
therefore allowed on the even stream and forbidden on the odd one (0 must be the only
fixed point).

**Properties, all exact and all verified numerically at `N = 3, 10, 11, 17`:**

| property | mechanism | measured residual |
|---|---|---|
| rotation invariance `θ → θ + φ` | only wrapped bonds enter | 3e-14 |
| oddness `θ → −θ` | parity algebra above | **exactly 0** |
| site reversal `θ_i → θ_{N+1−i}` | palindromic kernels + symmetric zero padding | 2e-14 |
| `2π` shift invariance | `wrap` on the bonds | 5e-13 |
| `Σ_j v_j = 0` | divergence form `f_j − f_{j−1}` | 4e-15 |
| chain-length agnostic | `N` read per forward, no graph | — |

It also **contains the analytic answer**: at `K=1`, one head channel of weight `J` and
no conv blocks, `v_j = J(sin Δ_j − sin Δ_{j−1}) = ∇_j log p₁`.

### Measured, 12 k steps, `N=10, J=2, B=256`, 50 k exact samples, identical schedule

| model | params | wall | `loss_b` | N=10 `<cosΔ>` | N=10 bond corr | N=64 `<cosΔ>` |
|---|---|---|---|---|---|---|
| `XYChainGNN` (notebook cfg) | 16 688 | 161 s | 0.3727 | −2.89 % | +0.058 | −1.59 % |
| **`XYParityNet` K4 C16 Co8 ×2** | **3 724** | **102 s** | 0.3701 | −2.55 % | **+0.038** | −3.21 % |

*(exact: `<cosΔ> = 0.69777`, bond corr = 0)*

**4.5× fewer parameters, 1.6× faster per step, equal or better.** The bond-correlation
column is the honest discriminator — 0 is the exact answer and the bond net is ~35 %
closer to it. On `<cos Δθ>` the two are within the seed spread (±0.3 %, measured over
two seeds). At longer chains `XYChainGNN` is better on `<cosΔ>` and worse on `<cos2Δ>`;
neither dominates.

The win is therefore **cost, not accuracy** — which is exactly what was asked for, and
per §1 accuracy was never the thing the loss was reporting.

### Sizing

`K=4, C=16, Co=8, n_blocks=2` (dilations 1, 2) is the recommended default. `K=3, C=12,
Co=6` gives 2 226 parameters within 0.3 % of the loss. Do **not** scale up: see §6.

---

## 4. G3 — correction to `XY_MODEL_REVIEW.md` §7.1

The bond CNN as specified there has an output head

```
w = Conv1d(C -> 1)(H)      # EVEN
f = w * sin(Delta)         # ODD
```

so the odd dependence of `v_j` reaches **only bonds `j` and `j−1`**. Everything else in
the network is even and cannot carry sign information. This is a hard ceiling, not a
capacity limit — three sizes spanning 3.7× in parameters plateau at the *same* value:

| variant | params | final `loss_b` (1 500 steps) |
|---|---|---|
| §7.1 bond CNN, K=6 C=16 dil 1/2/4 | 3 828 | 0.4558 |
| §7.1 bond CNN, K=4 C=8 dil 1/2 | 1 031 | 0.4554 |
| §7.2 bond MLP, R=2 (no depth) | 2 260 | 0.4558 |
| `XYChainGNN` | 16 688 | 0.3768 |
| **parity net (odd stream)** | 3 724 | **0.3793** |

Identical to four significant figures across the three — the signature of a structural
constraint. The fix is the odd *stream* in §3. §7.1's other claims (parity structure,
palindromic kernels, divergence form, analytic special case) all survive; only the head
is wrong.

An intermediate design was also tried and rejected: multi-lag features
`Θ^r_i = wrap(θ_{i+r} − θ_i)` with per-lag divergences. It reaches the right hypothesis
class but breaks site-reversal equivariance, because lag-`r` features have length `N−r`
and cannot be aligned on a common integer grid — recovering the symmetry needs a
half-integer token axis of length `2N−3`. Not worth it; the odd stream is cleaner.

---

## 5. G4 — correction to `XY_FIX_PLAN.md` Phase 4

Phase 4 was skipped partly on the reasoning that the projection is exact for `net_s` but
**not** for `net_b`, since "the drift must carry net rotation: `Σⱼ dⱼ` is the total twist
needed to carry a prior sample onto its partner".

The premise is right and the conclusion does not follow.

*Measured size of the term.* At `N=10, J=2, B=256` under the full coupling:

```
E|d|^2 per component            = 0.8689
of which the U(1) zero mode     = 0.0163   (1.88 %)
```

*Why zeroing it is exact.* Every `p_t` is U(1)-invariant, so writing `x = (φ, Δ)` with
`φ` the global phase, `p_t(φ, Δ) = Uniform(φ) × p_t(Δ)` **exactly**. The continuity
equation then factorises, and because `Uniform(φ)` is stationary under any φ-velocity
that does not depend on φ — which it cannot, by invariance — the φ-component of the
velocity is *unconstrained*. Zero is a legal choice. The regression target `d` does carry
a rotational component, but it is unlearnable noise in that direction: a zero-sum network
regresses onto the projection of `d`, which is the correct optimum, and the leftover
appears only as an additive constant in the loss.

Sampling is unaffected in distribution: the global phase is carried over unchanged from
`x0`, whose phase marginal is uniform and independent of its bonds — which is exactly
the target's phase marginal. Entropy accounting is likewise unaffected, since a
uniform→uniform direction contributes nothing to `∫div(b)dt`.

So the divergence form removes 1.9 % of the target's variance from the regression at no
cost in expressivity. It is built into §3 and is one of the reasons it trains at a
smaller size.

*The condition that would invalidate it* is unchanged from Phase 4: a coupling that
fixed a gauge, pinned a spin, or conditioned on the total angle would break the U(1)
invariance of `p_t` and turn the projection into a bias.

---

## 6. G7 — what the residual 2.5–3 % is, and what it is not

Every architecture and size converges to `<cos Δθ>` low by 2.4–3.2 % with a spurious
nearest-neighbour bond correlation of +0.03 to +0.06. Five candidate causes were tested
and **all are ruled out**:

| candidate | test | result |
|---|---|---|
| ODE discretisation | `n_steps` ∈ {20, 50, 100, 200, 500} × {heun, euler} | flat: −3.5 % to −4.1 %, no trend |
| under-training | 3 k → 12 k → 20 k steps | 12 k → 20 k is flat (−2.55 % → −2.39 %, seed spread ±0.3 %) |
| finite training data | 10 k → 200 k exact samples | no improvement (−2.57 % → −3.15 %) |
| SGD weight noise | EMA(0.999) on both architectures | within seed noise, sometimes worse |
| coupling marginal bias | `<cosΔ>`/`<cos2Δ>` of the **aligned base** `x0` | −1e-5 / −6e-4 — uniform to MC noise |

The coupling check is worth keeping: it directly verifies the marginal-preservation
argument in `eesi/ot.py`'s docstring for the *continuous* U(1) alignment, which
`tests/test_ot.py` covers only for the discrete group. All four ablation arms
(`align`/`batch`/`negate`/`reflect`) pass at the 1e-3 level.

Conclusion: this is a floor of the method, not of capacity. Adding parameters to chase it
is what produced the 16 k model.

**And it actively backfires.** The 53 832-parameter variant (4 blocks, dilations 1/2/4/8,
receptive field 31 bonds) is better at the training length and much worse away from it:

| model | N=10 `<cosΔ>` | N=32 `<cosΔ>` | N=32 `<cos2Δ>` |
|---|---|---|---|
| parity net, 3 724 | −4.43 % | −4.69 % | −3.04 % |
| parity net, 53 832 | −2.51 % | −10.8 % to −15.3 % | −30.5 % to −34.7 % |

The physical correlation length is `ξ = 1/ln(I₀/I₁) ≈ 2.8` sites at `J=2`. A receptive
field an order of magnitude past `ξ` has nothing left to model but the finite-size
structure of `N=10`. **Keep the receptive field near `ξ`**: 2 blocks, dilations (1, 2).

---

## 7. Open question — the acceptance criterion, and what is left on the table

§1 gives Phase 5 of `XY_FIX_PLAN.md` the criterion it was missing: model-free bond
moments and the bond correlation, against the analytic von Mises. That unblocks the
architecture work.

What it does **not** settle is whether 2.5 % is good enough, or where the last 2.5 %
lives. Two untested directions, in the order I would try them:

1. **The coupling, not the network.** `B=512` lowered the floor by 14 %; the trend says
   the minibatch-OT coupling is the dominant source of `Var(d｜x_t)`. Sweeping `B` (or
   using a better coupling) is the only lever measured so far that moves the floor
   itself. Cheap to test now that the net is 1.6× faster.
2. **A residual parametrisation anchored at the analytic score.** The net contains
   `J(sin Δ_j − sin Δ_{j−1})` exactly; writing `b = a(t)·(analytic) + residual` costs one
   parameter and starts training from the right endpoint. Untested.

Not recommended: more parameters, more steps, more data, EMA, finer integration — all
measured flat in §6.

---

## 8. Two fixes to make regardless of architecture

### 8.1 G5 — `entropy="dot"` reads an untrained network

Notebook cell 8 sets `learn_score=False`, so `EESI.loss` returns
`loss_s = torch.zeros_like(loss_b)` and `net_s` **never receives a gradient**. Verified
directly: after training, every `net_s` parameter is bit-identical to its initialisation,
and the network outputs ~4e-4 (the `xavier_uniform_(gain=1e-3)` head).

Cells 15 and 19 then call `entropy_estimate(..., "dot")` and `sample(..., entropy="dot")`,
both of which are `−∫b·s dt`. Under plain flow matching these must use **`entropy="div"`**,
which needs only `net_b`. If `experiments/xy_entr.npy` was produced from the current
notebook state rather than from the saved checkpoints, those numbers are invalid.

A guard in `EESI.entropy_estimate` / `EESI.sample` — raise if `method == "dot"` while
`learn_score` is False and `gamma == "none"` — would make this unmissable.

### 8.2 G6 — a device sync per training step

`eesi/ot.py:76`, `_outer_assignment`, calls `scipy.optimize.linear_sum_assignment` on
`M.detach().cpu().numpy()`, forcing a full GPU→CPU synchronisation every step. Its
docstring says "not worth moving to GPU", but `_hungarian_nd` directly below already has
the batched CUDA path, and `torch_linear_assignment` **is installed in this environment**
(`_HAS_BLA` is True). Routing `_outer_assignment` through it on CUDA is a few lines.

At ~1 ms of a 13 ms step this was minor; against the bond net's ~8 ms step it is not.

Related, and free: the notebook sets `torch.set_default_dtype(torch.float64)`. That was
justified when the ISM divergence was in the loss; under plain flow matching with
`learn_score=False` it no longer is. float32 is worth measuring.

---

## 9. Reproduction

Fixed benchmark, matching `XY_FIX_PLAN.md`'s: `N=10`, `J=2`, `B=256`, exact von-Mises-bond
sampler (`sample_p1_exact`), float64, CUDA, seed 0.

- **Run A** — 1 500 steps @ `lr=1e-3`, 10 k data. Architecture scan, §3/§4 loss tables.
- **Run B** — 2 000 @ 1e-3 + 1 000 @ 3e-4, 10 k data. First sample-quality pass, §6 table.
- **Run C** — integration sweep on the Run-B parity net, §6.
- **Run D** — convergence and data volume, §1(c) and §6.
- **Run E** — 6 000 @ 1e-3 + 3 000 @ 3e-4 + 3 000 @ 1e-4, 50 k data. **The headline table
  in §3**, plus the `B=512` arm in §1(b).
- **Run F** — Run E + EMA(0.999), two seeds. §6, and the ±0.3 % seed spread.
- **Run G** — 10 000 @ 1e-3 + 5 000 @ 3e-4 + 5 000 @ 1e-4, two seeds. Convergence check
  in §6 (flat vs Run E).
- **Run H** — coupling marginal check, 200 batches × 4 ablation arms. §6.

Sample quality is always 20 000 ODE samples, `n_steps=50`, Heun, evaluated at
`N ∈ {10, 32, 64}`.
