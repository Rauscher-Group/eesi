# XY chain: architecture review, symmetry audit, and redesign proposals

*Review of `eesi/models/xygnn.py`, `eesi/ot.py`, `eesi/interpolant.py::xyEESI`,
`eesi/train/xy.py` and `experiments/XY_chain_eqOT.ipynb`. No code was changed.
Date: 2026-07-24.*

---

## 0. Summary

Five findings, in descending order of how much I think they explain "huge model,
unimpressive results":

| # | Finding | Severity | Measured? |
|---|---------|----------|---|
| **F1** | The time embedding is **exactly periodic with period 1**, so `t=0` and `t=1` map to the *identical* input vector, while the true fields there differ completely. | **Critical** — a bug, not a style issue | **CONFIRMED**, §0.5 |
| **F2** | ~~No output preconditioning; the score target `-z/γ` forces outputs of ~5×10³.~~ | **WITHDRAWN** | **REFUTED**, §0.5 |
| **F3** | The target `p₁` is **exactly a product of independent von Mises bonds** — a one-parameter, strictly two-body object. A 22 656-parameter model is ~10²–10³× oversized for the structure that actually exists. | **Structural** | supported |
| **F4** | **Missing O(2) equivariance.** The model is not odd under `θ → -θ`, so it burns capacity learning a symmetry that can be imposed for free by dropping the `sin(kΔθ)` edge channels. The OT coupling likewise enumerates only 2 of the 4 discrete group elements. | **High** (and it is what you asked about in §3) | violation measured, §0.5 |
| **F5** | No node features `h`. This is the point Claude was making in (b), stated badly. The real content is below in §3. | **Moderate** | partially refuted, §0.5 |

Plus a set of smaller code-level issues in §5 (per-forward Python graph rebuild,
unbounded coordinate steps, a docstring/signature mismatch, length generalisation).

My recommendation: fix F1 (independent of architecture, helps whatever you build) and
F4, then decide on a rebuild. The ~2 000-parameter bond-CNN in §7.1 is
exactly-equivariant under the full group by construction and contains the analytic
answer as a one-parameter special case.

---

## 0.5 Diagnostic results (measured 2026-07-24)

Run against the trained notebook checkpoints (`experiments/xy_b_N10_B256.pth`,
`xy_s_N10_B256.pth`) and fresh short trainings. `N=10, J=2, B=256`, float64, CUDA.
Scripts are in the session scratchpad; nothing in the repo was modified.

### F1 — CONFIRMED, and it is the headline result

The time embedding collapses the two endpoints:

```
||emb(0)      - emb(1)||        = 1.7e-14      <- float64 zero
||emb(1e-6)   - emb(1-1e-6)||   = 4.9e-04
||emb(0.2)    - emb(0.8)||      = 5.8e+00      <- for scale; ||emb|| = 4.0
```

and therefore so does the trained network, evaluated on *identical* `x`:

```
net_b:  ||v(t=0) - v(t=1)|| / ||v|| = 1.5e-15     (bitwise identical)
net_s:  ||v(t=0) - v(t=1)|| / ||v|| = 1.1e-14
        control: ||v(t=0) - v(t=0.5)|| / ||v||  = 0.20 (net_b), 0.96 (net_s)
```

What that costs, against the analytic endpoints:

| | true `‖∇log p_t‖` | trained `‖net_s‖` |
|---|---|---|
| `t → 0` | **0** (`p_t` is exactly uniform) | 2.142 |
| `t → 1` | **4.904** (`= ‖∇log p₁‖`) | 2.143 |

The model is structurally incapable of separating the two, so it lands on a single
compromise value and is wrong at both ends. This is a clean, decisive confirmation.

### F2 — REFUTED. Withdrawn.

Two independent reasons, and I had it wrong:

1. **The loss is bounded.** With antithetic ±z the branches combine to
   `¼(‖s⁺‖²+‖s⁻‖²) + (1/2γ)(s⁺−s⁻)·z`, and `s⁺−s⁻ = 2γ(z·∇)s + O(γ³)` cancels the
   `1/γ` exactly, leaving `zᵀ∇s z → div s` (i.e. the objective converges to implicit
   score matching). Measured `loss_s` at `t=1e-6` is **+8.4**, not the `~-10⁸` a
   surviving `-½‖z/γ‖²` term would give. Bounded across the whole grid.
2. **The minimizer is bounded too** — this is the part I missed. The network regresses
   onto `E[-z/γ | x_t] = ∇log p_t`, not onto `-z/γ`. And `p_t → p₀ = uniform` as `t→0`
   (uniform base plus wrapped noise is still exactly uniform on the torus), so
   `∇log p_t → 0`, and at `t→1` it tends to `∇log p₁` with norm 4.90. **The true score
   never exceeds ~5 anywhere.** My `√N/γ(t)` reference was the *conditional* target
   scale, which is not what the net has to represent.

So there is no representation-range problem, no reason to precondition, and no reason
to raise `eps`. The `‖net_s‖/(√N/γ)` ratio I proposed as a "gate" was measuring against
the wrong quantity.

### F5 / depth (§3.3) — partially refuted

700 steps, `lr=1e-3`, identical seed/data/width, only `n_layers` varied:

| `n_layers` | params/net | wall-clock | `loss_b` | `loss_s` | `ΔU` err | `ΔS` err |
|---|---|---|---|---|---|---|
| 1 | 2 832 | 12.2 s | −2.280 | −3.988 | 20.2 % | 33.2 % |
| 2 | 5 664 | 21.9 s | −2.450 | −4.441 | **6.7 %** | **6.6 %** |
| 4 | 11 328 | 40.1 s | −2.469 | −4.411 | 5.0 % | 6.7 % |

My claim in §3.3 was that a 1-layer model would match the 4-layer one. **It does not** —
depth 1 is clearly insufficient, so the "depth is dormant at init" argument is wrong as
stated. What is true is weaker and still useful: **layers 3–4 buy nothing** (`ΔS` err
6.6 % → 6.7 %, `loss_s` slightly *worse*) for 2× the parameters and 2× the wall-clock.
The notebook's `n_layers=4` should be `n_layers=2`.

### F4 — violation measured

Oddness defect of the trained nets, `‖net(t,−x) + net(t,x)‖ / ‖net(t,x)‖`:

```
net_b: 0.044        net_s: 0.100
```

Small but real, and twice as large on the score — the field where the residual `ΔS`
error lives. Phase 2 drives both to machine zero by construction.

### Two bonus findings

- **`load_mc_data` is unnecessary.** §1's factorisation gives an *exact* `O(BN)`
  sampler — `θ₁ ~ U(-π,π]`, `Δᵢ ~ vonMises(0, J)` i.i.d., `θ = cumsum`. No
  equilibration, no autocorrelation, no 10⁶-step Python loop. All diagnostics above
  used it. This should replace `mcxy` for training data at any `N`, `J`.
- **Per-step timing is roughly linear in `n_layers`** (17 ms at L=1, 57 ms at L=4 ⟹
  ~13 ms/layer + ~4 ms fixed). So the per-forward graph rebuild is ~7–23 % of the step,
  not the dominant cost. Phase 0.1 is still worth doing but I overstated it in §5.1.

---

## 1. The problem is far easier than the model assumes

This frames everything else, so it goes first.

Take the open chain, `U(θ) = -J Σ_{i=1}^{N-1} cos(θ_{i+1} - θ_i)` (the energy in
`eesi/datasets/xy.py`). Change variables on the torus:

```
(θ_1, …, θ_N)  ->  (θ_1, Δ_1, …, Δ_{N-1}),    Δ_i = θ_{i+1} - θ_i  (mod 2π)
```

This is a bijection of `T^N` with unit Jacobian, and `U` depends only on the `Δ`. Hence

```
p₁(θ)  =  Uniform(θ_1)  ×  Π_{i=1}^{N-1} vonMises(Δ_i ; μ=0, κ=J)
```

**exactly**, with the bond variables mutually independent. This is the same fact that
gives you the closed forms you plot in the notebook:
`ΔS = (N-1)[ln I₀(J) - J I₁(J)/I₀(J)]`, `ΔU = -(N-1) J I₁(J)/I₀(J)`.

Consequences:

- The exact score of the target is **one parameter**:
  `∇_j log p₁ = J·[ sin Δ_j - sin Δ_{j-1} ]` (boundary terms dropped at `j=1, N`).
- The base `p₀` (i.i.d. uniform) has score exactly `0`, and its bonds are also i.i.d.
  uniform.
- The interpolant `x_t = wrap(x₀ + β(t)·d + γ(t)·z)` with `d = min_image(x₁ - x₀)`
  does *not* keep the bonds independent — `d_{i+1} - d_i` and `z_{i+1} - z_i` couple
  neighbours, and the minibatch/group OT adds weak long-range structure — but the
  induced correlations are short-ranged. The physical correlation length at `J=2` is
  `ξ = 1/ln(I₀/I₁) = 1/ln(1.4331) ≈ 2.8` sites.

So the true `b(t, ·)` and `s(t, ·)` are **short-ranged, nearly two-body, and anchored
at a known analytic endpoint**. A model with a window of ~5 bonds and a few hundred
parameters should be able to match them to well below the ~2–3 % error the notebook
currently reports:

| quantity (per spin, N=10, J=2) | exact | notebook | error |
|---|---|---|---|
| `ΔU/N` | −1.2560 | −1.23193 | 1.9 % |
| `−ΔS/N` | +0.5145 | +0.49887 | 3.0 % |

The results are not *wrong* — they are simply expensive. 22 656 parameters
(11 328 × 2 nets; `n_neighbors=4, hidden=24, mlp_layers=4, n_layers=4, edge_order=4,
time_order=16`) for a 9-degree-of-freedom problem with a one-parameter exact answer.

---

## 2. Point (a) — node features. Confirmed.

Yes. `XYChainGNN` has no `h` at all; the plan in `xygnn-plan` deliberately dropped
`node_mlp` and the per-node hidden state. That decision is the root of the problem in
(b), and it has three separate costs:

1. **Only one communication channel.** In the Satorras EGNN there are two state
   tensors: coordinates `x` and node features `h`. Information reaches distant nodes
   through `h`, which is unconstrained, 32-dimensional, residual, and *invisible to the
   output*. In `XYChainGNN` the only state is `θ`, a single scalar per node. See §3.
2. **No way to represent "where am I in the chain."** With sum aggregation and no `h`,
   an end site differs from a bulk site only by having fewer incident edges. That is a
   very weak signal, and it is the *only* one. A per-node feature (degree, or
   `[is_left_end, is_right_end]`, or a learned positional scalar) would make the
   boundary explicit. This also bears on the `N`-sweep in notebook cell 19 — see §5.4.
3. **Time has nowhere to live except the edge input.** In EGNN, `h` is initialised from
   `t`, updated per layer by `node_mlp`, and enters the edge MLP as `[h_row, h_col, …]`.
   So `t` becomes a *learned, layer-dependent, per-node latent* that modulates every
   message. In `XYChainGNN` the raw Fourier features of `t` are re-concatenated,
   unchanged, at every layer. See §4.

Fixing (a) is worthwhile if you keep this architecture. But note §7: if you switch to
the bond-CNN, `h` is subsumed — the channel dimension of the conv stack *is* `h`, and
the whole distinction disappears.

---

## 3. Point (b) — the "straightjacket" claim, unpacked

**You are right and the previous phrasing was wrong.** `θ ← θ + Δθ·φ` is exactly the
EGNN coordinate update specialised to `S¹`, it is the correct equivariant form, and it
is not the problem. Concretely, the analogy is tight:

| EGNN (`E_GCL`) | `XYChainConv` |
|---|---|
| direction `x_row - x_col`, normalised | `Δθ = wrap(θ_dst - θ_src)` |
| invariant `radial = ‖x_row - x_col‖²` | invariant-ish `[1/k, cos/sin(kΔθ)]` |
| `trans = direction · coord_mlp(edge_feat) · coords_range` | `trans = Δθ · net(edge_attr, t_emb)` |
| `coord += Σ trans` | `θ += Σ trans` |

So what *is* the real objection? It is **not** the radial update rule; it is the update
rule **in the absence of `h`**. Three precise statements:

### 3.1 One layer is exactly a bond-MLP

With no `h`, `φ` on edge `(i,j)` depends only on `(1/k, Fourier(Δθ_ij), Fourier(t))` —
a function of that single edge. So layer 1 computes

```
v_i^(1) = Σ_{j ∈ nbr(i)}  g_k(Δθ_ij, t),      g_k(δ, t) := δ · φ(δ, k, t)
```

which is a **pure two-body (bond) model**. Any many-body dependence must come from
depth. That is a fine design *if* depth is a cheap way to build many-body terms. Here
it isn't:

### 3.2 The only way to pass a message is to physically move a spin

To let site `i`'s update depend on site `i+3` when `n_neighbors=1`, the network must
actually displace `θ_{i+1}` and `θ_{i+2}` at an earlier layer. But those displacements

- **corrupt the geometry** that all later layers read (`Δθ` is recomputed from the
  *current* `θ`, not from the input), and
- **are added into the answer**, since the output is `θ^(L) − θ^(0) = Σ_layers Δθ_layer`.

There is no "move for communication only, then undo." In EGNN, `h` carries the message
at literally zero cost to the coordinate output. *That* is what "tangles depth with
geometry" should have said: **depth, receptive field, intermediate geometry, and the
final answer are all the same variable.**

### 3.3 The depth is dormant at initialisation

The head of each `XYChainConv` is initialised with `xavier_uniform_(gain=1e-3)`
(`xygnn.py:159`). At init every layer's step is ~0, so every layer sees essentially the
**same** `Δθ`, and the network collapses to

```
v ≈ Σ_{ℓ=1}^{L} (bond-MLP_ℓ)  =  a single bond-MLP
```

The extra depth contributes nothing at init and can only "switch on" through the
*second-order* effect of `θ` actually moving. The gradient that would differentiate
layer ℓ from layer ℓ′ is therefore `O(step²)` at the start of training. You have
4 layers × 4 hidden layers each = a deep stack whose depth is, for a long initial
phase, spending parameters to re-learn the same two-body function four times.

This is testable in ~2 minutes and I'd do it before anything else: **train a single
`XYChainConv` (`n_layers=1`) with the same width and compare.** If it matches the
4-layer model, §3.3 is confirmed and depth is buying nothing.

### 3.4 What is genuinely missing versus EGNN

Besides `h`: `E_GCL` bounds the step with `Tanh()` and a per-layer `coords_range`
budget (`15/3 = 5` for LJ13) and gates messages with a sigmoid attention MLP.
`XYChainConv` does neither — its steps are unbounded. Note that the `XYChainGNN`
**docstring documents `tanh` and `coords_range` arguments that the `__init__`
signature does not accept** (`xygnn.py:198-200` vs `xygnn.py:206-215`); the plan
called for them and they were dropped. Unbounded steps + accumulate-as-output is
precisely the combination that makes the deep stack hard to optimise.

---

## 4. Point (c) — time conditioning. Two real defects, and the EGNN comparison

Your scepticism about "concat-only conditioning is weak" is justified as stated: an MLP
on `[edge features, Fourier(t)]` is a universal approximator of the joint function, and
EGNN does essentially the same thing. FiLM/AdaLN-style conditioning (Perez et al. 2018;
Peebles & Xie 2023) is usually better-conditioned in practice, but that is a
second-order argument and I would not lead with it.

However, there are two defects here that are **not** a matter of taste.

### 4.1 F1 (critical): the time embedding is periodic in `t`, and `t` is not

`xygnn.py:273`: `g_t = fourier_expand(2π·t, time_order)`, i.e.
`[cos(2πkt), sin(2πkt)]_{k=1..K}`. The module docstring advertises this as a feature
("invariant to shifting `t` by an integer", `xygnn.py:36-37`). It is not a feature:

- **`t=0` and `t=1` map to the identical vector** `(1,…,1,0,…,0)`.
- Training samples `t ∈ [ε, 1-ε]` with `ε = 1e-6` (`interpolant.py:327`). At
  `K=16` the embedding difference between `t=ε` and `t=1-ε` has magnitude
  `~2π·K·ε ≈ 1e-4`.
- The required outputs at those two times are of order `10³` and *opposite in
  structure*: `γ̇(t) = 0.2·(1-2t)/(2√(t(1-t)))` → `+∞` at `t→0`, `−∞` at `t→1`.

So the network is asked for an effective Lipschitz constant in `t` of ~`10⁷` across a
region where its input barely changes. The model literally cannot distinguish the two
endpoints, and the endpoints are where the interpolant is singular. **This alone can
account for a large fraction of the residual error, especially in the score net**, and
therefore in the `entropy="dot"` estimator that cell 19 uses for the whole `N`-sweep.

Also, `time_order=16` puts frequencies up to `32π ≈ 100 rad` on a target whose
`t`-dependence is a smooth power law with endpoint singularities — not oscillatory at
all. High-frequency random/Fourier features are known to trade smoothness for capacity
(Tancik et al. 2020); here you are paying that price for structure the target does not
have.

**Fix.** Use a non-periodic embedding, and give the network the schedule directly:

```
t_feat = [ t, 2t-1, log γ(t), α(t), β(t), α̇(t), β̇(t), γ̇(t)·γ(t), … ]
```
or, if you want a learned basis, standard sinusoidal/Gaussian random features on
`log(t/(1-t))` (which is monotone and unbounded, hence non-periodic). Anything that
separates `t=0` from `t=1`.

### 4.2 F2 (critical): no output preconditioning

With `gamma="sqrt"`, `gamma_scale=0.2`, `eps=1e-6`:

| `t` | `γ(t)` | `|s_target| = |z|/γ` | `|γ̇|` |
|---|---|---|---|
| `0.5` | `0.100` | ~10 | 0 |
| `1e-2` | `0.0199` | ~50 | ~5 |
| `1e-4` | `0.0020` | ~500 | ~50 |
| `1e-6` | `0.0002` | ~5000 | ~500 |

The score net's output is `θ^(L) − θ^(0) = Σ_layers Σ_edges Δθ·φ` with `|Δθ| ≤ π`. To
reach 5000 it needs per-edge scalars of order `10³` out of a head initialised at gain
`1e-3`. Similarly `b_target = β̇d + γ̇z` has `|d| ≤ π` but a `γ̇z` term reaching ~500.

This is exactly the problem that preconditioning solves (Karras et al. 2022, EDM §5;
"ε-prediction" in Ho et al. 2020). Parameterise so the network always emits `O(1)`:

```
s(t, x) = -ẑ_ψ(t, x) / γ(t)                      # net predicts the standardised latent
b(t, x) = β̇(t)·d̂_φ(t, x) + γ̇(t)·ẑ_ψ(t, x)        # or a single c_out(t)·net + c_skip(t)·(…)
```

Both `d̂` and `ẑ` are `O(1)` at every `t`, the loss becomes roughly `t`-homogeneous, and
the endpoint singularity is handled analytically instead of being fitted. Independently,
raise `eps` from `1e-6` to ~`1e-3`: uniform `t` sampling puts ~1 sample in 40 batches at
`|s_target| > 500`, and those rare samples dominate the gradient. (The antithetic ±z
averaging in `interpolant.py:426` reduces the *variance* of these terms but not their
*scale*, because `x_t` differs between the two branches.)

### 4.3 Why the LJ13 EGNN looks so much more efficient

You asked specifically why 22 468 parameters look "efficient" on LJ13 and 22 656 look
"inefficient" on a 10-spin chain. The dominant reason is **the training objective, not
the architecture**:

| | LJ13 (`eesi/train/lj13.py`) | XY (`eesi/train/xy.py`) |
|---|---|---|
| objective | plain flow matching | stochastic interpolant, `γ = 0.2√(t(1-t))` |
| networks trained | **one** (velocity only) | **two** (velocity *and* score) |
| regression target | `x₁ - x₀`, magnitude `O(1)` at all `t` | `β̇d + γ̇z` and `-z/γ`, spanning 4 decades |
| endpoint behaviour | finite | singular at `t→0,1` |
| time embedding | learned 32-d, non-periodic, updated per layer | raw 32-d Fourier, **periodic in `t`**, frozen |
| step control | `tanh` × `coords_range = 5` per layer | unbounded |
| training budget | long, tuned, released checkpoint | 3 500 steps on 10 k samples |

The LJ13 net never has to represent a divergent score, never has to distinguish `t=0`
from `t=1` (its embedding is injective in `t`), and gets one bounded target. So the
comparison "22 k params solves LJ13 but not the XY chain" is not really a statement
about the two *systems*' complexity at all — it is a statement about two very
differently-conditioned regression problems. The secondary reasons are the ones in §2
and §3.4 (`h`, tanh-bounded steps, attention gating).

Note also the 3 500-step / 10 k-sample budget. Before concluding the architecture is at
fault, it's worth checking whether the loss is still descending at step 3 500.

---

## 5. Other concrete issues found

### 5.1 The chain graph is rebuilt in Python on every forward call
`XYChainGNN._batch_graph` → `chain_edge_index` (`xygnn.py:106-118`) runs a triple
Python loop over `N × n_neighbors × 2`, builds CPU tensors, and copies to device —
**on every forward**. Per training step that is 2 nets × 2 antithetic branches = 4
rebuilds, plus more during Heun sampling. This is pure overhead and is trivially fixed
with an `lru_cache` keyed on `(N, n_neighbors, device)`. It is probably a meaningful
share of the "horribly inefficient" wall-clock.

### 5.2 The gather/scatter is unnecessary for a chain
A banded, fixed-stencil graph on a 1-D chain does not need `edge_index` at all. `unfold`
or `conv1d` expresses the same computation densely, with no `index_add_`, and will be
substantially faster on GPU at these sizes (`B=256, N=10` → 18 432 edges is far too small
for the scatter kernels to amortise their launch overhead). See §7.1.

### 5.3 Docstring/signature mismatch
`XYChainGNN`'s docstring documents `tanh` and `coords_range` parameters that
`__init__` does not accept. Either implement them (§3.4 says you should) or delete the
lines.

### 5.4 Length generalisation in notebook cell 19
The model is trained only at `N=10` and then evaluated at `N ∈ {1,2,4,…,64}`. Two
issues:
- The receptive field is `n_neighbors × n_layers = 16` sites per side, so *reach* is
  fine relative to `ξ ≈ 2.8`.
- But at `N=10`, 20 % of sites are chain ends; at `N=64` only 3 %. With no explicit
  boundary feature (§2, item 2), the net's learned "bulk" behaviour is contaminated by
  end statistics it can only infer from aggregation degree. Some of the deviation in
  that figure may be this, not the architecture.

Suggested cheap control: train at mixed `N` (the net is already `N`-agnostic), or add
an explicit end-indicator node feature, and see whether the `N`-sweep tightens.

### 5.5 Minor: the MC sampler's proposal is asymmetric
`mcxy` proposes `θ + 0.1·U[0,1)^N` — displacements are strictly positive. Detailed
balance is nonetheless satisfied *in the bond variables*, since the move is
`θ + 0.05·1 + ε` with `ε ~ U(-0.05, 0.05)^N` and the uniform shift leaves every `Δ_i`
unchanged; the absolute angles just acquire a deterministic `+0.05` drift per accepted
step, which wraps and leaves the `θ₁` marginal uniform. So the data is fine. Worth a
comment in the code, because it reads like a bug.

---

## 6. Point (3) — O(2), not SO(2). Confirmed, with exact specs

You are right, and it is cleaner than expected on both sides.

### 6.1 The symmetry

The full symmetry group of both marginals is

```
G  =  O(2)  ×  Z₂^site
   =  { θ_i -> ± θ_i + φ }  ×  { θ_i -> θ_{N+1-i} }
```

- `p₀` (i.i.d. uniform on `(-π, π]`) is invariant under all of it.
- `p₁ ∝ exp(J Σ cos Δ_i)` is invariant: `cos` is even, so `Δ → -Δ` leaves `U` fixed;
  the site reversal maps `Δ_i → -Δ_{N-i}`, also leaving `U` fixed.

The discrete part has **4 elements**: `{id, negate} × {id, reverse}`. The continuous
`U(1)` factor stays where it is, absorbed by the closed-form circular mean — note that
`{negate} × U(1)` *is* the full set of improper elements `θ → φ - θ`, i.e. reflections
about every axis, so enumerating one negation plus optimising `φ` covers all of `O(2)`.

### 6.2 What the symmetry demands of the model

The velocity/score live in the tangent space `R^N` (trivialised by `dθ`), so:

| group element | action on input | required action on output |
|---|---|---|
| rotation `φ` | `θ_i → θ_i + φ` | `v_i → v_i` (**invariant**) |
| negation | `θ_i → -θ_i` | `v_i → -v_i` (**odd/equivariant**) |
| site reversal | `θ_i → θ_{N+1-i}` | `v_i → v_{N+1-i}` (**equivariant**) |

Cross-check against the regression targets in `xyEESI._interpolant_sample`: with
`d = min_image(x₁ - x₀)` and `z ~ N(0, I)`,
`b_target = β̇d + γ̇z` and `s_target = -z/γ`. Under a global rotation `d` is unchanged →
invariant ✓. Under negation `d → -d` and (by symmetry of the Gaussian) `z → -z` →
both targets flip sign ✓. Under site reversal both permute ✓. **So the required model
symmetry is exactly consistent with the targets** — imposing it is a free variance
reduction, equivalent to infinite data augmentation over a 4-element group.

`XYChainGNN` today satisfies rotation-invariance and reversal-equivariance, but **not**
oddness, because `edge_attr` contains `sin(kΔθ)` (`xygnn.py:281`).

### 6.3 The one-line model fix

`trans = Δθ · φ` is odd **iff `φ` is even**. `Δθ` is odd; `1/k` is invariant;
`cos(kΔθ)` is even; `sin(kΔθ)` is odd. So:

> **Restrict the edge Fourier features to cosines only.** Then `φ` is an even function
> of the configuration, `trans` is odd, `θ^(ℓ)` is odd at every layer by induction, and
> the output `θ^(L) - θ^(0)` is exactly odd.

No expressivity is lost within the correct hypothesis class: `Δθ·φ(cos Δθ, cos 2Δθ, …)`
spans the odd, `2π`-periodic functions of `Δθ` (e.g. `sin Δθ` is recovered with
`φ = sin(Δθ)/Δθ`, which is even, smooth, and continuous at the `±π` seam). This is the
exact analogue of EGNN's `O(n)`-equivariance: **invariant (even) scalars times an
equivariant (odd) direction.** The time features are already invariant under negation,
so they need no change.

A cheap architecture-agnostic alternative, if you want to keep `sin` channels for some
reason: explicit antisymmetrisation, `v(θ) := ½[F(θ) − F(−θ)]`. Exact, but 2× the
forward cost.

**Verification to add to `tests/test_xygnn.py`:** `net(t, -x) == -net(t, x)` to
`float64` tolerance, alongside the existing rotation/reversal tests. Today that test
should *fail*; after the cos-only change it should pass exactly.

### 6.4 The OT fix

Exactly as you described — extend the enumerated discrete group from 2 to 4 elements.
`eesi/ot.py:247` is the single place it is defined:

```python
def _xy_group_elements(x, reflect=True, negate=True):
    """Discrete orbit of the noise under Z2^site x Z2^spin. Yields (B, N) tensors."""
    g = (x,)
    if reflect: g += (x.flip(-1),)
    if negate:  g += (-x,)
    if reflect and negate: g += (-x.flip(-1),)
    return g
```

Everything downstream already works unchanged:

- `xy_cost_matrix` loops over `_xy_group_elements`, so it goes from 2 to 4 cost matrices
  — exactly the "additional two cost matrix calculations per batch pair" you predicted.
  Each is two `(B,N) @ (N,B)` matmuls, so the added cost is `2 × 2 × B²N` flops:
  at `B=256, N=10` that is ~2.6 MFLOP, i.e. nothing. The `argmin` over the stacked
  `(n_g, B, B)` already generalises.
- `_xy_apply_alignment` (the circular mean) is unchanged — the closed form
  `φ* = atan2(Σ sin d, Σ cos d)` applies to any group element, including `-x₀`.
- `xy_ot_couple`'s survivor logic (`torch.stack(g, 0)[pick, arange]`) already indexes an
  arbitrary-length tuple.
- The **outer minibatch OT is untouched**, as you specified.
- No wrapping issue: `x ∈ (-π, π] → -x ∈ [-π, π)`, and everything downstream goes
  through `sin`/`cos` or `angle_wrap`.

**Marginal preservation still holds.** The argument in the `eesi/ot.py` docstring needs
only that `p₀` and `p₁` are `G`-invariant, which §6.1 establishes for the enlarged `G`.
Extend `tests/test_ot.py::test_xy_marginal_preserved_over_z2_u1` to the 4-element group,
and add a `test_xy_negation_branch_is_live` mirroring the existing
`test_xy_reflection_branch_is_live`. The CPU oracle in `tests/ot_reference.py:145`
(`xy_match_pair`, which loops `for r in (0,1)`) needs the same 2→4 extension so the
comparison tests stay meaningful.

**Expected benefit.** Modest on transport cost alone — your own measurements
(`eesi/train/xy.py` docstring) show `align` contributes ~20 % while `batch` contributes
~32 % at `B=256`, so doubling a small discrete group will shave a few more percent. The
real payoff is (i) consistency — the coupling and the model then respect the same group
— and (ii) the model-side constraint in §6.3, which halves the effective hypothesis
class. I would not expect the OT change alone to move the entropy numbers much; I would
expect the model-side oddness to.

---

## 7. Point (2) — alternative architectures

Requirements, restated: input `θ ∈ (S¹)^N` as `[B, N]`; output `v ∈ R^N` (a tangent
vector, same shape); invariant under global rotation; odd under `θ → -θ`; equivariant
under site reversal.

**The key simplification.** The bond variables `Δ_i = wrap(θ_{i+1} - θ_i)`, `i = 1..N-1`,
are a *complete and non-redundant* set of `U(1)` invariants (the quotient
`T^N / U(1)` has dimension `N-1`). Therefore **every** admissible model has the form

```
v = F(Δ_1, …, Δ_{N-1}; t),        F : R^{N-1} × [0,1] -> R^N
```

Rotation-equivariance is not an architectural constraint at all — it is a change of
variables you do once, in the first line. What remains is a **1-D sequence-to-sequence
model on `N-1` bond tokens producing `N` site outputs**, with two `Z₂` parities to
respect. Any sequence architecture is now admissible.

### 7.1 Recommended: parity-structured bond CNN (a "divergence-form" net)

This is my primary recommendation. It is small, exactly equivariant, fast (no
gather/scatter), and — critically — **contains the analytic answer as a
one-parameter special case**.

```python
Δ    = wrap(θ[:, 1:] - θ[:, :-1])                 # [B, N-1]      bond variables
c    = stack([cos(k*Δ) for k in 1..K], dim=1)     # [B, K, N-1]   EVEN under θ -> -θ
h    = cat([c, t_feat.expand(...)], dim=1)        # [B, K+T, N-1] (t_feat is invariant)
H    = ResidualDilatedConv1d(h)                   # [B, C, N-1]   even, dilations 1,2,4
w    = Conv1d(C -> 1)(H)                          # [B, 1, N-1]   even scalar per bond
f    = w * sin(Δ)                                 # [B, N-1]      ODD bond "force"
v    = pad_right(f) - pad_left(f)                 # [B, N]        discrete divergence
```

Properties, all exact:

- **Rotation-invariant**: only `Δ` enters. ✓
- **Odd under `θ → -θ`**: `cos(kΔ)` is even so `w` is even; `sin Δ` is odd; `f` is odd;
  the difference of odd quantities is odd. ✓
- **Site-reversal-equivariant**, *provided the conv kernels are palindromic*
  (`w ← ½(w + flip(w))`, one line, and it halves the kernel parameters): under reversal
  `Δ_i → -Δ_{N-i}` ⟹ `c` reverses, `w` reverses, `sin Δ` reverses-and-negates ⟹
  `f_i → -f_{N-i}` ⟹ `v_j → v_{N+1-j}`. ✓
- **Contains the exact target score** at `K=1`, `w ≡ J`, zero conv layers:
  `v_j = J(sin Δ_j - sin Δ_{j-1}) = ∇_j log p₁`. This is the single most attractive
  property — the network starts from, or can be initialised at, the analytic answer,
  and only has to learn the `t`-dependent deviation.
- **Conserves total angle**: `Σ_j v_j = 0` by telescoping, matching the fact that the
  rotational mode is unlearnable (both marginals are `U(1)`-invariant, so the true drift
  has no net rotation component). This is the `S¹` analogue of `LJ13Dynamics`'s
  `vel - vel.mean(1)` COM projection. Currently `XYChainGNN` does *not* enforce this
  and must learn it.

Cost: `K=4, T=8, C=16`, 4 layers of kernel 3 (palindromic ⟹ 2 free weights each) is
roughly **1.5–3 k parameters per net**, versus 11 328 today — and the receptive field
with dilations `1,2,4,8` is 31 bonds, comfortably beyond `3ξ ≈ 8`.

Padding: use zero-padding (a missing bond has `sin Δ = 0`, which is the physically
correct "no coupling" boundary) or an explicit end-indicator channel.

References for the ingredients: dilated 1-D convolutions — van den Oord et al.,
*WaveNet* (arXiv:1609.03499, 2016); CNNs on lattice-field configurations —
Albergo, Kanwar & Shanahan, *Phys. Rev. D* **100**, 034515 (2019); masked CNNs for
lattice statistical mechanics — Wu, Wang & Zhang, *Phys. Rev. Lett.* **122**, 080602
(2019).

### 7.2 Bond-MLP / explicit body-order expansion (the strongest baseline)

Because §1 shows `p₁` is exactly two-body in the bonds, write

```
v_j = Σ_{k=-R..R} g_k(Δ_{j+k}; t)      (2-body)
    + Σ_{k,l}     g_{kl}(Δ_{j+k}, Δ_{j+l}; t)   (3-body, optional)
```

with each `g` an odd-in-`Δ` MLP (`g(δ) = δ · MLP(cos δ, cos 2δ, …, t)`). At `R=2`,
2-body only, this is a few hundred parameters. **I would run this first as a
diagnostic**: if a 2-body model with `R=2` matches the 22 k GNN on `ΔU` and `ΔS`, that
settles the question of whether the GNN's capacity is doing anything, and it becomes
the model you ship.

This is the 1-D analogue of systematic body-order expansions in interatomic potentials:
Drautz, *Atomic cluster expansion*, **Phys. Rev. B 99**, 014104 (2019); Batatia et al.,
*MACE*, NeurIPS 2022. The relevant idea is that a controlled body-order truncation
often beats an unstructured deep net on systems with short correlation length — which
this one certainly is.

### 7.3 Keep `XYChainGNN`, but repair it

If you prefer continuity with the existing code, the minimal repair list is:
cos-only edge features (§6.3); add `h` initialised from the time embedding with an
EGNN-style `node_mlp` (§2); `tanh` + `coords_range` on the coordinate step (§3.4);
cache the graph (§5.1); non-periodic time embedding (§4.1); output preconditioning
(§4.2); subtract the mean of `v` to kill the rotational mode (§7.1). That is close to a
rewrite, and the result would still carry the gather/scatter overhead for a stencil that
doesn't need it — which is why I'd rather go to §7.1.

### 7.4 Bond transformer (only if `N` grows a lot)

Self-attention over the `N-1` bond tokens with a relative-position bias, same parity
construction as §7.1 for the output head. Overkill at `N=10–64`, but it is the natural
scaling path if you later want `N ~ 10³` or 2-D lattices. Nothing here needs it now.

### 7.5 Orthogonal option: change the generative map, not the network

Worth knowing about, though it is a different project. The compact `U(1)` variables here
are exactly the setting of circular-spline normalising flows: Rezende et al.,
*Normalizing Flows on Tori and Spheres*, ICML 2020; Kanwar et al., *Equivariant
flow-based sampling for lattice gauge theory*, **Phys. Rev. Lett. 125**, 121601 (2020).
Since `p₁` factorises into independent von Mises bonds, a bond-wise circular spline
flow would be essentially exact with `O(10²)` parameters and an *exact* likelihood, so
`ΔS` would come from a closed-form log-det rather than a learned score. That would
sidestep F1/F2 entirely — but it abandons the stochastic-interpolant framing, which I
take to be the point of the project.

---

## 8. Suggested order of work

Cheap and decisive first:

1. **Diagnostic (minutes).** Train `n_layers=1` at the same width. If it matches the
   4-layer model, §3.3 is confirmed.
2. **Diagnostic (minutes).** Bin the drift/score losses by `t` and plot. If the mass is
   concentrated near `t ∈ {0, 1}`, F1/F2 are confirmed as dominant.
3. **F1 fix.** Non-periodic time embedding. One function, no architecture change.
4. **F2 fix.** Preconditioned output heads + `eps = 1e-3`. Touches `xyEESI`/`train.xy`,
   not the net.
5. **F4 model side.** Cos-only edge features + the `net(t, -x) == -net(t, x)` test.
   One line plus one test.
6. **F4 OT side.** `_xy_group_elements` 2 → 4 elements, plus the oracle and two tests.
7. **Baseline.** The §7.2 two-body bond-MLP, as a floor to beat.
8. **Rebuild.** The §7.1 bond CNN, if 7 shows there is many-body structure worth
   modelling.

Steps 3–6 are all independent of which architecture you end up with.

---

## 9. References

*(Cited from memory; worth a quick check of volume/page numbers before they go into a
manuscript.)*

**Architectures & equivariance**
- Satorras, Hoogeboom & Welling, "E(n) Equivariant Graph Neural Networks", ICML 2021.
- Köhler, Klein & Noé, "Equivariant Flows: Exact Likelihood Generative Learning for
  Symmetric Densities", ICML 2020. — the invariant-density / equivariant-map theorem
  underlying `eesi/ot.py`'s marginal-preservation argument.
- Drautz, "Atomic cluster expansion for accurate and transferable interatomic
  potentials", Phys. Rev. B **99**, 014104 (2019).
- Batatia, Kovács, Simm, Ortner & Csányi, "MACE: Higher Order Equivariant Message
  Passing Neural Networks", NeurIPS 2022.
- van den Oord et al., "WaveNet: A Generative Model for Raw Audio", arXiv:1609.03499
  (2016).

**Interpolants, flow matching, couplings**
- Albergo, Boffi & Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for
  Flows and Diffusions", arXiv:2303.08797 (2023).
- Klein, Krämer & Noé, "Equivariant Flow Matching", NeurIPS 2023.
- Tong et al., "Improving and Generalizing Flow-Based Generative Models with Minibatch
  Optimal Transport", TMLR 2024.

**Conditioning & preconditioning**
- Karras, Aittala, Aila & Laine, "Elucidating the Design Space of Diffusion-Based
  Generative Models", NeurIPS 2022. — §5, the `c_skip/c_out/c_in/c_noise`
  preconditioning that §4.2 recommends.
- Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020. —
  ε-prediction.
- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low
  Dimensional Domains", NeurIPS 2020.
- Perez, Strub, de Vries, Dumoulin & Courville, "FiLM: Visual Reasoning with a General
  Conditioning Layer", AAAI 2018.
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), ICCV 2023. —
  adaLN-Zero conditioning.

**Compact/periodic variables**
- Rezende, Papamakarios, Racanière, Albergo, Kanwar, Shanahan & Cranmer, "Normalizing
  Flows on Tori and Spheres", ICML 2020.
- Kanwar, Albergo, Boyda, Cranmer, Hackett, Racanière, Rezende & Shanahan,
  "Equivariant flow-based sampling for lattice gauge theory", Phys. Rev. Lett. **125**,
  121601 (2020).
- Albergo, Kanwar & Shanahan, "Flow-based generative models for Markov chain Monte
  Carlo in lattice field theory", Phys. Rev. D **100**, 034515 (2019).
- Wu, Wang & Zhang, "Solving Statistical Mechanics Using Variational Autoregressive
  Networks", Phys. Rev. Lett. **122**, 080602 (2019).
