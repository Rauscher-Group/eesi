"""EESI: general stochastic interpolant in Euclidean space.

Single class, no inheritance chain. Wraps two field networks (`net_b`, `net_s` --
any module with the right signature; see the `eesi.systems` subpackages) and exposes:

    .loss(x1, x0, entropy)            training loss dict {"b": loss_b, "s": loss_s},
                                      plus a detached "ent_dot"/"ent_zdot" channel
                                      when `entropy` is set.
    .sample(x0, n_steps, eps, ...)    one integrator for the whole family.
    .entropy_estimate(x1, x0, method) interpolant-based entropy estimator.

`sample` folds the ODE/SDE, plain/entropy, and final/trajectory variants into a
single call controlled by flags:

    eps=0.0            -> probability-flow ODE (Heun/Euler, net_b only);
    eps>0              -> reverse-time SDE (Euler-Maruyama, net_b + net_s).
    entropy=None       -> no entropy channel;
    entropy="dot"      -> accumulate -integral b.s dt   (needs net_s);
    entropy="div"      -> accumulate  integral div(b) dt (needs only net_b).
    return_traj=False  -> return the t=1 sample (and final entropy);
    return_traj=True   -> return the whole path (and running entropy) + time grid.

The two entropy accumulators agree in expectation because, under p_t,
E[b.s] = -E[div b] (integration by parts), so -integral b.s = integral div(b).

`entropy_estimate(x1, x0, method)` is a separate, interpolant-based Monte-Carlo
estimator (it samples the interpolant directly rather than integrating the
learned dynamics); `method="div"` traces net_b, `method="dot"` uses -b.s, and
`method="zdot"` replaces the learned score in -b.s with the exact conditional
score -z/gamma(t) that the interpolant draw already carries (needs no net_s and
no autograd; estimator only, since an integrated trajectory has no latent z).
The `entropy` keyword of `loss` gives the same "dot"/"zdot" accumulators as a
training-time diagnostic, reusing the draw the loss already made rather than
taking a fresh one.

Time convention: t goes from 0 (x0, base) to 1 (x1, data). The interpolant is
a general stochastic interpolant with a latent variable z ~ N(0, I):

    x_t   = alpha(t)·x0 + beta(t)·x1 + gamma(t)·z
    dx/dt = alpha'(t)·x0 + beta'(t)·x1 + gamma'(t)·z

`gamma(t)` is a multiplicative noise schedule that vanishes at the endpoints
(gamma(0)=gamma(1)=0). The path (alpha, beta) and the schedule gamma are each
chosen by name from `_PATHS` / `_GAMMAS` (see below).

The drift field b(t, x_t) regresses onto dx/dt; its L2 optimum is the
probability-flow drift E[dx/dt | x_t]. The score field s(t, x_t) targets
∇log p_t(x_t). How they are trained depends on the latent schedule gamma
(see `EESI.loss`):

- gamma="none" (deterministic interpolant): drift squared regression, and the
  score via implicit score matching E[||s||^2 + 2 div(s)], with div(s) estimated
  by the exact Jacobian diagonal ("exact") or the Hutchinson trace estimator
  ("hutchinson", default).
- gamma!="none": the conditional x_t | (x0, x1) is Gaussian, so drift and score
  use the cheaper denoising objectives with no divergence estimate. Antithetic
  sampling (+z and -z, averaged) cancels the endpoint 1/gamma and gamma'
  singularities.
"""
from __future__ import annotations

import math

import torch
from torch import nn



# ---- interpolant schedules -------------------------------------------------
#
# Each path returns (alpha, beta, alpha_dot, beta_dot); each gamma returns
# (gamma, gamma_dot). All outputs broadcast against x from t, whose trailing
# singleton dims (`_draw_time`) match the per-sample rank: [B, 1] for [B, d]
# states, [B, 1, 1] for [B, N, 3] point clouds.


def _path_linear(t: torch.Tensor):
    """alpha=1-t, beta=t: the straight line x0 -> x1."""
    one = torch.ones_like(t)
    return 1.0 - t, t, -one, one


def _path_trig(t: torch.Tensor):
    """alpha=cos(pi t/2), beta=sin(pi t/2): variance-preserving path."""
    half_pi = math.pi / 2.0
    a = torch.cos(half_pi * t)
    b = torch.sin(half_pi * t)
    return a, b, -half_pi * b, half_pi * a


def _path_trig2(t: torch.Tensor):
    """alpha=cos^2(pi t/2), beta=sin^2(pi t/2): trig path with flat endpoints.

    Squares the `trig` path, so alpha+beta=1 (like `linear`/`trig`) but the
    derivatives vanish at both endpoints: alpha'(0)=alpha'(1)=beta'(0)=beta'(1)=0.
    Using d/dt cos^2(pi t/2) = -(pi/2) sin(pi t), the slopes are +/-(pi/2) sin(pi t).
    """
    half_pi = math.pi / 2.0
    a = torch.cos(half_pi * t) ** 2
    b = torch.sin(half_pi * t) ** 2
    b_dot = half_pi * torch.sin(math.pi * t)
    return a, b, -b_dot, b_dot


def _path_encdec(t: torch.Tensor):
    """Encoding-decoding path via c=cos^2(pi t).

    For t < 1/2 the interpolant tracks x0 (alpha=c, beta=0); for t >= 1/2 it
    tracks x1 (alpha=0, beta=c). It passes through the origin at t=1/2 and never
    mixes x0 with x1. The seam is smooth because c'(1/2) = 0.
    """
    c = torch.cos(math.pi * t) ** 2
    c_dot = -math.pi * torch.sin(2.0 * math.pi * t)
    first = (t < 0.5).to(t.dtype)      # 1 on [0, 1/2), 0 on [1/2, 1]
    second = 1.0 - first
    return c * first, c * second, c_dot * first, c_dot * second


_PATHS = {
    "linear": _path_linear,
    "trig": _path_trig,
    "trig2": _path_trig2,
    "encdec": _path_encdec,
}


def _gamma_none(t: torch.Tensor, eps: float):
    """gamma=0: recovers the deterministic interpolant (no latent noise)."""
    z = torch.zeros_like(t)
    return z, z


def _gamma_quad(t: torch.Tensor, eps: float):
    """gamma=t(1-t): smooth derivative everywhere, zero at the endpoints."""
    return t * (1.0 - t), 1.0 - 2.0 * t


def _gamma_sqrt(t: torch.Tensor, eps: float):
    """gamma=sqrt(t(1-t)): the standard interpolant choice; derivative diverges at the endpoints.

    The inner term is floored by `eps` so gamma_dot stays finite; callers should
    also keep t away from 0 and 1.
    """
    inner = (t * (1.0 - t)).clamp_min(eps)
    g = inner.sqrt()
    g_dot = (1.0 - 2.0 * t) / (2.0 * g)
    return g, g_dot


def _gamma_sin2(t: torch.Tensor, eps: float):
    """gamma=sin^2(pi t): vanishes at the endpoints with zero slope there.

    gamma'(t) = pi sin(2 pi t), so gamma'(0)=gamma'(1)=0 as well.
    """
    g = torch.sin(math.pi * t) ** 2
    g_dot = math.pi * torch.sin(2.0 * math.pi * t)
    return g, g_dot


_GAMMAS = {
    "none": _gamma_none,
    "quad": _gamma_quad,
    "sqrt": _gamma_sqrt,
    "sin2": _gamma_sin2,
}


def _div_exact(s: torch.Tensor, x_t: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Exact divergence of s w.r.t. x_t via Jacobian diagonal sum.

    `create_graph` must be True when the divergence feeds a loss that is later
    backpropped (training); pass False for inference-only use (e.g. entropy
    estimation) to avoid building — and accumulating — a second-order graph.
    """
    B = s.shape[0]
    s_flat = s.reshape(B, -1)
    nd = s_flat.shape[1]
    div = x_t.new_zeros(B)
    for i in range(nd):
        (g,) = torch.autograd.grad(
            s_flat[:, i].sum(), x_t,
            create_graph=create_graph, retain_graph=True,
        )
        div = div + g.reshape(B, -1)[:, i]
    return div


def _div_hutchinson(s: torch.Tensor, x_t: torch.Tensor, n_probes: int, create_graph: bool = True,
                    noise_fn=None) -> torch.Tensor:
    """Hutchinson trace estimator for div(s): E_v[v · grad_x(v·s)].

    `create_graph` must be True when the divergence feeds a loss that is later
    backpropped (training); pass False for inference-only use (e.g. entropy
    estimation) to avoid building — and accumulating — a second-order graph.

    `noise_fn` draws the probe vectors (default `torch.randn_like`). A subclass on
    a constrained subspace passes a projecting draw so the trace is taken there:
    with probes v ~ N(0, P) and a P-equivariant field, E[v^T J v] = tr(PJ), the
    subspace divergence (see `LJ13EESI`).
    """
    noise_fn = noise_fn or torch.randn_like
    B = s.shape[0]
    div = x_t.new_zeros(B)
    for _ in range(n_probes):
        v = noise_fn(s)
        (g,) = torch.autograd.grad(
            (v * s).sum(), x_t,
            create_graph=create_graph, retain_graph=True,
        )
        div = div + (v * g).flatten(1).sum(-1)
    return div / n_probes


def score_loss(s: torch.Tensor, div_s: torch.Tensor) -> torch.Tensor:
    """ISM loss E[||s||^2 + 2·div(s)]."""
    return (s.square().flatten(1).sum(-1) + 2.0 * div_s).mean()


def _denoising_loss(net_out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample denoising loss 1/2||net||^2 - net·target, summed over features.

    Equals 1/2||net - target||^2 up to a target-only constant, which is dropped
    because it carries no gradient w.r.t. the network. Returns shape [B]. Averaging
    this over the antithetic +z / -z pair reproduces the drift/score objectives
    used in `EESI.loss`. Features are flattened, so any per-sample shape ([B, d] or
    [B, N, 3]) reduces to the same [B].
    """
    return 0.5 * net_out.square().flatten(1).sum(-1) - (net_out * target).flatten(1).sum(-1)


class EESI(nn.Module):
    """General stochastic interpolant in Euclidean space.

    Args:
        net_b: Module for the velocity field b(t, x). Forward: (t, x) -> [B, d].
        net_s: Module for the score field s(t, x). Forward: (t, x) -> [B, d].
        d: spatial dimension. Optional and unused at run time — every method
            reads the shape from the incoming `x0`/`x1` tensors, so a single
            instance handles any dimension. Retained only as informational
            metadata (e.g. `xyEESI` samples chains of any length `N`).
        path: interpolant path (alpha, beta), one of `_PATHS`:
            "linear" (default), "trig", "trig2", "encdec". "trig2" uses
            alpha=cos^2(pi t/2), beta=sin^2(pi t/2), which meet the endpoints
            with zero slope.
        gamma: latent-noise schedule, one of `_GAMMAS`:
            "none", "quad" (default), "sqrt", "sin2". gamma(0)=gamma(1)=0. With
            "none" the loss uses implicit score matching; otherwise it uses the
            cheaper antithetic denoising objectives (see `loss`).
        gamma_scale: multiplicative coefficient on gamma (and gamma').
        eps: floors `t(1-t)` for the "sqrt" schedule and keeps sampled t in
            [eps, 1-eps] so the drift target stays finite at the endpoints.
        score_div_method: divergence estimator for the ISM loss (used only when
            gamma="none") — "hutchinson" (default, O(n_probes) passes) or
            "exact" (O(d) passes).
        n_hutchinson_probes: number of probe vectors when using "hutchinson".
    """

    def __init__(
        self,
        net_b: nn.Module,
        net_s: nn.Module,
        d: int | None = None,
        path: str = "linear",
        gamma: str = "sqrt",
        gamma_scale: float = 1.0,
        eps: float = 1e-6,
        learn_score = True,
        score_div_method: str = "hutchinson",
        n_hutchinson_probes: int = 32,
    ):
        super().__init__()
        self.net_b = net_b
        self.net_s = net_s
        self.d = d
        if path not in _PATHS:
            raise ValueError(f"path must be one of {sorted(_PATHS)}, got {path!r}")
        if gamma not in _GAMMAS:
            raise ValueError(f"gamma must be one of {sorted(_GAMMAS)}, got {gamma!r}")
        self.path = path
        self.gamma = gamma
        self._path = _PATHS[path]
        self._gamma = _GAMMAS[gamma]
        self.gamma_scale = float(gamma_scale)
        self.eps = float(eps)
        if score_div_method not in ("hutchinson", "exact"):
            raise ValueError(f"score_div_method must be 'hutchinson' or 'exact', got {score_div_method!r}")
        self.score_div_method = score_div_method
        self.n_hutchinson_probes = int(n_hutchinson_probes)
        self.learn_score = learn_score

    def _divergence(self, s: torch.Tensor, x_t: torch.Tensor, create_graph: bool = True,
                    t: torch.Tensor | None = None) -> torch.Tensor:
        # `t` is unused by the autograd estimators here -- the time is already
        # baked into the graph of `s` -- but is threaded through by the callers so
        # a subclass override can recompute the field (e.g. `LJ13EESI`, whose
        # forward-mode subspace divergence re-evaluates the net at `t`).
        if self.score_div_method == "exact":
            return _div_exact(s, x_t, create_graph=create_graph)
        return _div_hutchinson(s, x_t, self.n_hutchinson_probes, create_graph=create_graph,
                               noise_fn=self._noise_like)

    # ---- the two draws a subclass may need to constrain ---------------------
    #
    # Every raw Gaussian this class samples -- the latent z, the SDE diffusion
    # term, and the Hutchinson probes -- goes through `_noise_like`, and every
    # time vector through `_draw_time`. A geometry that lives on a subspace (e.g.
    # `LJ13EESI` on the COM-free subspace) overrides `_noise_like` alone; the base
    # returns an unconstrained standard normal.

    def _noise_like(self, ref: torch.Tensor) -> torch.Tensor:
        """A standard-normal draw shaped like `ref`. Override to constrain it."""
        return torch.randn_like(ref)

    def _draw_time(self, x: torch.Tensor):
        """Sample t in [eps, 1-eps], broadcastable to `x`. Returns (t, t_b).

        t has shape (B, 1, ..., 1) so it broadcasts against any per-sample shape
        ([B, d] or [B, N, 3]); t_b is the flat [B] the networks consume.
        """
        B = x.shape[0]
        shape = (B,) + (1,) * (x.dim() - 1)
        t = torch.rand(shape, device=x.device, dtype=x.dtype)
        t = t * (1.0 - 2.0 * self.eps) + self.eps
        return t, t.reshape(B)

    def _scaled_gamma(self, t: torch.Tensor):
        """gamma(t) and gamma'(t) with the `gamma_scale` coefficient applied."""
        g, g_dot = self._gamma(t, self.eps)
        return self.gamma_scale * g, self.gamma_scale * g_dot

    @staticmethod
    def _assert_matched(x0: torch.Tensor, x1: torch.Tensor) -> None:
        """Fail loudly if the endpoint tensors disagree in shape.

        The dimension is read from the inputs (never `self.d`), so a mismatch
        would otherwise broadcast silently into a wrong-shaped interpolant.
        """
        if x0.shape != x1.shape:
            raise ValueError(f"x0 and x1 must have the same shape, got {tuple(x0.shape)} and {tuple(x1.shape)}")

    # ---- interpolant sampling (the one topology-aware hook) -----------------
    #
    # `loss` and the `entropy_estimate_*` methods build the interpolant only
    # through `_interpolant_sample`, so a subclass changes the geometry by
    # overriding this single method (e.g. `xyEESI` for the periodic angle
    # manifold). Antithetic sampling just calls it with +z and -z.

    def _interpolant_sample(
        self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor
    ):
        """Interpolant position and regression targets for one latent draw `z`.

        Returns:
            x_t:      interpolant position  I_t + gamma·z     (the network input).
            b_target: drift target          dI/dt + gamma'·z  (the velocity dx/dt).
            s_target: score target          -z / gamma        (the conditional score).

        Euclidean straight line: I_t = alpha·x0 + beta·x1. For antithetic sampling
        call with +z and -z (the *same* z) and average the two per-branch losses.
        This is the only geometry-aware method; `xyEESI` overrides it.
        """
        self._assert_matched(x0, x1)
        alpha, beta, alpha_dot, beta_dot = self._path(t)
        g, g_dot = self._scaled_gamma(t)
        x_t = alpha * x0 + beta * x1+ g * z        
        dI_dt = alpha_dot * x0 + beta_dot * x1
        b_target = dI_dt + g_dot * z
        s_target = -z / g.clamp_min(1e-12)
        return x_t, b_target, s_target

    def _check_entropy_channel(self, entropy: str | None) -> None:
        """Validate the `entropy` keyword of `loss`. See that method's docstring."""
        if entropy is None:
            return
        if entropy not in ("dot", "zdot", "both"):
            raise ValueError(f"entropy must be None, 'dot', 'zdot', or 'both', got {entropy!r}")
        if entropy in ("zdot", "both") and self.gamma == "none":
            raise ValueError(
                "entropy='zdot'/'both' needs a non-zero latent schedule: with gamma='none' the "
                "interpolant carries no latent z and the conditional score -z/gamma is "
                "undefined. Use entropy='dot' instead."
            )
        if entropy in ("dot", "both") and self.gamma == "none" and not self.learn_score:
            raise ValueError(
                "entropy='dot'/'both' reads net_s, which with gamma='none' and "
                "learn_score=False receives no gradient -- the accumulator would report a "
                "randomly initialised score. Use entropy='zdot' instead."
            )

    def loss(
        self,
        x1: torch.Tensor,
        x0: torch.Tensor,
        entropy: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Drift and score losses for one batch.

        Returns {"b": loss_b, "s": loss_s}. Caller sums for backward:

            losses = model.loss(x1, x0)
            (losses["b"] + losses["s"]).backward()

        `entropy` adds a detached entropy channel to the returned dict, computed
        from the interpolant draw this call already made — no extra network
        evaluation, no autograd, O(d) elementwise work. It is the training-time
        counterpart of `entropy_estimate`, which redraws its own (t, z):

        - "dot":  adds "ent_dot",  the batch mean of -(b · s) with the LEARNED
          score, averaged over the antithetic +z/-z pair (`entropy_estimate`
          uses one branch; both are unbiased, so the average is too).
        - "zdot": adds "ent_zdot", the same accumulator with the exact conditional
          score -z/gamma(t), antithetically averaged. Matches
          `entropy_estimate(method="zdot")` draw for draw.
        - "both": adds both. Their disagreement measures how far net_s is from
          the true score.

        "zdot" needs gamma != "none"; "dot" needs a net_s that is actually
        trained, which fails only for gamma="none" with `learn_score=False` --
        there nothing touches net_s and the accumulator would read an untrained
        network. (With gamma != "none" the denoising objective trains net_s
        whatever `learn_score` says, which is what makes "dot" available to
        `LJ13EESI`/`TAPEESI`; they force the flag off only to disable ISM, whose
        subspace divergence is not differentiable.) Both are errors, not silent
        defaults. See
        `entropy_estimate`'s docstring on why "zdot" is unbiased without net_s,
        and on its `eps` sensitivity — the channel inherits the training `eps`.

        The score objective depends on the latent schedule `gamma`:

        - gamma="none": no latent noise, so x_t = I_t is deterministic given
          (x0, x1). The drift is a squared regression onto dx/dt and the score
          uses implicit score matching E[||s||^2 + 2·div(s)], with div(s)
          estimated by `score_div_method` / `n_hutchinson_probes`.
        - gamma!="none": the conditional x_t | (x0, x1) is Gaussian, so the
          score is known in closed form and we use the much cheaper denoising
          objectives — no divergence estimate. Antithetic sampling (each sample
          evaluated at +z and -z, then averaged) cancels the 1/gamma and gamma'
          endpoint singularities. `score_div_method` / `n_hutchinson_probes` are
          ignored in this case.
        """
        self._check_entropy_channel(entropy)
        t, t_b = self._draw_time(x1)

        if self.gamma == "none":
            # Deterministic interpolant (no latent noise): drift regression + ISM.
            x_t, b_target, _ = self._interpolant_sample(t, x0, x1, torch.zeros_like(x1))
            b = self.net_b(t_b, x_t)
            loss_b = (b - b_target).square().mean()

            if self.learn_score:
                with torch.enable_grad():
                    x_t_s = x_t.detach().requires_grad_(True)
                    s = self.net_s(t_b, x_t_s)
                    div_s = self._divergence(s, x_t_s, t=t_b)
                    loss_s = score_loss(s, div_s)
            else:
                loss_s = torch.zeros_like(loss_b)

            out = {"b": loss_b, "s": loss_s}
            if entropy is not None:
                # Only "dot" reaches here -- "zdot" needs a latent z and was
                # rejected above. `s` is built under enable_grad, hence detach().
                with torch.no_grad():
                    out["ent_dot"] = -(b * s.detach()).flatten(1).sum(-1).mean()
            return out

        # Non-zero gamma: antithetic denoising. Sample z once and evaluate the
        # per-branch denoising losses at +z and -z; the -z partner cancels the
        # endpoint 1/gamma and gamma' singularities. Only `_interpolant_sample`
        # is topology-aware, so periodicity handling lives entirely there.
        z = self._noise_like(x1)
        loss_b = x1.new_zeros(())
        loss_s = x1.new_zeros(())
        ent_dot = x1.new_zeros(())
        ent_zdot = x1.new_zeros(())
        for z_branch in (z, -z):
            x_t, b_target, s_target = self._interpolant_sample(t, x0, x1, z_branch)
            b = self.net_b(t_b, x_t)
            s = self.net_s(t_b, x_t)
            loss_b = loss_b + 0.5 * _denoising_loss(b, b_target).mean()
            loss_s = loss_s + 0.5 * _denoising_loss(s, s_target).mean()
            #loss_b = loss_b + 0.5 * (b - b_target).square().mean()
            #loss_s = loss_s + 0.5 * (s - s_target).square().mean()

            # The entropy channel is free here: both accumulators are elementwise
            # products of tensors the losses above already built. no_grad keeps
            # them out of the graph the caller is about to backward through.
            if entropy is not None:
                with torch.no_grad():
                    if entropy in ("dot", "both"):
                        ent_dot = ent_dot - 0.5 * (b * s).flatten(1).sum(-1).mean()
                    if entropy in ("zdot", "both"):
                        ent_zdot = ent_zdot - 0.5 * (b * s_target).flatten(1).sum(-1).mean()

        out = {"b": loss_b, "s": loss_s}
        if entropy in ("dot", "both"):
            out["ent_dot"] = ent_dot
        if entropy in ("zdot", "both"):
            out["ent_zdot"] = ent_zdot
        return out

    @torch.no_grad()
    def entropy_estimate(
        self,
        x1: torch.Tensor,
        x0: torch.Tensor,
        method: str = "div",
    ) -> torch.Tensor:
        r"""Estimated entropy difference between P0 and P1 for one batch.

        Interpolant-based Monte-Carlo estimator: it samples the interpolant
        directly (draw t and z, build x_t) rather than integrating the learned
        dynamics. Returns a tensor of shape [B]; the entropy estimate is its mean.

        `method` selects the accumulator. "dot"/"div" share their vocabulary with
        the `entropy` flag of :meth:`sample`; "zdot" is estimator-only, because an
        integrated trajectory has no latent z to condition on:

        - "div" (default): the divergence of the velocity field (net_b) sampled
          along the interpolant, handled by `self._divergence` (Hutchinson by
          default; the exact subspace trace in `LJ13EESI`). Needs only net_b.
        - "dot": the accumulator -(b \cdot s) along the interpolant. Needs a
          learned score in addition to the velocity; no divergence is required,
          though errors in the score now compound those in the velocity.
        - "zdot": the same accumulator, but with the LEARNED score replaced by
          the exact conditional score -z/gamma(t) that the interpolant draw
          already carries. Needs only net_b, and no autograd at all. Requires
          gamma != "none".

        All three agree in expectation (E[b.s] = -E[div b] under p_t).

        Why "zdot" is unbiased without net_s: the exact score is the conditional
        expectation of the conditional score, ∇log p_t(x) = E[-z/gamma | x_t = x]
        (the identity behind denoising score matching). Since b(t, x_t) depends on
        the draw only through x_t, the tower property gives E[b·s] = E[b·(-z/gamma)],
        so swapping in the per-sample -z/gamma is exact for the TRUE score --
        whatever net_s has or has not learned drops out.

        The price is variance: a single branch carries a 1/gamma that diverges at
        both endpoints. Antithetic sampling removes it. Averaging the +z and -z
        branches at the same (t, x0, x1) gives

            (1/(2 gamma)) · z·(b(x_t^+) - b(x_t^-))  ->  z^T (grad b) z   as gamma -> 0,

        i.e. a central-difference Hutchinson trace of b with step gamma(t): "zdot"
        is a derivative-free "div", and both branches are individually unbiased
        (-z is as valid a draw as +z), so the average is too.

        Note the cancellation this leaves behind: the two branches differ by O(gamma),
        so at very small gamma their difference is lost to float precision. `eps`
        floors gamma through the time draw, and the training default (1e-6) is far
        too tight for this method -- construct with a larger `eps` (~1e-3) when
        using "zdot", especially with gamma="quad", whose gamma(eps) ~ eps is the
        most exposed of the schedules.
        """
        if method not in ("dot", "div", "zdot", "bdot"):
            raise ValueError(f"method must be 'dot', 'div', 'bdot', or 'zdot', got {method!r}")
        if method in ["zdot","bdot"] and self.gamma == "none":
            raise ValueError(
                "method='zdot'/'bdot' needs a non-zero latent schedule: with gamma='none' the "
                "interpolant carries no latent z and the conditional score -z/gamma is "
                "undefined. Use method='div' (or 'dot') instead."
            )
        t, t_b = self._draw_time(x1)

        if method == "zdot":
            # Antithetic pair; see the docstring for why it is needed and what the
            # gamma -> 0 limit is. Geometry stays confined to `_interpolant_sample`,
            # which supplies both x_t and the matching conditional score -z/gamma.
            z = self._noise_like(x1)
            ent = x1.new_zeros(x1.shape[0])
            for z_branch in (z, -z):
                x_t, _, s_target = self._interpolant_sample(t, x0, x1, z_branch)
                b = self.net_b(t_b, x_t)
                ent = ent - 0.5 * (b * s_target).flatten(1).sum(-1)
            return ent

        if method == "bdot":
            z = self._noise_like(x1)
            ent = x1.new_zeros(x1.shape[0])
            x_t, b_target, _ = self._interpolant_sample(t, x0, x1, z)
            s = self.net_s(t_b, x_t)
            return -(b_target * s).flatten(1).sum(-1)


        z = self._noise_like(x1) if self.gamma != "none" else torch.zeros_like(x1)
        x_t, _, _ = self._interpolant_sample(t, x0, x1, z)


        if method == "dot":
            b = self.net_b(t_b, x_t)
            s = self.net_s(t_b, x_t)
            return -(b * s).flatten(1).sum(-1)




        # if not caught above, use the divergence
        with torch.enable_grad():
            x_t_b = x_t.detach().requires_grad_(True)
            b = self.net_b(t_b, x_t_b)
            # Inference only: no backward through the divergence, so don't build
            # a second-order graph (which would accumulate over probes -> OOM).
            return self._divergence(b, x_t_b, create_graph=False, t=t_b)

    # --------------------------------------------------------------- sampler

    def _ent_incr(
        self,
        entropy: str,
        t_b: torch.Tensor,
        x: torch.Tensor,
        dt: float,
        b: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Per-step entropy increment at the point (t_b, x). Returns shape [B].

        - entropy="dot": -dt·(b·s), reusing precomputed `b`/`s` when given.
        - entropy="div": +dt·div(b), always recomputing net_b on a fresh graph
          (inference only, `create_graph=False`) so `self._divergence` can trace
          it; any passed `b`/`s` are ignored.
        """
        if entropy == "dot":
            if b is None:
                b = self.net_b(t_b, x)
            if s is None:
                s = self.net_s(t_b, x)
            return -dt * (b * s).flatten(1).sum(-1)
        # entropy == "div"
        with torch.enable_grad():
            x_g = x.detach().requires_grad_(True)
            b_g = self.net_b(t_b, x_g)
            return dt * self._divergence(b_g, x_g, create_graph=False, t=t_b)

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        n_steps: int = 100,
        eps: float = 0.0,
        method: str = "heun",
        entropy: str | None = None,
        return_traj: bool = False,
    ):
        """Integrate the learned dynamics from t=0 (x0) to t=1.

        One integrator for the whole family; the return shape is set by the flags
        (see the module docstring).

        Args:
            x0: [B, d] (or any per-sample shape), initial state.
            n_steps: integration steps.
            eps: diffusion coefficient. `eps == 0.0` selects the probability-flow
                ODE (net_b only); `eps > 0` selects the reverse-time SDE
                (Euler-Maruyama, drift `b + 0.5·eps^2·s`, diffusion `eps`). The
                SDE benefits from more steps than the ODE (was 200 vs 100).
            method: ODE integrator, "heun" (2 evals/step) or "euler" (1 eval/step).
                Ignored for the SDE (always Euler-Maruyama).
            entropy: entropy channel, one of None, "dot" (-∫b·s dt, needs net_s),
                or "div" (∫div(b) dt, needs only net_b). Heun evaluates the
                increment at the step midpoint, Euler/SDE at the current point.
            return_traj: if True, keep every intermediate state (and the running
                entropy) instead of only the endpoint.

        Returns:
            entropy None, return_traj False -> x1                         [B, d]
            entropy None, return_traj True  -> (traj, ts)
            entropy set,  return_traj False -> (x1, ent)                  ent [B]
            entropy set,  return_traj True  -> (traj, ent_traj, ts)
            where traj is [n_steps+1, B, d] (traj[0]==x0, traj[-1] the t=1 sample),
            ent_traj is [n_steps+1, B] (ent_traj[0]==0), and ts is [n_steps+1], the
            time grid from 0 to 1.
        """
        if method not in ("heun", "euler"):
            raise ValueError(f"unknown method {method!r}")
        if entropy not in (None, "dot", "div"):
            raise ValueError(f"entropy must be None, 'dot' or 'div', got {entropy!r}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")

        is_sde = eps > 0.0
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        ent = x.new_zeros(B) if entropy is not None else None

        traj = [x.clone()] if return_traj else None
        ent_traj = [ent.clone()] if (return_traj and entropy is not None) else None

        for t in ts:
            t_b = t.expand(B)
            if is_sde:
                b = self.net_b(t_b, x)
                s = self.net_s(t_b, x)
                if entropy is not None:
                    ent = ent + self._ent_incr(entropy, t_b, x, dt, b=b, s=s)
                drift = b + 0.5 * (eps ** 2) * s
                noise = (dt ** 0.5) * eps * self._noise_like(x)
                x = x + dt * drift + noise
            else:
                v1 = self.net_b(t_b, x)
                if method == "euler":
                    if entropy is not None:
                        ent = ent + self._ent_incr(entropy, t_b, x, dt, b=v1)
                    x = x + dt * v1
                else:
                    if entropy is not None:
                        x_mid = x + 0.5 * dt * v1
                        t_mid = (t + 0.5 * dt).expand(B)
                        ent = ent + self._ent_incr(entropy, t_mid, x_mid, dt)
                    x_pred = x + dt * v1
                    v2 = self.net_b((t + dt).expand(B), x_pred)
                    x = x + 0.5 * dt * (v1 + v2)

            if return_traj:
                traj.append(x.clone())
                if entropy is not None:
                    ent_traj.append(ent.clone())

        if return_traj:
            grid = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=x.device, dtype=x.dtype)
            if entropy is not None:
                return torch.stack(traj), torch.stack(ent_traj), grid
            return torch.stack(traj), grid
        if entropy is not None:
            return x, ent
        return x
