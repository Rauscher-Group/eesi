"""EESI: general stochastic interpolant in Euclidean space.

Single class, no inheritance chain. Wraps two field networks (`net_b`, `net_s` --
any module with the right signature; see `eesi.models`) and exposes:

    .loss(x1, x0)                     training loss dict {"b": loss_b, "s": loss_s}.
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
learned dynamics); `method="div"` traces net_b, `method="dot"` uses -b.s.

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

from .models.lj13_dynamics import divergence as _subspace_divergence


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


def _min_image(d: torch.Tensor) -> torch.Tensor:
    """Minimum-image angle difference, wrapped into (-pi, pi].

    Matches the convention in `eesi.datasets.xy`:
    d - 2*pi*rint(d / 2*pi). Used by `xyEESI` to interpolate along the shortest
    geodesic on the periodic angle manifold (S^1)^N.
    """
    two_pi = 2.0 * math.pi
    return d - two_pi * torch.round(d / two_pi)


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

    def loss(
        self,
        x1: torch.Tensor,
        x0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Drift and score losses for one batch.

        Returns {"b": loss_b, "s": loss_s}. Caller sums for backward:

            losses = model.loss(x1, x0)
            (losses["b"] + losses["s"]).backward()

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

            return {"b": loss_b, "s": loss_s}

        # Non-zero gamma: antithetic denoising. Sample z once and evaluate the
        # per-branch denoising losses at +z and -z; the -z partner cancels the
        # endpoint 1/gamma and gamma' singularities. Only `_interpolant_sample`
        # is topology-aware, so periodicity handling lives entirely there.
        z = self._noise_like(x1)
        loss_b = x1.new_zeros(())
        loss_s = x1.new_zeros(())
        for z_branch in (z, -z):
            x_t, b_target, s_target = self._interpolant_sample(t, x0, x1, z_branch)
            b = self.net_b(t_b, x_t)
            s = self.net_s(t_b, x_t)
            loss_b = loss_b + 0.5 * _denoising_loss(b, b_target).mean()
            loss_s = loss_s + 0.5 * _denoising_loss(s, s_target).mean()

        return {"b": loss_b, "s": loss_s}

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

        `method` selects the accumulator (same "dot"/"div" vocabulary as the
        `entropy` flag of :meth:`sample`):

        - "div" (default): the divergence of the velocity field (net_b) sampled
          along the interpolant, handled by `self._divergence` (Hutchinson by
          default; the exact subspace trace in `LJ13EESI`). Needs only net_b.
        - "dot": the accumulator -(b \cdot s) along the interpolant. Needs a
          learned score in addition to the velocity; no divergence is required,
          though errors in the score now compound those in the velocity.

        The two agree in expectation (E[b.s] = -E[div b] under p_t).
        """
        if method not in ("dot", "div"):
            raise ValueError(f"method must be 'dot' or 'div', got {method!r}")
        t, t_b = self._draw_time(x1)

        z = self._noise_like(x1) if self.gamma != "none" else torch.zeros_like(x1)
        x_t, _, _ = self._interpolant_sample(t, x0, x1, z)

        if method == "dot":
            b = self.net_b(t_b, x_t)
            s = self.net_s(t_b, x_t)
            return -(b * s).flatten(1).sum(-1)

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


class xyEESI(EESI):
    """Stochastic interpolant on the periodic angle manifold (S^1)^N.

    Identical to `EESI` in every objective (drift/score losses, entropy
    estimators, samplers) but with a periodicity-aware interpolant, achieved by
    overriding the single geometry-aware hook `_interpolant_sample`. Rather than
    the Euclidean straight line `alpha·x0 + beta·x1`, which can cross the 0/2*pi
    seam and produce spuriously large velocity targets, `xyEESI` interpolates
    along the minimum-image geodesic. With `d = min_image(x1 - x0)` (each
    component wrapped into (-pi, pi]):

        x_t      = wrap(x0 + beta(t)·d + gamma(t)·z)   # position, wrapped onto (S^1)^N
        b_target = beta'(t)·d + gamma'(t)·z            # tangent-space velocity dx/dt
        s_target = -z / gamma(t)                       # tangent-space conditional score

    The latent noise `z` is added in the tangent space and the sampled point is
    wrapped back onto the manifold before it reaches the network. The path's
    `alpha` is unused: the geodesic is parameterised by `beta`, which runs 0 -> 1
    for the `linear`/`trig`/`trig2` paths (`linear` gives exactly `x0 + t·d`).

    The predicted velocity/score live in the tangent space R^N, so pair `xyEESI`
    with a periodicity-aware network such as `XYChainGNN`, whose output is a
    per-node tangent scalar. Note the endpoints are only recovered modulo 2*pi
    (x_t at t=1 equals x1 up to wrapping), which is exactly the manifold identity.

    The ODE/SDE samplers are inherited unchanged; because a periodicity-aware
    network is invariant to wrapping its input, integrated states may drift off
    (-pi, pi] but represent the same manifold points — wrap the final samples if a
    canonical representative is wanted.
    """

    def _interpolant_sample(
        self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor
    ):
        """Geodesic interpolant position and tangent-space targets (see class doc)."""
        self._assert_matched(x0, x1)
        _, beta, _, beta_dot = self._path(t)
        g, g_dot = self._scaled_gamma(t)
        d = _min_image(x1 - x0)
        x_t = _min_image(x0 + beta * d + g * z)
        b_target = beta_dot * d + g_dot * z
        s_target = -z / g.clamp_min(1e-12)
        return x_t, b_target, s_target


class LJ13EESI(EESI):
    """Stochastic interpolant for LJ13-type point clouds on the COM-free subspace.

    The LJ13 analogue of `xyEESI`, for states of shape (B, N, 3) -- N-agnostic, so
    the same class serves 13 particles today and larger clusters later. The
    interpolant geometry is the plain Euclidean straight line (the mean-zero
    subspace is flat), so `_interpolant_sample` is inherited unchanged. The ONE
    specialisation is that every Gaussian this class draws must live on that
    subspace, which is achieved by overriding the single `_noise_like` hook.

    Why that is the whole story (see plans/LJ13_SI_PLAN.md and the `eesi.ot`
    docstring): LJ13 lives on V = {x : sum_i x_i = 0}, where both the prior p0 (a
    COM-free Gaussian) and the target p1 (the LJ13 Boltzmann law) are supported.
    The base x0 and the latent z are then the same kind of object -- centered
    Gaussians in a *flat* subspace -- so no tangent-space / exp-map machinery is
    needed, unlike a curved manifold. Centering the latent is what keeps the whole
    path x_t = alpha x0 + beta x1 + gamma z on V (x0, x1, z all mean-zero, the map
    linear); an off-subspace z would inject a spurious center-of-mass at
    intermediate t even with mean-zero endpoints.

    The latent is kept INDEPENDENT of the (OT-coupled) base on purpose: the
    one-sided equivalence that would let one fold alpha x0 + gamma z into a single
    Gaussian only holds for an uncoupled base, and `equivariant_ot_couple` aligns
    x0 to x1. With z independent, the antithetic denoising score stays exact.

    Divergence/entropy: `entropy_estimate(method="div")` (and `sample(entropy="div")`)
    trace net_b by Hutchinson with the
    same centered draw, so the probes are v ~ N(0, P) and E[v^T J v] = tr(P J) is
    the divergence on the DOF = (N-1)*3 subspace -- the estimator analogue of
    `eesi.models.lj13_dynamics.divergence`. Pair with two `LJ13Dynamics` fields,
    whose output is already mean-free.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force ISM off, regardless of what the caller passed. `learn_score` only
        # matters for gamma="none" (the base uses it to gate implicit score
        # matching), and ISM needs a differentiable divergence -- but this class
        # traces net_b with `_subspace_divergence`, which runs under no_grad (see
        # `_divergence`). Learning the score here would silently backprop through a
        # detached graph, so it is disabled outright.
        self.learn_score = False

    def _divergence(self, s: torch.Tensor, x_t: torch.Tensor, create_graph: bool = True,
                    t: torch.Tensor | None = None) -> torch.Tensor:
        """Exact subspace divergence of net_b, via `lj13_dynamics.divergence`.

        Traces the velocity field over the DOF = (N-1)*3 COM-free directions with
        forward-mode jvp -- the exact analogue of the base class's Hutchinson
        estimator, and the same routine used for the free-energy log-det. It
        re-evaluates net_b at `t`, so `s` (the precomputed output) is unused, as is
        `create_graph`: the estimator is `@torch.no_grad()` and never differentiable
        (which is why `learn_score` is forced off; see `__init__`). The live
        callers are `entropy_estimate(method="div")` and `sample(entropy="div")`.
        """
        return _subspace_divergence(self.net_b, t, x_t)

    def _noise_like(self, ref: torch.Tensor) -> torch.Tensor:
        """A COM-free standard-normal draw shaped like `ref` (B, N, 3).

        Removes the per-configuration center of mass (mean over the particle axis),
        projecting the draw onto the mean-zero subspace. Serves every Gaussian the
        base class samples: the latent z, the SDE diffusion term, and the Hutchinson
        probes.
        """
        g = torch.randn_like(ref)
        return g - g.mean(dim=-2, keepdim=True)
