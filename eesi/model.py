"""EESI: general stochastic interpolant in Euclidean space.

Single class, no inheritance chain. Wraps two `EGNN` instances and exposes:

    .loss(x1, x0)                     training loss dict {"b": loss_b, "s": loss_s}.
    .sample_ode(x0, n_steps)          Heun ODE integration (uses net_b)
    .sample_sde(x0, n_steps, eps)     Euler-Maruyama (uses net_b + net_s)
    .sample_ode_entropy(x0, n_steps)  ODE + augmented entropy integral
    .sample_sde_entropy(x0, n_steps, eps) SDE + augmented entropy integral

The `*_traj` variants below mirror these but keep every intermediate state,
returning the full path (and the running entropy integral) so trajectories can
be visualised:

    .sample_ode_traj(x0, n_steps)          -> (traj, ts)
    .sample_sde_traj(x0, n_steps, eps)     -> (traj, ts)
    .sample_ode_entropy_traj(x0, n_steps)  -> (traj, ent_traj, ts)
    .sample_sde_entropy_traj(x0, n_steps, eps) -> (traj, ent_traj, ts)

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
# (gamma, gamma_dot). All outputs broadcast against x [B, d] from t [B, 1].


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


_GAMMAS = {
    "none": _gamma_none,
    "quad": _gamma_quad,
    "sqrt": _gamma_sqrt,
}


def _div_exact(s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    """Exact divergence of s w.r.t. x_t via Jacobian diagonal sum."""
    B = s.shape[0]
    s_flat = s.reshape(B, -1)
    nd = s_flat.shape[1]
    div = x_t.new_zeros(B)
    for i in range(nd):
        (g,) = torch.autograd.grad(
            s_flat[:, i].sum(), x_t,
            create_graph=True, retain_graph=True,
        )
        div = div + g.reshape(B, -1)[:, i]
    return div


def _div_hutchinson(s: torch.Tensor, x_t: torch.Tensor, n_probes: int) -> torch.Tensor:
    """Hutchinson trace estimator for div(s): E_v[v · grad_x(v·s)]."""
    B = s.shape[0]
    div = x_t.new_zeros(B)
    for _ in range(n_probes):
        v = torch.randn_like(s)
        (g,) = torch.autograd.grad(
            (v * s).sum(), x_t,
            create_graph=True, retain_graph=True,
        )
        div = div + (v * g).sum(dim=-1)
    return div / n_probes


def score_loss(s: torch.Tensor, div_s: torch.Tensor) -> torch.Tensor:
    """ISM loss E[||s||^2 + 2·div(s)]."""
    return (s.square().sum(dim=-1) + 2.0 * div_s).mean()


class EESI(nn.Module):
    """General stochastic interpolant in Euclidean space.

    Args:
        net_b: Module for the velocity field b(t, x). Forward: (t, x) -> [B, d].
        net_s: Module for the score field s(t, x). Forward: (t, x) -> [B, d].
        d: spatial dimension.
        path: interpolant path (alpha, beta), one of `_PATHS`:
            "linear" (default), "trig", "encdec".
        gamma: latent-noise schedule, one of `_GAMMAS`:
            "none", "quad" (default), "sqrt". gamma(0)=gamma(1)=0. With
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
        d: int,
        path: str = "linear",
        gamma: str = "quad",
        gamma_scale: float = 1.0,
        eps: float = 1e-5,
        score_div_method: str = "hutchinson",
        n_hutchinson_probes: int = 1,
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

    def _divergence(self, s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        if self.score_div_method == "exact":
            return _div_exact(s, x_t)
        return _div_hutchinson(s, x_t, self.n_hutchinson_probes)

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
        B = x1.shape[0]
        device = x1.device

        # Keep t in [eps, 1-eps] so gamma'(t) stays finite at the endpoints.
        t = torch.rand((B, 1), device=device, dtype=x1.dtype)
        t = t * (1.0 - 2.0 * self.eps) + self.eps
        t_b = t.view(B)

        alpha, beta, alpha_dot, beta_dot = self._path(t)
        I_t = alpha * x0 + beta * x1
        v_det = alpha_dot * x0 + beta_dot * x1

        if self.gamma == "none":
            # Deterministic interpolant: implicit score matching for s.
            x_t = I_t
            b = self.net_b(t_b, x_t)
            loss_b = (b - v_det).square().mean()

            with torch.enable_grad():
                x_t_s = x_t.detach().requires_grad_(True)
                s = self.net_s(t_b, x_t_s)
                div_s = self._divergence(s, x_t_s)
                loss_s = score_loss(s, div_s)

            return {"b": loss_b, "s": loss_s}

        # Non-zero gamma: antithetic denoising losses (no divergence estimate).
        g, g_dot = self._gamma(t, self.eps)
        g = self.gamma_scale * g
        g_dot = self.gamma_scale * g_dot

        z = torch.randn_like(x1)
        gz = g * z
        x_plus = I_t + gz
        x_minus = I_t - gz

        # Drift: antithetic average of E[1/2||b||^2 - (v_det + g'·z)·b]. The
        # noise term is written as g'·z·(b_+ - b_-) so the large-g' factor
        # multiplies the O(g) difference b_+ - b_- (finite, no cancellation).
        b_plus = self.net_b(t_b, x_plus)
        b_minus = self.net_b(t_b, x_minus)
        quad_b = 0.25 * (b_plus.square() + b_minus.square()).sum(dim=-1)
        lin_det = 0.5 * (v_det * (b_plus + b_minus)).sum(dim=-1)
        lin_noise = 0.5 * (g_dot * z * (b_plus - b_minus)).sum(dim=-1)
        loss_b = (quad_b - lin_det - lin_noise).mean()

        # Score: antithetic average of E[1/2||s||^2 + (s·z)/g]. The cross term
        # is (s_+ - s_-)·z / (2g): the O(g) difference divided by g stays finite
        # as g -> 0. A tiny clamp on the divisor is defensive insurance.
        s_plus = self.net_s(t_b, x_plus)
        s_minus = self.net_s(t_b, x_minus)
        quad_s = 0.25 * (s_plus.square() + s_minus.square()).sum(dim=-1)
        g_div = g.view(B).clamp_min(1e-12)
        cross_s = ((s_plus - s_minus) * z).sum(dim=-1) / (2.0 * g_div)
        loss_s = (quad_s + cross_s).mean()

        return {"b": loss_b, "s": loss_s}

    # --------------------------------------------------------------- samplers

    @torch.no_grad()
    def sample_ode(
        self,
        x0: torch.Tensor,
        n_steps: int = 100,
        method: str = "heun",
    ) -> torch.Tensor:
        """Integrate the learned ODE from t=0 (x0) to t=1.

        Args:
            x0: [B, d], initial state.
            n_steps: integration steps.
            method: "heun" (2 evals/step) or "euler" (1 eval/step).

        Returns:
            x1 [B, d].
        """
        if method not in ("heun", "euler"):
            raise ValueError(f"unknown method {method!r}")
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        for t in ts:
            t_b = t.expand(B)
            v1 = self.net_b(t_b, x)
            if method == "euler":
                x = x + dt * v1
            else:
                x_pred = x + dt * v1
                v2 = self.net_b((t + dt).expand(B), x_pred)
                x = x + 0.5 * dt * (v1 + v2)
        return x

    @torch.no_grad()
    def sample_sde(
        self,
        x0: torch.Tensor,
        n_steps: int = 200,
        eps: float = 0.1,
    ) -> torch.Tensor:
        """Euler-Maruyama from t=0 (x0) to t=1.

        Drift is `b + 0.5 * eps^2 * s`; diffusion is `eps`.

        Args:
            x0: [B, d], initial state.
            n_steps: integration steps.
            eps: diffusion coefficient.

        Returns:
            x1 [B, d].
        """
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        for t in ts:
            t_b = t.expand(B)
            b = self.net_b(t_b, x)
            s = self.net_s(t_b, x)
            drift = b + 0.5 * (eps ** 2) * s
            noise = (dt ** 0.5) * eps * torch.randn_like(x)
            x = x + dt * drift + noise
        return x

    @torch.no_grad()
    def sample_ode_entropy(
        self,
        x0: torch.Tensor,
        n_steps: int = 100,
        method: str = "heun",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate the ODE from t=0 to t=1, tracking entropy change.

        Augments the trajectory with the running integral

            ent[b] = -∫₀¹ b(x_t, t) · s(x_t, t) dt

        Args:
            x0: [B, d], initial state.
            n_steps: integration steps.
            method: "heun" (entropy at midpoint) or "euler" (entropy at current point).

        Returns:
            (x1, ent) — x1 is [B, d]; ent is [B].
        """
        if method not in ("heun", "euler"):
            raise ValueError(f"unknown method {method!r}")
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        ent = x.new_zeros(B)
        for t in ts:
            t_b = t.expand(B)
            v1 = self.net_b(t_b, x)
            if method == "euler":
                s = self.net_s(t_b, x)
                ent = ent - dt * (v1 * s).sum(dim=-1)
                x = x + dt * v1
            else:
                x_mid = x + 0.5 * dt * v1
                t_mid = (t + 0.5 * dt).expand(B)
                b_mid = self.net_b(t_mid, x_mid)
                s_mid = self.net_s(t_mid, x_mid)
                ent = ent - dt * (b_mid * s_mid).sum(dim=-1)
                x_pred = x + dt * v1
                v2 = self.net_b((t + dt).expand(B), x_pred)
                x = x + 0.5 * dt * (v1 + v2)
        return x, ent

    @torch.no_grad()
    def sample_sde_entropy(
        self,
        x0: torch.Tensor,
        n_steps: int = 200,
        eps: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler-Maruyama from t=0 to t=1, tracking entropy change.

        Augments the SDE trajectory with the running integral

            ent[b] = -∫₀¹ b(x_t, t) · s(x_t, t) dt

        Args:
            x0: [B, d], initial state.
            n_steps: integration steps.
            eps: diffusion coefficient.

        Returns:
            (x1, ent) — x1 is [B, d]; ent is [B].
        """
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        ent = x.new_zeros(B)
        for t in ts:
            t_b = t.expand(B)
            b = self.net_b(t_b, x)
            s = self.net_s(t_b, x)
            ent = ent - dt * (b * s).sum(dim=-1)
            drift = b + 0.5 * (eps ** 2) * s
            noise = (dt ** 0.5) * eps * torch.randn_like(x)
            x = x + dt * drift + noise
        return x, ent

    # ------------------------------------------ trajectory-returning samplers

    @torch.no_grad()
    def sample_ode_traj(
        self,
        x0: torch.Tensor,
        n_steps: int = 100,
        method: str = "heun",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate the learned ODE from t=0 to t=1, keeping the whole path.

        Same integrator as :meth:`sample_ode`, but every intermediate state is
        recorded.

        Args:
            x0: [B, d], initial state.
            n_steps: integration steps.
            method: "heun" (2 evals/step) or "euler" (1 eval/step).

        Returns:
            (traj, ts) — traj [n_steps+1, B, d] with traj[0] == x0 and traj[-1]
            the t=1 sample; ts [n_steps+1] the time grid from 0 to 1.
        """
        if method not in ("heun", "euler"):
            raise ValueError(f"unknown method {method!r}")
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        traj = [x.clone()]
        for t in ts:
            t_b = t.expand(B)
            v1 = self.net_b(t_b, x)
            if method == "euler":
                x = x + dt * v1
            else:
                x_pred = x + dt * v1
                v2 = self.net_b((t + dt).expand(B), x_pred)
                x = x + 0.5 * dt * (v1 + v2)
            traj.append(x.clone())
        grid = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=x.device, dtype=x.dtype)
        return torch.stack(traj), grid

    @torch.no_grad()
    def sample_sde_traj(
        self,
        x0: torch.Tensor,
        n_steps: int = 200,
        eps: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler-Maruyama from t=0 to t=1, keeping the whole path.

        Same integrator as :meth:`sample_sde`, but every intermediate state is
        recorded.

        Args:
            x0: [B, d], initial state.
            n_steps: integration steps.
            eps: diffusion coefficient.

        Returns:
            (traj, ts) — traj [n_steps+1, B, d] with traj[0] == x0 and traj[-1]
            the t=1 sample; ts [n_steps+1] the time grid from 0 to 1.
        """
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        traj = [x.clone()]
        for t in ts:
            t_b = t.expand(B)
            b = self.net_b(t_b, x)
            s = self.net_s(t_b, x)
            drift = b + 0.5 * (eps ** 2) * s
            noise = (dt ** 0.5) * eps * torch.randn_like(x)
            x = x + dt * drift + noise
            traj.append(x.clone())
        grid = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=x.device, dtype=x.dtype)
        return torch.stack(traj), grid

    @torch.no_grad()
    def sample_ode_entropy_traj(
        self,
        x0: torch.Tensor,
        n_steps: int = 100,
        method: str = "heun",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate the ODE, recording both the path and the running entropy.

        Trajectory-returning counterpart of :meth:`sample_ode_entropy`. The
        entropy channel is the running integral

            ent(t) = -∫₀ᵗ b(x_u, u) · s(x_u, u) du,

        so ``ent_traj[k]`` is the entropy change accumulated up to ``ts[k]``,
        with ``ent_traj[0] == 0``.

        Args:
            x0: [B, d], initial state.
            n_steps: integration steps.
            method: "heun" (entropy at midpoint) or "euler" (entropy at current point).

        Returns:
            (traj, ent_traj, ts) — traj [n_steps+1, B, d]; ent_traj [n_steps+1, B];
            ts [n_steps+1] the time grid from 0 to 1.
        """
        if method not in ("heun", "euler"):
            raise ValueError(f"unknown method {method!r}")
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        ent = x.new_zeros(B)
        traj = [x.clone()]
        ent_traj = [ent.clone()]
        for t in ts:
            t_b = t.expand(B)
            v1 = self.net_b(t_b, x)
            if method == "euler":
                s = self.net_s(t_b, x)
                ent = ent - dt * (v1 * s).sum(dim=-1)
                x = x + dt * v1
            else:
                x_mid = x + 0.5 * dt * v1
                t_mid = (t + 0.5 * dt).expand(B)
                b_mid = self.net_b(t_mid, x_mid)
                s_mid = self.net_s(t_mid, x_mid)
                ent = ent - dt * (b_mid * s_mid).sum(dim=-1)
                x_pred = x + dt * v1
                v2 = self.net_b((t + dt).expand(B), x_pred)
                x = x + 0.5 * dt * (v1 + v2)
            traj.append(x.clone())
            ent_traj.append(ent.clone())
        grid = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=x.device, dtype=x.dtype)
        return torch.stack(traj), torch.stack(ent_traj), grid

    @torch.no_grad()
    def sample_sde_entropy_traj(
        self,
        x0: torch.Tensor,
        n_steps: int = 200,
        eps: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Euler-Maruyama, recording both the path and the running entropy.

        Trajectory-returning counterpart of :meth:`sample_sde_entropy`. The
        entropy channel is the running integral

            ent(t) = -∫₀ᵗ b(x_u, u) · s(x_u, u) du,

        with ``ent_traj[0] == 0``.

        Args:
            x0: [B, d], initial state.
            n_steps: integration steps.
            eps: diffusion coefficient.

        Returns:
            (traj, ent_traj, ts) — traj [n_steps+1, B, d]; ent_traj [n_steps+1, B];
            ts [n_steps+1] the time grid from 0 to 1.
        """
        x = x0.clone()
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, steps=n_steps, device=x.device, dtype=x.dtype)
        B = x.shape[0]
        ent = x.new_zeros(B)
        traj = [x.clone()]
        ent_traj = [ent.clone()]
        for t in ts:
            t_b = t.expand(B)
            b = self.net_b(t_b, x)
            s = self.net_s(t_b, x)
            ent = ent - dt * (b * s).sum(dim=-1)
            drift = b + 0.5 * (eps ** 2) * s
            noise = (dt ** 0.5) * eps * torch.randn_like(x)
            x = x + dt * drift + noise
            traj.append(x.clone())
            ent_traj.append(ent.clone())
        grid = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=x.device, dtype=x.dtype)
        return torch.stack(traj), torch.stack(ent_traj), grid
