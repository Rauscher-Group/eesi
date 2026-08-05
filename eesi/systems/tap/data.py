"""The TAP system itself: reference data, the semiflexible prior, subspace geometry.

Everything here is a closed-form fact about the system, true whether or not a model
has ever been trained on it:

    REF_DATA_PATH, load_ref_data    the reference trajectory samples
    sample_prior, log_prior         the semiflexible chain base distribution
    sample_bond_lengths             the radial law the prior is built from
    sample_bond_cosines             the bending law it is built from
    bond_moments, bond_vectors      the one place the bond-law convention lives
    angle_moments, bond_cosines     the one place the bending convention lives
    prior_entropy                   S[p0], the reference point for an absolute entropy
    end_to_end_mean_sq              E[Re^2] under the prior, for calibration
    solve_cos_theta_0               invert it: the angle that matches a target E[Re^2]
    DOF, dof, subspace_dirs         geometry of the tail-anchored subspace
    anchor                          projection onto that subspace
    end_to_end_sq, gyration_sq      structural observables, for tests and notebooks
    rouse_modes, rouse_mode_moments backbone structure at every wavelength at once

Anything that needs a velocity field to mean anything lives in
`eesi.systems.tap.dynamics`; the dependency runs one way, dynamics -> data.

NO ENERGY FUNCTION LIVES HERE, and that is deliberate. A tangentially active polymer
is driven by a force along the local chain tangent, which is NOT the gradient of any
potential: the steady state is not a Boltzmann distribution, there is no U(x) to
evaluate, and consequently none of LJ13's importance-weighted free-energy machinery
(`delta_energy`, `free_energy`) has an analogue. That is the whole reason the system
is interesting here -- the interpolant estimates an entropy that no reweighting
scheme can reach.

The prior below does carry a harmonic bond potential, and that is not a contradiction:
it is the base distribution the flow starts FROM, chosen because it resembles the
target, not a claim about what the target is. p0's Boltzmann factor is a fact about
p0 alone. If a conservative piece of the DYNAMICS is ever wanted as a diagnostic
(excluded volume, say), it belongs here too, and must be labelled just as carefully --
as a component of the drive, never as a target density.


Geometry
--------
Configurations are gauge-fixed by pinning the tail (particle 0) at the origin, so
they live on the affine subspace

    V = {x in R^(N x 3) : x_0 = 0},    dim V = DOF = 3 (N - 1)

whose tangent space is {v : v_0 = 0}. `anchor` is the projection onto it, and serves
both roles: applied to points it shifts the tail to the origin, applied to tangents it
subtracts the tail's velocity, v_i -> v_i - v_0 (see
`eesi.systems.tap.dynamics.TAPDynamics`, which uses it for exactly that).

Note the contrast with LJ13, which removes the center of mass instead. Both fix the
3 translational directions, but they fix them differently, and the difference is
load-bearing for the OT coupling: O(3) here acts about the PINNED TAIL, not about
the centroid, so `eesi.systems.tap.ot` must not center its inputs.


The prior: a semiflexible chain
------------------------------
The base distribution is the real TAP polymer with the activity and the excluded volume
switched off: harmonic bonds of stiffness k about a finite equilibrium length b, plus a
harmonic BENDING potential between successive bonds, with positions the running sum.

    x_0 = 0,  x_k = sum_{j <= k} bond_j

Four parameters, all properties of the system and none with a default. Two govern the
bond magnitudes Q = |bond|, which carry the Q^2 surface Jacobian:

    p(Q) = Q^2 exp(-(k/2) (Q - b)^2) / Z1,   Q > 0
    Z1   = int_0^inf Q^2 exp(-(k/2) (Q - b)^2) dQ

which is NOT a Gaussian: it is peaked near b, not at the origin, which is the whole
point of the finite equilibrium length. Two more govern the angle between successive
bonds via gamma (cos theta - cos theta_0)^2. Writing u = cos theta = u_k . u_{k-1}, the
solid-angle element sin theta dtheta dphi = du dphi makes the azimuth uniform and leaves
u a GAUSSIAN TRUNCATED TO [-1, 1]:

    p(u) = exp(-gamma (u - u_0)^2) / Z_ang,   u_0 = cos theta_0 in [-1, 1]
    Z_ang = int_{-1}^{1} exp(-gamma (u - u_0)^2) du     <- a difference of error functions

`bond_moments` and `angle_moments` evaluate Z1, Z_ang and the moments each needs in
closed form, and are the single source of truth for their respective conventions --
`log_prior`, `prior_entropy` and `end_to_end_mean_sq` all route through them, and the
two samplers are tested against them, so they cannot disagree.

Why bending: with iid bond orientations the chain's size is fixed by the bond length
alone, and the reference data's E[Re^2] is 2.2x what its own bond statistics predict --
its bonds are orientationally correlated. gamma sets the stiffness and u_0 tunes the
size, so the bond law can stay physical while E[Re^2] is matched. See
`solve_cos_theta_0`.

NOTE THE 4 pi / 2 pi ASYMMETRY. The first bond's orientation is isotropic and normalizes
over the full sphere; each of the other N-2 is conditioned on its predecessor, and there
the du integral has already been absorbed into Z_ang, leaving only the azimuth:

    bond 1:    exp(-(k/2)(Q-b)^2) / (4 pi Z1)
    bond j>=2: exp(-(k/2)(Q-b)^2) exp(-gamma (u_j - u_0)^2) / (2 pi Z1 Z_ang)

so there are N-1 radial factors but only N-2 angular ones. Getting this wrong is a clean
constant offset that no relative comparison would reveal, which is why the entropy test
exists.

`log_prior` is exact and needs no Jacobian correction. The bond-to-position map is the
cumulative sum, which in the bond basis is unit lower triangular, so |det J| = 1; and
placing each bond in the frame whose z-axis is its predecessor is a ROTATION, so that is
unit too. The density therefore transfers unchanged:

    log p0(x) = -sum_j (k/2) (Q_j - b)^2 - gamma sum_{j>=2} (u_j - u_0)^2
                - log(4 pi Z1) - (N-2) log(2 pi Z1 Z_ang)

Azimuthal averaging makes the bond DIRECTIONS an exact Markov chain, so correlations
decay geometrically, <u_i . u_j> = <u>^|i-j| -- the freely-rotating-chain result, and the
sharpest available test of the sampler's frame construction.

Two nested special cases, both exact and both tested, so an older run can be reproduced:

    gamma = 0            uniform on [-1, 1]: the freely-jointed harmonic-bond chain
    gamma = 0 and b = 0  the ideal (Gaussian) chain, sigma^2 = 1/k; the original
                         ReSqr parameterization is this slice with k = 3(N-1)/ReSqr

The prior is still O(3)-invariant: the first bond is isotropic and every later one is
defined by dot products against its predecessor, so a global rotation leaves the density
unchanged. That is what makes the alignment layer of `eesi.systems.tap.ot`
marginal-preserving, and it is the load-bearing property -- not Gaussianity, not iid
bonds, neither of which the coupling or the interpolant ever assumed.
"""
from __future__ import annotations

import math
import os
import pathlib
from functools import lru_cache

import numpy as np
import torch

N_DEFAULT = 20      # the usual chain length
N_DIMS = 3

# Reference trajectories are kept OUT of the package, in a top-level `data/`, exactly
# as the LJ13 samples are, and read through the same `EESI_DATA_DIR` override so one
# environment variable relocates every system's data at once.
DATA_DIR = pathlib.Path(
    os.environ.get("EESI_DATA_DIR", pathlib.Path(__file__).resolve().parents[3] / "data")
)
# Provisional: the generator that writes this file is external, so the name is a
# placeholder until it is pinned down. `load_ref_data` takes an explicit path, and
# nothing else in the package depends on this constant.
REF_DATA_PATH = DATA_DIR / "tap_N20_Pe0.npy"


# --- subspace geometry ------------------------------------------------------


def dof(n_particles: int = N_DEFAULT, n_dims: int = N_DIMS) -> int:
    """Dimension of the tail-anchored subspace, (N - 1) * d."""
    return (n_particles - 1) * n_dims


DOF = dof()  # 57, for the default 20-particle chain


def anchor(x: torch.Tensor) -> torch.Tensor:
    """Shift the tail (particle 0) to the origin. x: (..., N, d).

    The TAP analogue of `eesi.ot.center`. Idempotent, and a no-op on data that is
    already anchored up to float drift.
    """
    return x - x[..., :1, :]


def subspace_dirs(n_particles: int = N_DEFAULT, n_dims: int = N_DIMS) -> torch.Tensor:
    """Orthonormal basis of the tail-anchored subspace as (DOF, N, d) ambient tangents.

    Just the standard basis with particle 0's rows dropped -- the tangent space
    {v : v_0 = 0} is already axis-aligned, so unlike LJ13 (whose mean-zero subspace
    needs an eigendecomposition of the centering projector) there is nothing to
    diagonalize. Consumed by `eesi.systems.tap.dynamics.divergence`.
    """
    d = dof(n_particles, n_dims)
    dirs = torch.zeros(d, n_particles, n_dims)
    idx = torch.arange(d)
    dirs[idx, idx // n_dims + 1, idx % n_dims] = 1.0
    return dirs


# --- the prior --------------------------------------------------------------


def _require_3d(n_dims: int) -> None:
    """The prior is 3D-only, and says so loudly.

    The Q^2 radial Jacobian and the closed forms in `bond_moments` are specific to
    d = 3; a d-generic version would need Q^(d-1) and incomplete-gamma moments, and
    has no consumer. Everything ELSE in this module is d-generic, so the check lives
    here rather than at import time.
    """
    if n_dims != 3:
        raise ValueError(
            f"the semiflexible prior is defined for n_dims=3 only, got {n_dims}. "
            f"The radial law p(Q) ~ Q^2 exp(-(k/2)(Q-b)^2) carries the 3D surface "
            f"Jacobian; the subspace geometry in this module is d-generic, the prior "
            f"is not."
        )


def _gauss_tail_moments(a: float, j_max: int) -> list[float]:
    """G_j = int_{-a}^inf u^j exp(-u^2/2) du, for j = 0 .. j_max.

    From G_0 = sqrt(pi/2)(1 + erf(a/sqrt(2))) and G_1 = exp(-a^2/2) by the
    integration-by-parts recursion G_j = (j-1) G_{j-2} + (-a)^(j-1) exp(-a^2/2).

    Every term is positive for a >= 0, so the recursion is free of cancellation. The
    large-a limit is also clean: exp(-a^2/2) underflows to zero, leaving G_j at its
    untruncated Gaussian value, which is the right answer.
    """
    e = math.exp(-0.5 * a * a)
    g = [math.sqrt(0.5 * math.pi) * (1.0 + math.erf(a / math.sqrt(2.0))), e]
    for j in range(2, j_max + 1):
        g.append((j - 1) * g[j - 2] + (-a) ** (j - 1) * e)
    return g[:j_max + 1]


def bond_moments(k: float, b: float) -> tuple[float, float, float]:
    """(Z1, E[Q], E[Q^2]) for the radial bond law p(Q) ~ Q^2 exp(-(k/2)(Q-b)^2).

    THE single source of truth for the bond convention. Z1 is the unnormalized
    radial integral int_0^inf Q^2 exp(-(k/2)(Q-b)^2) dQ, so the normalized 3D bond
    density is exp(-(k/2)(|Q|-b)^2) / (4 pi Z1) -- the 4 pi is the sphere's surface
    area and is NOT folded in here, because keeping Z1 purely radial is what makes
    the moment formulas below read directly off the same sums.

    Substituting Q = b + sigma u with sigma = 1/sqrt(k) and a = b/sigma turns every
    integral into a truncated Gaussian moment:

        int_0^inf Q^m . Q^2 exp(-(k/2)(Q-b)^2) dQ
            = sigma . sum_j C(m+2, j) b^(m+2-j) sigma^j G_j

    with G_j from `_gauss_tail_moments`. So one recursion gives the normalizer and
    both moments; there is no quadrature and no special-casing of b = 0.

    This is exact, not asymptotic. Two identities worth knowing, both used as tests:
    E[Q^2] = b E[Q] + 3/k (integration by parts), and at b = 0 the law collapses to
    Maxwell-Boltzmann with E[Q^2] = 3/k = 3 sigma^2, the old ideal chain.
    """
    if not k > 0.0:
        raise ValueError(f"the bond spring constant k must be positive, got {k}")
    if b < 0.0:
        raise ValueError(f"the bond equilibrium length b must be non-negative, got {b}")

    sigma = 1.0 / math.sqrt(k)
    g = _gauss_tail_moments(b / sigma, 4)
    m = [sigma * math.fsum(math.comb(p, j) * b ** (p - j) * sigma ** j * g[j]
                           for j in range(p + 1))
         for p in (2, 3, 4)]
    return m[0], m[1] / m[0], m[2] / m[0]


def angle_moments(gamma: float, cos_theta_0: float) -> tuple[float, float, float]:
    """(Z_ang, E[u], E[(u-u_0)^2]) for p(u) ~ exp(-gamma (u - u_0)^2) on u in [-1, 1].

    The angular counterpart of `bond_moments`, and the single source of truth for the
    bending convention. u = cos theta between successive bonds; the truncation to
    [-1, 1] is what makes this a genuine cosine rather than an unbounded Gaussian, and
    it is not a detail -- at small gamma the law is nowhere near normal.

    Standard truncated-normal formulas with sigma = 1/sqrt(2 gamma) and standardized
    limits a = (-1 - u_0)/sigma, c = (1 - u_0)/sigma:

        Z_ang        = sigma sqrt(2 pi) (Phi(c) - Phi(a))       <- erf difference
        E[u]         = u_0 + sigma (phi(a) - phi(c)) / (Phi(c) - Phi(a))
        E[(u-u_0)^2] = sigma^2 [1 + (a phi(a) - c phi(c)) / (Phi(c) - Phi(a))]

    Note the third is the second moment about u_0, the POTENTIAL's centre, not about
    the distribution's own mean -- that is what the entropy needs, and the two differ
    whenever the truncation bites.

    gamma = 0 is a real special case, branched rather than taken as a limit: the law is
    uniform on [-1, 1], giving Z_ang = 2, E[u] = 0 and E[(u-u_0)^2] = 1/3 + u_0^2. That
    is the freely-jointed chain, and it must come out exact.

    `cos_theta_0` is required to lie in [-1, 1]. Outside it the untruncated Gaussian
    sits entirely beyond the interval, Phi(c) - Phi(a) underflows to zero, and every
    formula above divides by it. u_0 = +-1 is the maximum-alignment setting; for a
    stiffer chain raise gamma, do not push u_0 out of range.
    """
    if gamma < 0.0:
        raise ValueError(f"the bending constant gamma must be non-negative, got {gamma}")
    if not -1.0 <= cos_theta_0 <= 1.0:
        raise ValueError(
            f"cos_theta_0 must lie in [-1, 1], got {cos_theta_0}. It is the cosine of "
            f"the equilibrium bond angle; outside that range the truncated-normal "
            f"normalizer underflows. To stiffen the chain further, raise gamma."
        )
    if gamma == 0.0:
        return 2.0, 0.0, 1.0 / 3.0 + cos_theta_0 ** 2

    sigma = 1.0 / math.sqrt(2.0 * gamma)
    a = (-1.0 - cos_theta_0) / sigma
    c = (1.0 - cos_theta_0) / sigma
    pdf_a = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    pdf_c = math.exp(-0.5 * c * c) / math.sqrt(2.0 * math.pi)
    mass = 0.5 * (math.erf(c / math.sqrt(2.0)) - math.erf(a / math.sqrt(2.0)))

    z_ang = sigma * math.sqrt(2.0 * math.pi) * mass
    mean_u = cos_theta_0 + sigma * (pdf_a - pdf_c) / mass
    mean_sq_dev = sigma ** 2 * (1.0 + (a * pdf_a - c * pdf_c) / mass)
    return z_ang, mean_u, mean_sq_dev


def bond_vectors(x: torch.Tensor) -> torch.Tensor:
    """Successive bond vectors b_k = x_k - x_{k-1}. (..., N, d) -> (..., N-1, d)."""
    return x[..., 1:, :] - x[..., :-1, :]


def bond_cosines(x: torch.Tensor) -> torch.Tensor:
    """cos theta between successive bonds. (..., N, d) -> (..., N-2).

    The observable the bending potential is written on, and the one to measure on
    reference data when calibrating gamma. Reads normalized bond vectors, so it is
    invariant to O(3) and to translation, like the density itself.
    """
    u = bond_vectors(x)
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
    return (u[..., 1:, :] * u[..., :-1, :]).sum(dim=-1)


def sample_bond_lengths(n: int, k: float, b: float, dtype=torch.float64, device="cpu",
                        generator=None) -> torch.Tensor:
    """Draw n bond magnitudes from p(Q) ~ Q^2 exp(-(k/2)(Q-b)^2). -> (n,), all > 0.

    Rejection sampling on the non-dimensional radius y = Q sqrt(k), whose law is
    y^2 exp(-(y-a)^2 / 2) with a = b sqrt(k). The proposal is a unit normal centred
    on the target's mode y* = (a + sqrt(a^2 + 8))/2 -- the positive root of
    y^2 - a y - 2 = 0 -- truncated to y > 0. Since a - y* = -2/y* exactly, the
    density ratio collapses to exp(2 log y - 2y/y*) up to a constant, and dividing by
    its maximum (at y = y*, by construction) gives the log-acceptance below. The
    bound is therefore tight: measured acceptance runs from 0.68 at a = 0 to above
    0.99 for a >~ 10, so the batch is oversampled by 1/0.65 and one pass almost
    always suffices.

    b = 0 skips the loop entirely: the law is Maxwell-Boltzmann, i.e. the norm of an
    N(0, I_3 / k) vector, which is drawn directly.

    `generator` is threaded through both draws, so results are reproducible for a
    fixed generator. They are NOT stream-aligned with other samplers: rejection
    consumes a data-dependent number of variates.
    """
    if not k > 0.0:
        raise ValueError(f"the bond spring constant k must be positive, got {k}")
    if b < 0.0:
        raise ValueError(f"the bond equilibrium length b must be non-negative, got {b}")

    sqrt_k = math.sqrt(k)
    if b == 0.0:
        v = torch.randn(n, 3, dtype=dtype, device=device, generator=generator)
        return torch.linalg.vector_norm(v, dim=-1) / sqrt_k

    a = b * sqrt_k
    y_star = 0.5 * (a + math.sqrt(a * a + 8.0))

    kept, got = [], 0
    while got < n:
        draw = int((n - got) / 0.65) + 32
        y = torch.randn(draw, dtype=dtype, device=device, generator=generator) + y_star
        y = y[y > 0.0]
        u = torch.rand(y.shape, dtype=dtype, device=device, generator=generator)
        log_accept = 2.0 * torch.log(y / y_star) - (2.0 / y_star) * (y - y_star)
        accepted = y[torch.log(u) <= log_accept]
        kept.append(accepted)
        got += accepted.numel()
    return torch.cat(kept)[:n] / sqrt_k


def sample_bond_cosines(n: int, gamma: float, cos_theta_0: float, dtype=torch.float64,
                        device="cpu", generator=None) -> torch.Tensor:
    """Draw n values of cos theta from p(u) ~ exp(-gamma (u-u_0)^2) on [-1, 1]. -> (n,).

    Inverse-CDF, so it is EXACT and consumes exactly one uniform per sample -- unlike
    `sample_bond_lengths`, whose radial law has no invertible CDF and needs rejection.
    With sigma = 1/sqrt(2 gamma), A = erf((-1-u_0)/(sigma sqrt 2)) and
    C = erf((1-u_0)/(sigma sqrt 2)), the truncated normal's inverse CDF is

        u = u_0 + sigma sqrt(2) erfinv(A + V (C - A)),   V ~ U(0, 1)

    gamma = 0 short-circuits to a uniform draw on [-1, 1].

    Both clamps are load-bearing rather than defensive. At large gamma the interval
    covers many standard deviations, A and C saturate at -1 and +1 in floating point,
    and erfinv of the endpoint is infinite; clamping its argument inside +-1 caps the
    draw at ~5.7 sigma, far outside the region carrying any mass. The final clamp to
    [-1, 1] then guarantees the postcondition downstream trigonometry depends on --
    sqrt(1 - u^2) must not see a negative argument.
    """
    if gamma < 0.0:
        raise ValueError(f"the bending constant gamma must be non-negative, got {gamma}")
    if not -1.0 <= cos_theta_0 <= 1.0:
        raise ValueError(f"cos_theta_0 must lie in [-1, 1], got {cos_theta_0}")

    if gamma == 0.0:
        return torch.rand(n, dtype=dtype, device=device, generator=generator) * 2.0 - 1.0

    sigma = 1.0 / math.sqrt(2.0 * gamma)
    lo = math.erf((-1.0 - cos_theta_0) / (sigma * math.sqrt(2.0)))
    hi = math.erf((1.0 - cos_theta_0) / (sigma * math.sqrt(2.0)))
    v = torch.rand(n, dtype=dtype, device=device, generator=generator)
    arg = torch.clamp(lo + v * (hi - lo), -1.0 + 1e-15, 1.0 - 1e-15)
    u = cos_theta_0 + sigma * math.sqrt(2.0) * torch.erfinv(arg)
    return torch.clamp(u, -1.0, 1.0)


def _perpendicular_basis(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An orthonormal pair spanning the plane perpendicular to unit vectors u: (..., 3).

    Gram-Schmidt against whichever of z-hat or x-hat is less aligned with u, chosen
    branchlessly so the whole batch stays on one kernel. The choice matters: projecting
    out a nearly-parallel axis leaves a residual of norm ~0 and normalizing it amplifies
    roundoff into a garbage direction.

    Which perpendicular pair comes out is arbitrary and does not matter -- the azimuth
    is drawn uniformly, so any orthonormal basis of the plane gives the same law.
    """
    axis = torch.zeros_like(u)
    axis[..., 2] = 1.0
    alt = torch.zeros_like(u)
    alt[..., 0] = 1.0
    axis = torch.where(u[..., 2:3].abs() > 0.9, alt, axis)

    e1 = axis - (axis * u).sum(dim=-1, keepdim=True) * u
    e1 = e1 / torch.linalg.vector_norm(e1, dim=-1, keepdim=True)
    return e1, torch.linalg.cross(u, e1, dim=-1)


def sample_prior(n_batch: int, k: float, b: float, gamma: float, cos_theta_0: float,
                 n_particles: int = N_DEFAULT, n_dims: int = N_DIMS, dtype=torch.float64,
                 device="cpu", generator=None) -> torch.Tensor:
    """Semiflexible prior samples as (n_batch, N, d), tail at the origin.

    Magnitudes come from `sample_bond_lengths` and are independent of the directions.
    The first direction is isotropic; each later one is placed in the frame whose
    z-axis is its predecessor, at a polar angle from `sample_bond_cosines` and a
    uniform azimuth. Positions are the cumulative sum with particle 0 prepended, so the
    result is exactly on the subspace -- row 0 is a literal zero, not a zero up to
    roundoff.

    The direction recursion is sequential in the N-2 conditioned bonds and vectorized
    over the batch, which is the right way round: N is ~20 and the batch is ~10^5.
    """
    _require_3d(n_dims)
    n_bonds = n_particles - 1
    q = sample_bond_lengths(n_batch * n_bonds, k, b, dtype=dtype, device=device,
                            generator=generator)

    head_dir = torch.randn(n_batch, n_dims, dtype=dtype, device=device,
                           generator=generator)
    dirs = [head_dir / torch.linalg.vector_norm(head_dir, dim=-1, keepdim=True)]
    for _ in range(n_bonds - 1):
        prev = dirs[-1]
        cos = sample_bond_cosines(n_batch, gamma, cos_theta_0, dtype=dtype, device=device,
                                  generator=generator).unsqueeze(-1)
        phi = torch.rand(n_batch, 1, dtype=dtype, device=device,
                         generator=generator) * (2.0 * math.pi)
        e1, e2 = _perpendicular_basis(prev)
        sin = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0))
        dirs.append(cos * prev + sin * (torch.cos(phi) * e1 + torch.sin(phi) * e2))

    bonds = q.reshape(n_batch, n_bonds, 1) * torch.stack(dirs, dim=1)
    head = torch.zeros(n_batch, 1, n_dims, dtype=dtype, device=device)
    return torch.cat([head, bonds.cumsum(dim=1)], dim=1)


def log_prior(x: torch.Tensor, k: float, b: float, gamma: float,
              cos_theta_0: float) -> torch.Tensor:
    """Semiflexible log-density on the DOF-dim subspace. x: (B, N, d) -> (B,).

    The analytic density of `sample_prior`'s distribution. Exact: the cumulative-sum
    map from bonds to positions has unit Jacobian determinant (unit lower triangular
    in the bond basis), and the per-bond change to the predecessor's frame is a
    rotation, so the bond density transfers with no correction term either way. See
    the module docstring.

    The constant is log(4 pi Z1) + (N-2) log(2 pi Z1 Z_ang), and the asymmetry is
    real: only the first bond normalizes over the full sphere, because for every other
    the polar integral has already been absorbed into Z_ang. This is the piece that has
    to come from `bond_moments` and `angle_moments`, and it is what turns an estimated
    entropy difference into an absolute entropy (see `prior_entropy`).

    Assumes `x` is anchored; the tail row contributes nothing either way, since the
    density is read off the bond vectors, but an unanchored input means the caller
    is working off-subspace and every other quantity will be wrong too.
    """
    n, d = x.shape[-2], x.shape[-1]
    _require_3d(d)
    z1, _, _ = bond_moments(k, b)
    z_ang, _, _ = angle_moments(gamma, cos_theta_0)

    q = torch.linalg.vector_norm(bond_vectors(x), dim=-1)
    radial = 0.5 * k * ((q - b) ** 2).sum(dim=-1)
    bending = gamma * ((bond_cosines(x) - cos_theta_0) ** 2).sum(dim=-1)
    const = (math.log(4.0 * math.pi * z1)
             + (n - 2) * math.log(2.0 * math.pi * z1 * z_ang))
    return -radial - bending - const


def prior_entropy(k: float, b: float, gamma: float, cos_theta_0: float,
                  n_particles: int = N_DEFAULT, n_dims: int = N_DIMS) -> float:
    """S[p0] = -E[log p0] in nats, exactly. The reference point for absolute entropy.

    The interpolant estimators in `eesi.interpolant` return the DIFFERENCE
    S[p1] - S[p0] and never touch the prior's density, so this closed form is what
    makes S[p1] = S[p0] + dS a number rather than a shift. Magnitudes and angles are
    independent and each is iid across the chain, so the expectations separate:

        S = (N-1) (k/2) E[(Q-b)^2] + (N-2) gamma E[(u-u_0)^2]
            + log(4 pi) + (N-2) log(2 pi) + (N-1) log Z1 + (N-2) log Z_ang

    with E[(Q-b)^2] = 3/k - b (E[Q] - b) from the E[Q^2] = b E[Q] + 3/k identity noted
    in `bond_moments`, and E[(u-u_0)^2] the third return of `angle_moments` -- taken
    about u_0 rather than about E[u], which is exactly why `angle_moments` reports that
    moment and not the variance.

    At gamma = 0 the bending terms drop out and log Z_ang = log 2 absorbs the 2 pi into
    a 4 pi, recovering (N-1) [log(4 pi Z1) + 3/2 - (k b/2)(E[Q] - b)], the
    freely-jointed value.
    """
    _require_3d(n_dims)
    z1, mean_q, _ = bond_moments(k, b)
    z_ang, _, mean_sq_dev = angle_moments(gamma, cos_theta_0)
    n_bonds, n_angles = n_particles - 1, n_particles - 2

    radial = n_bonds * (0.5 * k * (3.0 / k - b * (mean_q - b)) + math.log(z1))
    bending = n_angles * (gamma * mean_sq_dev + math.log(z_ang))
    return (radial + bending
            + math.log(4.0 * math.pi) + n_angles * math.log(2.0 * math.pi))


def end_to_end_mean_sq(k: float, b: float, gamma: float, cos_theta_0: float,
                       n_particles: int = N_DEFAULT) -> float:
    """E[Re^2] under the prior, in closed form.

    Averaging over the uniform azimuth kills every component of a bond perpendicular to
    its predecessor, which makes the DIRECTIONS an exact Markov chain with
    <u_i . u_j> = <u>^|i-j| -- the freely-rotating-chain result. Magnitudes are
    independent of directions and of each other, so with n = N-1 bonds

        E[Re^2] = n E[Q^2] + 2 E[Q]^2 sum_{s=1}^{n-1} (n - s) <u>^s

    The cross terms are what the bending potential buys: at gamma = 0, <u> = 0 and this
    collapses to n E[Q^2], the freely-jointed value, exactly.

    The calibration diagnostic. Fix k and b from the data's bond statistics and gamma
    from its var(cos theta), then move cos_theta_0 until this matches the data's
    measured E[Re^2] -- see `solve_cos_theta_0`.
    """
    n_bonds = n_particles - 1
    _, mean_q, mean_q_sq = bond_moments(k, b)
    _, mean_u, _ = angle_moments(gamma, cos_theta_0)

    total = n_bonds * mean_q_sq
    if mean_u != 0.0:
        total += 2.0 * mean_q ** 2 * math.fsum(
            (n_bonds - s) * mean_u ** s for s in range(1, n_bonds))
    return total


def solve_cos_theta_0(k: float, b: float, gamma: float, target_re_sqr: float,
                      n_particles: int = N_DEFAULT, tol: float = 1e-12) -> float:
    """The cos_theta_0 whose chain has E[Re^2] == target_re_sqr. Bisection on [-1, 1].

    `end_to_end_mean_sq` is monotone increasing in cos_theta_0 (raising it raises <u>,
    and every cross term is a positive power of <u>), so bisection is safe and needs no
    derivative. Hand-rolled rather than scipy's: this is fifteen lines and keeps the
    package's runtime dependencies to torch and numpy.

    Raises if the target is unreachable, reporting the achievable interval. That is not
    a numerical failure but a physical one -- at this gamma the chain cannot be made
    that stiff, and the fix is a larger gamma, not a cos_theta_0 outside [-1, 1].
    """
    lo_val = end_to_end_mean_sq(k, b, gamma, -1.0, n_particles)
    hi_val = end_to_end_mean_sq(k, b, gamma, 1.0, n_particles)
    if not lo_val <= target_re_sqr <= hi_val:
        raise ValueError(
            f"E[Re^2] = {target_re_sqr} is unreachable at gamma={gamma}: cos_theta_0 in "
            f"[-1, 1] spans [{lo_val:.4f}, {hi_val:.4f}]. Raise gamma to stiffen the "
            f"chain further; cos_theta_0 is a cosine and cannot leave [-1, 1]."
        )

    lo, hi = -1.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if end_to_end_mean_sq(k, b, gamma, mid, n_particles) < target_re_sqr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- structural observables -------------------------------------------------


def end_to_end_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared end-to-end distance |x_{N-1} - x_0|^2. (..., N, d) -> (...,)."""
    return ((x[..., -1, :] - x[..., 0, :]) ** 2).sum(-1)


def gyration_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared radius of gyration, mean squared distance to the centroid. -> (...,)."""
    xc = x - x.mean(dim=-2, keepdim=True)
    return (xc ** 2).sum(-1).mean(-1)


# --- Rouse modes ------------------------------------------------------------
#
# Re_sq and Rg_sq are global, the bond length and <cos theta> are local, and nothing
# in between reports on the chain at intermediate wavelengths. The Rouse (discrete
# cosine) modes do: <|X_k|^2> versus k is the backbone's structure factor, resolving
# every length scale from the whole chain (k=1) down to a single bond (k=N-1) at once,
# and the cross-mode covariances <X_k . X_l> catch correlations that a correct
# spectrum can still hide. Together they are the sharpest check available here that
# generated configurations are structurally CONSISTENT and not merely right on
# average -- which matters more than usual for TAP, since the system has no target
# density and so no free-energy cross-check exists.


@lru_cache(maxsize=None)
def _rouse_matrix(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """(N, N) transform behind `rouse_modes`; cached, and read-only downstream."""
    j = torch.arange(n, dtype=dtype, device=device) + 0.5   # j - 1/2 for j = 1..N
    k = torch.arange(n, dtype=dtype, device=device)
    return math.sqrt(2.0 / n) * torch.cos(math.pi * k.unsqueeze(-1) * j / n)


def rouse_modes(x: torch.Tensor) -> torch.Tensor:
    """Rouse (discrete cosine) modes of a chain. (..., N, d) -> (..., N, d).

        X_k = sqrt(2/N) sum_{j=1..N} R_j cos(pi k (j - 1/2) / N),   k = 0 .. N-1

    Conventions, all of which the tests pin:

    k = 0 is the CENTRE OF MASS, X_0 = sqrt(2N) * mean_j R_j, and it is the one mode
    that is not translation invariant. Under TAP's tail-anchored gauge that makes it a
    real, gauge-dependent number rather than an artefact -- but it is not a structural
    mode, so leave it out when plotting a spectrum or comparing against free-chain
    results.

    For k >= 1 the row sums vanish (sum_j cos(pi k (j-1/2)/N) = 0), so those modes are
    translation invariant. Every X_k is O(3)-EQUIVARIANT, X_k -> R X_k, which is what
    makes |X_k|^2 and X_k . X_l legitimate to compare between prior, generated and
    reference samples: those are only ever defined up to a rotation.

    The transform is NOT orthonormal as written. A uniform sqrt(2/N) gives rows of norm
    sqrt(2) at k = 0 and 1 for k >= 1 (orthonormal DCT-II would use sqrt(1/N) on row 0),
    so Parseval picks up an extra centre-of-mass term:

        sum_k |X_k|^2 = sum_j |R_j|^2 + N |mean_j R_j|^2

    This is the polymer literature's normalization and the one the caller asked for, so
    it is documented rather than changed.

    For an ideal chain <|X_k|^2> ~ 1 / (4 sin^2(pi k / 2N)), i.e. ~ k^-2 for k << N --
    the shape to expect on a log-log plot. It is not asserted anywhere: TAP's ensemble
    is tail-anchored, not free, and anchoring is a constraint the free-chain result
    does not account for.
    """
    return _rouse_matrix(x.shape[-2], x.dtype, x.device) @ x


def rouse_mode_moments(x: torch.Tensor):
    """Mean squared mode amplitudes and the cross-mode covariance over a sample.

    (B, N, d) -> (msq (N,), cov (N, N)) with

        msq[k]    = <|X_k|^2>
        cov[k, l] = <X_k . X_l>,     diag(cov) == msq

    Second moments about the ORIGIN, not about the sample mean of X: these describe a
    gauge-fixed distribution in which <X_k> is not zero, so centering would subtract a
    physically meaningful quantity. Normalize with cov[k,l] / sqrt(msq[k] msq[l]) to
    read the off-diagonals as correlations.
    """
    if x.dim() != 3:
        raise ValueError(f"expected a (B, N, d) sample; got {tuple(x.shape)}")
    modes = rouse_modes(x)
    cov = torch.einsum("bkd,bld->kl", modes, modes) / modes.shape[0]
    return torch.diagonal(cov).clone(), cov


# --- reference data ---------------------------------------------------------


def load_ref_data(path=REF_DATA_PATH, n: int | None = None,
                  n_particles: int = N_DEFAULT, n_dims: int = N_DIMS,
                  dtype=torch.float64) -> torch.Tensor:
    """Tail-anchored TAP configurations as (n, N, d).

    Accepts either a flat (M, N*d) or an already-shaped (M, N, d) array, memory-mapped
    so `n` slices before materializing. The generator writes the tail at the origin
    already; re-anchoring here only removes float drift, and guarantees the invariant
    that the prior, the OT coupling and the velocity field all assume.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"TAP reference data not found at {path}. It is generated externally, not "
            f"part of the repo: point this at the `.npy` you produced, or set "
            f"EESI_DATA_DIR to the directory holding it."
        )
    raw = np.load(path, mmap_mode="r")
    raw = np.asarray(raw if n is None else raw[:n]).astype(np.float64)
    x = torch.from_numpy(raw).reshape(-1, n_particles, n_dims).to(dtype)
    return anchor(x)
