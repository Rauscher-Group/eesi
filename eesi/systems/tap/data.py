"""The TAP system itself: reference data, the harmonic-bond prior, subspace geometry.

Everything here is a closed-form fact about the system, true whether or not a model
has ever been trained on it:

    REF_DATA_PATH, load_ref_data    the reference trajectory samples
    sample_prior, log_prior         the harmonic-bond chain base distribution
    sample_bond_lengths             the radial law the prior is built from
    bond_moments, bond_vectors      the one place the bond-law convention lives
    prior_entropy                   S[p0], the reference point for an absolute entropy
    end_to_end_mean_sq              E[Re^2] under the prior, for calibrating (k, b)
    DOF, dof, subspace_dirs         geometry of the tail-anchored subspace
    anchor                          projection onto that subspace
    end_to_end_sq, gyration_sq      structural observables, for tests and notebooks

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


The prior: a harmonic-bond chain
-------------------------------
The base distribution is the real TAP polymer with the activity and the excluded
volume switched off: N-1 iid bonds drawn from a harmonic spring of stiffness k and
finite equilibrium length b, with positions their running sum.

    p(bond) = exp(-(k/2) (|bond| - b)^2) / (4 pi Z1),   k = 1 .. N-1
    x_0 = 0,  x_k = sum_{j <= k} bond_j

Both parameters are properties of the system and have no defaults. Orientations are
isotropic; the magnitudes Q = |bond| carry the 4 pi Q^2 surface Jacobian, so the
radial law is

    p(Q) = Q^2 exp(-(k/2) (Q - b)^2) / Z1,   Q > 0
    Z1   = int_0^inf Q^2 exp(-(k/2) (Q - b)^2) dQ

which is NOT a Gaussian: it is peaked near b, not at the origin, which is the whole
point of the finite equilibrium length. `bond_moments` evaluates Z1 and the first two
moments in closed form and is the single source of truth for the convention --
`log_prior`, `prior_entropy` and `end_to_end_mean_sq` all route through it, and
`sample_bond_lengths` is tested against it, so they cannot disagree.

Setting b = 0 recovers the old ideal (Gaussian) chain exactly, with sigma^2 = 1/k:
the radial law becomes Maxwell-Boltzmann and the bond vector becomes N(0, I_3 / k).
The previous ReSqr parameterization is the b = 0 slice with k = 3 (N-1) / ReSqr.

`log_prior` is exact and needs no Jacobian correction. The bond-to-position map is
the cumulative sum, which in the bond basis is unit lower triangular, so |det J| = 1
and the density transfers unchanged -- that argument never mentioned the bond law, so
it survives the change intact:

    log p0(x) = -sum_k (k/2) (|x_k - x_{k-1}| - b)^2 - (N-1) log(4 pi Z1)

The prior is still O(3)-invariant (isotropic bonds about a pinned origin), which is
what makes the alignment layer of `eesi.systems.tap.ot` marginal-preserving. Note that
this is the load-bearing property, not Gaussianity: nothing in the OT coupling or the
interpolant ever assumed the latter.
"""
from __future__ import annotations

import math
import os
import pathlib

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
            f"the harmonic-bond prior is defined for n_dims=3 only, got {n_dims}. "
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


def bond_vectors(x: torch.Tensor) -> torch.Tensor:
    """Successive bond vectors b_k = x_k - x_{k-1}. (..., N, d) -> (..., N-1, d)."""
    return x[..., 1:, :] - x[..., :-1, :]


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


def sample_prior(n_batch: int, k: float, b: float, n_particles: int = N_DEFAULT,
                 n_dims: int = N_DIMS, dtype=torch.float64, device="cpu",
                 generator=None) -> torch.Tensor:
    """Harmonic-bond prior samples as (n_batch, N, d), tail at the origin.

    Draws N-1 iid bonds -- a magnitude from `sample_bond_lengths` times an
    independent isotropic direction -- and takes their cumulative sum, prepending the
    fixed particle 0. The result is exactly on the subspace: row 0 is a literal zero,
    not a zero up to roundoff.
    """
    _require_3d(n_dims)
    n_bonds = n_particles - 1
    q = sample_bond_lengths(n_batch * n_bonds, k, b, dtype=dtype, device=device,
                            generator=generator)
    u = torch.randn(n_batch, n_bonds, n_dims, dtype=dtype, device=device,
                    generator=generator)
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
    bonds = q.reshape(n_batch, n_bonds, 1) * u
    head = torch.zeros(n_batch, 1, n_dims, dtype=dtype, device=device)
    return torch.cat([head, bonds.cumsum(dim=1)], dim=1)


def log_prior(x: torch.Tensor, k: float, b: float) -> torch.Tensor:
    """Harmonic-bond log-density on the DOF-dim subspace. x: (B, N, d) -> (B,).

    The analytic density of `sample_prior`'s distribution. Exact: the cumulative-sum
    map from bonds to positions has unit Jacobian determinant (unit lower triangular
    in the bond basis), so the bond density transfers with no correction term. See
    the module docstring.

    The constant is (N-1) log(4 pi Z1), NOT a Gaussian normalizer -- it is the one
    piece that has to come from `bond_moments`, and it is what turns an estimated
    entropy difference into an absolute entropy (see `prior_entropy`).

    Assumes `x` is anchored; the tail row contributes nothing either way, since the
    density is read off the bond vectors, but an unanchored input means the caller
    is working off-subspace and every other quantity will be wrong too.
    """
    n, d = x.shape[-2], x.shape[-1]
    _require_3d(d)
    z1, _, _ = bond_moments(k, b)
    q = torch.linalg.vector_norm(bond_vectors(x), dim=-1)
    quad = 0.5 * k * ((q - b) ** 2).sum(dim=-1)
    return -quad - (n - 1) * math.log(4.0 * math.pi * z1)


def prior_entropy(k: float, b: float, n_particles: int = N_DEFAULT,
                  n_dims: int = N_DIMS) -> float:
    """S[p0] = -E[log p0] in nats, exactly. The reference point for absolute entropy.

    The interpolant estimators in `eesi.interpolant` return the DIFFERENCE
    S[p1] - S[p0] and never touch the prior's density, so this closed form is what
    makes S[p1] = S[p0] + dS a number rather than a shift. Bonds are iid, so it is
    (N-1) times the per-bond entropy

        S1 = log(4 pi Z1) + 3/2 - (k b / 2) (E[Q] - b)

    which follows from S1 = log(4 pi Z1) + (k/2) E[(Q-b)^2] together with
    E[(Q-b)^2] = 3/k - b (E[Q] - b), itself a rearrangement of the
    E[Q^2] = b E[Q] + 3/k identity noted in `bond_moments`.

    At b = 0 this reduces to (3/2)(1 + log 2 pi sigma^2) per bond with sigma^2 = 1/k,
    i.e. the ideal chain's Gaussian entropy.
    """
    _require_3d(n_dims)
    z1, mean_q, _ = bond_moments(k, b)
    s1 = math.log(4.0 * math.pi * z1) + 1.5 - 0.5 * k * b * (mean_q - b)
    return (n_particles - 1) * s1


def end_to_end_mean_sq(k: float, b: float, n_particles: int = N_DEFAULT) -> float:
    """E[Re^2] under the prior, (N-1) E[Q^2]. The bonds are iid, so cross terms vanish.

    The calibration diagnostic: pick (k, b) so this lands near the reference data's
    measured mean squared end-to-end distance. Under the old parameterization this
    was ReSqr, an input; here it is derived, which is the honest direction -- k and b
    are properties of the polymer, and the chain's size is a consequence.
    """
    return (n_particles - 1) * bond_moments(k, b)[2]


# --- structural observables -------------------------------------------------


def end_to_end_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared end-to-end distance |x_{N-1} - x_0|^2. (..., N, d) -> (...,)."""
    return ((x[..., -1, :] - x[..., 0, :]) ** 2).sum(-1)


def gyration_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared radius of gyration, mean squared distance to the centroid. -> (...,)."""
    xc = x - x.mean(dim=-2, keepdim=True)
    return (xc ** 2).sum(-1).mean(-1)


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
