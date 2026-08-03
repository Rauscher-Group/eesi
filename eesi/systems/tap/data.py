"""The TAP system itself: reference data, the Gaussian-polymer prior, subspace geometry.

Everything here is a closed-form fact about the system, true whether or not a model
has ever been trained on it:

    REF_DATA_PATH, load_ref_data    the reference trajectory samples
    sample_prior, log_prior         the ideal-chain (Gaussian polymer) base distribution
    bond_sigma, bond_vectors        the one place the variance convention lives
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
scheme can reach. If a conservative piece is ever wanted as a diagnostic (harmonic
bonds, excluded volume), it belongs in this module, clearly labelled as a component
of the dynamics rather than as a target density.


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


The prior: an ideal (Gaussian) chain
------------------------------------
The base distribution is the freely-jointed chain in its continuum limit: N-1 iid
isotropic Gaussian bond vectors, with positions their running sum.

    b_k ~ N(0, sigma^2 I_3),  k = 1 .. N-1        sigma^2 = ReSqr / (3 (N-1))
    x_0 = 0,  x_k = sum_{j <= k} b_j

The variance convention is fixed by ReSqr, the mean squared end-to-end distance,
which is a parameter of the system and has no default:

    E|b_k|^2 = 3 sigma^2 = ReSqr / (N-1)
    E|x_{N-1} - x_0|^2 = (N-1) E|b|^2 = ReSqr           <- the definition

`log_prior` is exact and needs no Jacobian correction. The bond-to-position map is
the cumulative sum, which in the bond basis is unit lower triangular, so |det J| = 1
and the density transfers unchanged:

    log p0(x) = -sum_k |x_k - x_{k-1}|^2 / (2 sigma^2) - (DOF/2) log(2 pi sigma^2)

The prior is O(3)-invariant (isotropic bonds about a pinned origin), which is what
makes the alignment layer of `eesi.systems.tap.ot` marginal-preserving.
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


def bond_sigma(re_sqr: float, n_particles: int = N_DEFAULT, n_dims: int = N_DIMS) -> float:
    """Per-COMPONENT standard deviation of a bond vector.

    sigma^2 = ReSqr / (d (N-1)), so that E|b|^2 = d sigma^2 = ReSqr/(N-1) and the
    end-to-end mean square comes out at ReSqr. The single source of truth for the
    convention -- `sample_prior` and `log_prior` both route through it, so they can
    never disagree.
    """
    return math.sqrt(re_sqr / (n_dims * (n_particles - 1)))


def bond_vectors(x: torch.Tensor) -> torch.Tensor:
    """Successive bond vectors b_k = x_k - x_{k-1}. (..., N, d) -> (..., N-1, d)."""
    return x[..., 1:, :] - x[..., :-1, :]


def sample_prior(n_batch: int, re_sqr: float, n_particles: int = N_DEFAULT,
                 n_dims: int = N_DIMS, dtype=torch.float64, device="cpu",
                 generator=None) -> torch.Tensor:
    """Ideal-chain prior samples as (n_batch, N, d), tail at the origin.

    Draws N-1 iid isotropic bonds of scale `bond_sigma` and takes their cumulative
    sum, prepending the fixed particle 0. The result is exactly on the subspace: row
    0 is a literal zero, not a zero up to roundoff.
    """
    sigma = bond_sigma(re_sqr, n_particles, n_dims)
    b = torch.randn(n_batch, n_particles - 1, n_dims, dtype=dtype, device=device,
                    generator=generator) * sigma
    head = torch.zeros(n_batch, 1, n_dims, dtype=dtype, device=device)
    return torch.cat([head, b.cumsum(dim=1)], dim=1)


def log_prior(x: torch.Tensor, re_sqr: float) -> torch.Tensor:
    """Ideal-chain log-density on the DOF-dim subspace. x: (B, N, d) -> (B,).

    The analytic density of `sample_prior`'s distribution. Exact: the cumulative-sum
    map from bonds to positions has unit Jacobian determinant (unit lower triangular
    in the bond basis), so the Gaussian density on the bonds transfers with no
    correction term. See the module docstring.

    Assumes `x` is anchored; the tail row contributes nothing either way, since the
    density is read off the bond vectors, but an unanchored input means the caller
    is working off-subspace and every other quantity will be wrong too.
    """
    n, d = x.shape[-2], x.shape[-1]
    sigma = bond_sigma(re_sqr, n, d)
    b = bond_vectors(x)
    quad = (b ** 2).sum(dim=(-1, -2)) / (2.0 * sigma ** 2)
    return -quad - 0.5 * dof(n, d) * math.log(2.0 * math.pi * sigma ** 2)


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
