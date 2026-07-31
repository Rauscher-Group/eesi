"""The LJ13 system itself: reference data, the prior, energies, and subspace geometry.

Everything here is a closed-form fact about the system, true whether or not a model
has ever been trained on it:

    REF_DATA_PATH, load_ref_data    the OSF reference samples
    sample_prior, log_prior         the COM-free Gaussian base distribution
    lj_energy .. delta_energy       Lennard-Jones + harmonic energies
    DOF, subspace_dirs              geometry of the mean-zero subspace

Anything that needs a velocity field to mean anything -- sampling through the flow,
its divergence, log-densities, free energies -- lives with the model it runs on, in
`eesi.systems.lj13.dynamics`. This module deliberately imports nothing from
`eesi.systems.lj13.dynamics`: the dependency runs one way, dynamics -> data.
"""
from __future__ import annotations

import math
import os
import pathlib

import numpy as np
import torch

# The OSF reference samples (1.5 GB) are kept OUT of the package, in a top-level
# `data/`, so an install never has to copy them. Anchored to the repo root rather
# than the caller's cwd; `EESI_DATA_DIR` overrides it when the checkout is not the
# data's home (e.g. a non-editable install, or a shared copy on a scratch disk).
DATA_DIR = pathlib.Path(
    os.environ.get("EESI_DATA_DIR", pathlib.Path(__file__).resolve().parents[3] / "data")
)
REF_DATA_PATH = DATA_DIR / "all_data_LJ13-1000.npy"
#REF_DATA_PATH = DATA_DIR / "all_data_LJ55-1000-part1.npy"


def load_ref_data(path=REF_DATA_PATH, n: int | None = None,
                  dtype=torch.float64) -> torch.Tensor:
    """COM-free LJ13 reference configurations as (n, 13, 3).

    The `.npy` is 10M x 39 and memory-mapped, so `n` slices before materializing --
    never load the whole 1.5 GB unless you mean to. The shipped data is already
    COM-free; re-centering only removes float drift, and guarantees the invariant
    the prior and the velocity field both assume.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LJ13 reference data not found at {path}. It is a 1.5 GB third-party "
            f"download, not part of the repo: fetch `all_data_LJ13-1000.npy` from "
            f"OSF https://osf.io/srqg7/ and put it there, or point EESI_DATA_DIR at "
            f"wherever it already lives. See data/README.md."
        )
    raw = np.load(path, mmap_mode="r")
    raw = np.asarray(raw if n is None else raw[:n]).astype(np.float64)
    x = torch.from_numpy(raw).view(-1, 13, 3).to(dtype)
    #x = torch.from_numpy(raw).view(-1, 55, 3).to(dtype)
    return x - x.mean(1, keepdim=True)


def sample_prior(n_batch: int, n_particles: int = 13, n_dims: int = 3,
                 dtype=torch.float64, device="cpu", generator=None) -> torch.Tensor:
    """Center-of-gravity-zero Gaussian prior; samples live on the mean-zero subspace."""
    x = torch.randn(n_batch, n_particles, n_dims, dtype=dtype, device=device,
                    generator=generator)
    return x - x.mean(1, keepdim=True)


# --- energy -----------------------------------------------------------------
#
# The target this flow was trained on (Koehler et al. 2020 / Klein et al. 2023;
# en_flows `deprecated/eqnode/test_systems.py::LennardJonesPotential`) is, at T=1:
#
#     U_target(x) = U_LJ(x) + U_osc(x),   U_osc = 1/2 * sum_i |x_i - x_cm|^2
#
# with the Lennard-Jones pair term in the r_m parameterization (well minimum at
# r = r_m, depth eps), summed over ALL ordered pairs i != j:
#
#     U_LJ = sum_{i != j} eps * [ (r_m/r)^12 - 2 (r_m/r)^6 ]
#          = 2 * sum_{i < j} eps * [ (r_m/r)^12 - 2 (r_m/r)^6 ]     (eps = r_m = 1)
#
# Two conventions matter and are *not* the textbook defaults; both were verified
# empirically against the shipped dataset (see the notebook):
#   * length scale r_m = 1 (minimum at r=1), i.e. sigma = 2^(-1/6), NOT sigma = 1;
#   * ordered-pair counting => a factor of 2 vs the physical i<j sum.
# The reference data satisfies the configurational-temperature identity
# <|grad U|^2>/<lap U> = 1 only under this convention.
#
# U_osc is identical to the prior's energy (COM-free unit-variance Gaussian), so
# the *change* in energy from prior to target is purely Lennard-Jones:
#
#     dU(x) = U_target(x) - U_prior(x) = U_LJ(x).

_IU_13 = torch.triu_indices(13, 13, offset=1)  # 78 unique pairs
_IU_55 = torch.triu_indices(55, 55, offset=1)  # 78 unique pairs

def lj_energy(x: torch.Tensor, eps: float = 1.0, rm: float = 1.0, ordered: bool = True,
              soft_eps: float = 1e-12) -> torch.Tensor:
    """Lennard-Jones energy of LJ13 configs. x: (B, 13, 3) -> (B,).

    `ordered=True` reproduces the training target's ordered-pair sum (2x the
    physical i<j energy); `ordered=False` gives the physical unique-pair energy
    whose global minimum is the standard -44.327 eps. Distances are formed
    manually (diff -> d^2 -> sqrt) so the function is safe under forward/reverse
    autograd, unlike `torch.cdist`.
    """
    iu = _IU_13.to(x.device)
    diff = x[:, iu[0]] - x[:, iu[1]]                       # (B, 78, 3)
    r = torch.sqrt((diff ** 2).sum(-1) + soft_eps)         # (B, 78)
    e = (eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)).sum(-1)
    return 2 * e if ordered else e


def oscillator_energy(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Harmonic confinement 1/2 * scale * sum_i |x_i - x_cm|^2. x: (B,13,3) -> (B,)."""
    xc = x - x.mean(1, keepdim=True)
    return 0.5 * scale * (xc ** 2).sum(dim=(1, 2))


def target_energy(x: torch.Tensor) -> torch.Tensor:
    """Full training target U_LJ (ordered) + U_osc at T=1. x: (B,13,3) -> (B,)."""
    return lj_energy(x, ordered=True) + oscillator_energy(x)


def delta_energy(x: torch.Tensor) -> torch.Tensor:
    """Energy change from prior to target, U_target - U_prior. The oscillator
    cancels exactly (target confinement == prior energy), so this is just U_LJ."""
    return lj_energy(x, ordered=True)


# --- subspace geometry --------------------------------------------------------
#
# LJ13 configurations live on the mean-zero (COM-free) subspace, not in the full
# 39-dim ambient space: the 3 translational directions carry no probability. DOF
# and `subspace_dirs` are the same fact in two forms -- the dimension of that
# subspace, and an orthonormal basis for it. The prior normalizer uses the former;
# any divergence/trace taken against the flow uses the latter (see
# `eesi.systems.lj13.dynamics.divergence`).

DOF = 36  # (13 - 1) * 3


def subspace_dirs(n_particles: int = 13, n_dims: int = 3) -> torch.Tensor:
    """Orthonormal basis of the mean-zero subspace as (DOF, n, d) ambient tangents."""
    P = torch.eye(n_particles) - torch.ones(n_particles, n_particles) / n_particles
    evals, evecs = torch.linalg.eigh(P)
    Q = evecs[:, evals > 1e-6]                      # (n, n-1) mean-zero basis
    dirs = [torch.zeros(n_particles, n_dims).index_copy(1, torch.tensor([d]), Q[:, k:k+1])
            for k in range(Q.shape[1]) for d in range(n_dims)]
    return torch.stack(dirs)


def log_prior(x: torch.Tensor) -> torch.Tensor:
    """COM-free standard-Gaussian log-density on the DOF-dim subspace. x:(B,n,d)->(B,).

    The analytic density of `sample_prior`'s distribution -- a closed-form property of
    the base distribution, which is why it lives here and not with the model. The flow's
    own log-density is built from it: log q(x1) = log_prior(x0) - A, with A from
    `eesi.systems.lj13.dynamics.integrate_with_logdet`.
    """
    return -0.5 * x.pow(2).sum(dim=(1, 2)) - 0.5 * DOF * math.log(2 * math.pi)
