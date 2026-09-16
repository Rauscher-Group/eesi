"""EESI -- entropy estimation by stochastic interpolants.

Layout: a small general core plus one self-contained subpackage per system.

    eesi.interpolant     EESI, the general stochastic interpolant
    eesi.ot              group-agnostic OT helpers
    eesi.egnn            the Satorras E(n)-GNN backbone, shared by point-cloud systems
    eesi.paths           DATA_DIR / CHECKPOINTS_DIR, resolved once for every consumer
    eesi.systems.gmm     1D/2D Gaussian-mixture examples
    eesi.systems.xy      the 1D XY chain
    eesi.systems.lj13    the LJ13 cluster
    eesi.systems.tap     the tangentially active polymer

The names below are re-exported for convenience; the system subpackages are the
authoritative home for anything system-specific. Training loops are not imported
here -- see `eesi.systems`.

Deliberately NOT re-exported: TAP's `sample_prior`, `log_prior`, `load_ref_data`,
`subspace_dirs` and `DOF`. Those names are already bound to LJ13's versions here, and
the two systems mean different things by them -- a semiflexible chain on a tail-anchored
subspace versus an isotropic Gaussian on a COM-free one. Shadowing one with the other
at package level would be a silent, hard-to-trace bug, so TAP's live in
`eesi.systems.tap` alone.
"""
from .interpolant import EESI
from .ot import center, transport_cost
from .paths import CHECKPOINTS_DIR, DATA_DIR
from .systems.lj13 import (
    DOF,
    REF_DATA_PATH,
    LJ13Dynamics,
    LJ13EESI,
    delta_energy,
    divergence,
    equivariant_ot_couple,
    free_energy,
    integrate_with_logdet,
    lj_energy,
    load_ref_data,
    log_prior,
    oscillator_energy,
    rk4_sample,
    sample_prior,
    subspace_dirs,
    target_energy,
)
from .systems.gmm import (
    GaussianMixture,
    TimeMLP,
    gmm_cost_matrix,
    gmm_ot_couple,
    gmm_transport_cost,
)
from .systems.tap import (
    TAPDynamics,
    TAPEESI,
    tap_cost_matrix,
    tap_ot_couple,
)
from .systems.xy import (
    XYChainGNN,
    mcxy,
    sample_p1_exact,
    xy_ot_couple,
    xy_transport_cost,
    xyEESI,
)

__all__ = [
    # --- core ---
    "EESI",
    "center",
    "transport_cost",
    "DATA_DIR",
    "CHECKPOINTS_DIR",
    # --- GMM system (eesi.systems.gmm) ---
    "GaussianMixture",
    "TimeMLP",
    "gmm_cost_matrix",
    "gmm_ot_couple",
    "gmm_transport_cost",
    # --- LJ13 system: closed-form facts (eesi.systems.lj13.data) ---
    "REF_DATA_PATH",
    "load_ref_data",
    "sample_prior",
    "log_prior",
    "lj_energy",
    "oscillator_energy",
    "target_energy",
    "delta_energy",
    "DOF",
    "subspace_dirs",
    # --- LJ13 model: the flow and what you get out of it (eesi.systems.lj13.dynamics) ---
    "LJ13Dynamics",
    "rk4_sample",
    "divergence",
    "integrate_with_logdet",
    "free_energy",
    "LJ13EESI",
    "equivariant_ot_couple",
    # --- tangentially active polymer (eesi.systems.tap) ---
    # only the unambiguous names; see the module docstring for what is withheld
    "TAPDynamics",
    "TAPEESI",
    "tap_ot_couple",
    "tap_cost_matrix",
    # --- 1D XY chain (eesi.systems.xy) ---
    "mcxy",
    "sample_p1_exact",
    "XYChainGNN",
    "xyEESI",
    "xy_ot_couple",
    "xy_transport_cost",
]
