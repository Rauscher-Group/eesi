"""EESI -- entropy estimation by stochastic interpolants.

Layout: a small general core plus one self-contained subpackage per system.

    eesi.interpolant     EESI, the general stochastic interpolant
    eesi.ot              group-agnostic OT helpers
    eesi.systems.toy     1D/2D Gaussian-mixture examples
    eesi.systems.xy      the 1D XY chain
    eesi.systems.lj13    the LJ13 cluster

The names below are re-exported for convenience; the system subpackages are the
authoritative home for anything system-specific. Training loops are not imported
here -- see `eesi.systems`.
"""
from .interpolant import EESI
from .ot import center, transport_cost
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
from .systems.toy import GaussianMixture, TimeMLP
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
    # --- toy systems (eesi.systems.toy) ---
    "GaussianMixture",
    "TimeMLP",
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
    # --- 1D XY chain (eesi.systems.xy) ---
    "mcxy",
    "sample_p1_exact",
    "XYChainGNN",
    "xyEESI",
    "xy_ot_couple",
    "xy_transport_cost",
]
