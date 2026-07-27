"""The LJ13 cluster: data, dynamics, interpolant, and OT coupling.

    data.py         closed-form facts -- energies, prior, reference data
    dynamics.py     LJ13Dynamics (the flow) and what you get out of it
    interpolant.py  LJ13EESI, the COM-free-subspace interpolant
    ot.py           equivariant OT over S(N) x SO(3)
    train.py        training entry point (`python -m eesi.systems.lj13.train`)
"""
from .data import (
    DOF,
    REF_DATA_PATH,
    delta_energy,
    lj_energy,
    load_ref_data,
    log_prior,
    oscillator_energy,
    sample_prior,
    subspace_dirs,
    target_energy,
)
from .dynamics import (
    LJ13Dynamics,
    divergence,
    free_energy,
    integrate_with_logdet,
    rk4_sample,
)
from .interpolant import LJ13EESI
from .ot import equivariant_ot_couple, lj_cost_matrix

__all__ = [
    # --- closed-form facts (eesi.systems.lj13.data) ---
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
    # --- the flow and what you get out of it (eesi.systems.lj13.dynamics) ---
    "LJ13Dynamics",
    "rk4_sample",
    "divergence",
    "integrate_with_logdet",
    "free_energy",
    # --- interpolant + OT ---
    "LJ13EESI",
    "equivariant_ot_couple",
    "lj_cost_matrix",
]
