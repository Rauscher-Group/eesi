"""TAP, a tangentially active polymer: data, dynamics, interpolant, and OT coupling.

    data.py         closed-form facts -- the semiflexible prior, subspace geometry
    dynamics.py     TAPDynamics (the flow) and what you get out of it
    interpolant.py  TAPEESI, the tail-anchored-subspace interpolant
    ot.py           equivariant OT over O(3)
    train.py        training entry point (`python -m eesi.systems.tap.train`)

The chain is gauge-fixed by pinning its tail at the origin, so configurations live on
a 3(N-1)-dimensional subspace. Two things follow, and both differ from LJ13:

  * the symmetry group is O(3) ALONE -- the tangential drive makes the chain a
    directed object, so head-to-tail relabelling is not a symmetry and the coupling
    has no permutation layer;
  * rotations act about the PINNED TAIL, not the centroid, so nothing here centers.

The system is driven by a non-conservative force, so it has no Boltzmann target and no
free energy to reweight toward -- which is precisely why an interpolant estimate of its
entropy is worth having.

Note that `sample_prior`, `log_prior`, `load_ref_data`, `subspace_dirs` and `DOF` are
system-specific and share their names with LJ13's versions, which are the ones
re-exported at the top level of `eesi`. Reach TAP's through this subpackage.
"""
from .data import (
    DOF,
    N_DEFAULT,
    REF_DATA_PATH,
    anchor,
    angle_moments,
    bond_cosines,
    bond_moments,
    bond_vectors,
    dof,
    end_to_end_mean_sq,
    end_to_end_sq,
    gyration_sq,
    load_ref_data,
    log_prior,
    prior_entropy,
    rouse_mode_moments,
    rouse_modes,
    sample_bond_cosines,
    sample_bond_lengths,
    sample_prior,
    solve_cos_theta_0,
    subspace_dirs,
)
from .dynamics import TAPDynamics, divergence, integrate_with_logdet, rk4_sample
from .interpolant import TAPEESI
from .ot import tap_cost_matrix, tap_ot_couple

__all__ = [
    # --- closed-form facts (eesi.systems.tap.data) ---
    "REF_DATA_PATH",
    "load_ref_data",
    "sample_prior",
    "sample_bond_lengths",
    "sample_bond_cosines",
    "log_prior",
    "prior_entropy",
    "bond_moments",
    "angle_moments",
    "bond_vectors",
    "bond_cosines",
    "end_to_end_sq",
    "end_to_end_mean_sq",
    "solve_cos_theta_0",
    "gyration_sq",
    "rouse_modes",
    "rouse_mode_moments",
    "anchor",
    "DOF",
    "dof",
    "N_DEFAULT",
    "subspace_dirs",
    # --- the flow and what you get out of it (eesi.systems.tap.dynamics) ---
    "TAPDynamics",
    "rk4_sample",
    "divergence",
    "integrate_with_logdet",
    # --- interpolant + OT ---
    "TAPEESI",
    "tap_ot_couple",
    "tap_cost_matrix",
]
