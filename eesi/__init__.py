from .datasets.base import GaussianMixture
from .datasets.data import ParticleDataset, make_loader
from .datasets.lj13 import (
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
from .datasets.xy import mcxy
from .interpolant import EESI, xyEESI
from .models.lj13_dynamics import (
    LJ13Dynamics,
    divergence,
    free_energy,
    integrate_with_logdet,
    rk4_sample,
)
from .models.mlp import TimeMLP
from .models.xygnn import XYChainGNN
from .ot import equivariant_ot_couple, transport_cost, xy_ot_couple, xy_transport_cost

__all__ = [
    "GaussianMixture",
    "ParticleDataset",
    "make_loader",
    "TimeMLP",
    "EESI",
    "xyEESI",
    "XYChainGNN",
    # --- LJ13 system: closed-form facts (eesi.datasets.lj13) ---
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
    # --- LJ13 model: the flow and what you get out of it (eesi.models.lj13_dynamics) ---
    "LJ13Dynamics",
    "rk4_sample",
    "divergence",
    "integrate_with_logdet",
    "free_energy",
    # --- 1D XY chain (eesi.datasets.xy) ---
    "mcxy",
    # --- equivariant OT coupling (eesi.ot) ---
    "equivariant_ot_couple",
    "transport_cost",
    "xy_ot_couple",
    "xy_transport_cost",
]
