"""The 1D XY chain: data, network, interpolant, and OT coupling.

    data.py         exact/MC sampling of the open-chain Boltzmann law
    gnn.py          XYChainGNN, the periodicity-aware velocity/score net
    interpolant.py  xyEESI, the geodesic interpolant on (S^1)^N
    ot.py           equivariant OT over O(2) x Z2^site
    train.py        training entry point (`python -m eesi.systems.xy.train`)
"""
from .data import energy, mcxy, sample_p1_exact
from .gnn import XYChainGNN, angle_wrap
from .interpolant import xyEESI
from .ot import xy_cost_matrix, xy_ot_couple, xy_transport_cost

__all__ = [
    "energy",
    "mcxy",
    "sample_p1_exact",
    "XYChainGNN",
    "angle_wrap",
    "xyEESI",
    "xy_cost_matrix",
    "xy_ot_couple",
    "xy_transport_cost",
]
