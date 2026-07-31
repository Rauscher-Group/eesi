"""GMM system: the low-dimensional Gaussian-mixture examples.

A Gaussian-mixture target, a plain time-conditioned MLP and a plain minibatch-OT
coupling, used by the pedagogical notebook (`experiments/GMM/GMM.ipynb`) and as the
generic test net for the core interpolant.
"""
from .data import GaussianMixture
from .mlp import TimeMLP
from .ot import gmm_cost_matrix, gmm_ot_couple, gmm_transport_cost

__all__ = [
    "GaussianMixture",
    "TimeMLP",
    "gmm_cost_matrix",
    "gmm_ot_couple",
    "gmm_transport_cost",
]
