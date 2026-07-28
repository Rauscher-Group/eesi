"""Toy systems: the low-dimensional 1D/2D examples.

A Gaussian-mixture target, a plain time-conditioned MLP and a plain minibatch-OT
coupling, used by the pedagogical notebooks (`experiments/Toy/2D_NonGaussian.ipynb`,
`experiments/Toy/40D_GMM.ipynb`) and as the generic test net for the core interpolant.
"""
from .data import GaussianMixture
from .mlp import TimeMLP
from .ot import toy_cost_matrix, toy_ot_couple, toy_transport_cost

__all__ = [
    "GaussianMixture",
    "TimeMLP",
    "toy_cost_matrix",
    "toy_ot_couple",
    "toy_transport_cost",
]
