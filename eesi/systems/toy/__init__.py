"""Toy systems: the low-dimensional 1D/2D examples.

A Gaussian-mixture target and a plain time-conditioned MLP, used by the
pedagogical notebooks (`experiments/1D_NonGaussian.ipynb`, `2D_GMM.ipynb`) and
as the generic test net for the core interpolant.
"""
from .data import GaussianMixture
from .mlp import TimeMLP

__all__ = ["GaussianMixture", "TimeMLP"]
