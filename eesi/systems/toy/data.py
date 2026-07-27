"""A fixed mixture of high-dimensional multivariate Gaussians in R^d."""
import torch
from torch import nn


class GaussianMixture(nn.Module):
    r"""A frozen mixture of ``n`` multivariate Gaussians in :math:`\mathbb{R}^d`.

    The target density is a weighted sum of full-covariance normals,

    .. math::
        p(x) = \sum_{k=1}^{n} p_k \, \mathcal{N}(x \mid m_k, C_k),

    whose parameters are drawn *once*, at construction, and then held constant:

    * weights ``p_k`` from a symmetric ``Dirichlet(alpha * 1_n)`` (so they are
      non-negative and sum to one);
    * means ``m_k ~ N(0, sigma^2 I_d)``;
    * covariances ``C_k = (1/d) W_k^T W_k + I_d`` with ``(W_k)_{ij} ~ N(0, 1)``,
      which is symmetric positive-definite by construction.

    Each instance is thus a single *realization* of this random construction:
    ``p_k``, ``m_k`` and ``C_k`` are fixed for the lifetime of the object, and
    :meth:`sample` / :meth:`log_prob` may be called at will.

    Sampling and density evaluation go through
    :class:`torch.distributions.MixtureSameFamily` over a batched
    :class:`~torch.distributions.MultivariateNormal`; the per-component
    Cholesky factors are computed once and cached, so drawing samples is cheap
    even in high dimension.

    Args:
        d: dimensionality of the space.
        n: number of mixture components (Gaussians).
        sigma: standard deviation of the mean prior, ``m_k ~ N(0, sigma^2 I_d)``.
        alpha: Dirichlet concentration. A scalar gives a symmetric
            ``Dirichlet(alpha * 1_n)``; ``alpha = 1`` is uniform on the simplex,
            larger values push the weights toward equality, smaller values make
            them sparser.
        seed: optional int for reproducible parameter draws (does not perturb
            the global RNG stream).
        dtype: dtype of the stored parameters (default ``torch.float32``).
        device: device on which to place the parameters.
    """

    def __init__(self, d, n, sigma=1.0, alpha=1.0, seed=None,
                 dtype=torch.float32, device=None):
        super().__init__()
        if d < 1:
            raise ValueError(f"d must be >= 1; got {d}")
        if n < 1:
            raise ValueError(f"n must be >= 1; got {n}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0; got {sigma}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0; got {alpha}")

        weights, means, cov = self._draw_parameters(d, n, sigma, alpha, seed)
        scale_tril = torch.linalg.cholesky(cov)

        self.d = d
        self.n = n
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.event_shape = (d,)

        self.register_buffer("weights", weights.to(dtype), persistent=True)
        self.register_buffer("means", means.to(dtype), persistent=True)
        self.register_buffer("scale_tril", scale_tril.to(dtype), persistent=True)
        if device is not None:
            self.to(device)
        self._build_dist()

    @staticmethod
    def _draw_parameters(d, n, sigma, alpha, seed):
        """Draw (weights [n], means [n, d], covariances [n, d, d]) in float64.

        Parameters are generated under a temporarily seeded global RNG (when
        ``seed`` is given) so the draw is reproducible without leaking into the
        surrounding random stream.
        """
        if seed is not None:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(int(seed))
        try:
            # p_k ~ Dirichlet(alpha * 1_n).
            conc = torch.full((n,), float(alpha), dtype=torch.float64)
            weights = torch.distributions.Dirichlet(conc).sample()
            # m_k ~ N(0, sigma^2 I_d).
            means = torch.randn(n, d, dtype=torch.float64) * sigma
            # C_k = (1/d) W_k^T W_k + I_d, (W_k)_ij ~ N(0, 1).
            W = torch.randn(n, d, d, dtype=torch.float64)
            eye = torch.eye(d, dtype=torch.float64)
            cov = W.transpose(-1, -2) @ W / d + eye
        finally:
            if seed is not None:
                torch.random.set_rng_state(rng_state)
        return weights, means, cov

    def _build_dist(self):
        component = torch.distributions.MultivariateNormal(
            loc=self.means, scale_tril=self.scale_tril
        )
        mixture = torch.distributions.Categorical(probs=self.weights)
        self.dist = torch.distributions.MixtureSameFamily(mixture, component)

    @property
    def covariances(self):
        """The component covariance matrices ``C_k``, shape ``[n, d, d]``."""
        return self.scale_tril @ self.scale_tril.transpose(-1, -2)

    def sample(self, sample_shape=()):
        """Draw samples of shape ``sample_shape + (d,)``.

        ``sample_shape`` may be an int (e.g. a batch size ``B``) or a tuple.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return self.dist.sample(torch.Size(sample_shape))

    def log_prob(self, x):
        """Log-density ``log p(x)`` for ``x`` of shape ``... + (d,)``."""
        return self.dist.log_prob(x)

    def _apply(self, fn, *args, **kwargs):
        new_self = super()._apply(fn, *args, **kwargs)
        new_self._build_dist()
        return new_self

    def extra_repr(self):
        return f"d={self.d}, n={self.n}, sigma={self.sigma}, alpha={self.alpha}"
