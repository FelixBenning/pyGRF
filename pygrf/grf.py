import scipy as sp
import numpy as np

from numbers import Number

from .adapted_span import LazyAdaptedSpan
from .kernels import LinearIsotropicKernel


class BlockCholesky:
    __slots__ = ("_data",)

    def __init__(self):
        self._data = []

    def append_row(self, mixed_covariance):
        n = len(mixed_covariance)
        for idx in range(n):  # cholesky
            for jdx in range(idx):
                mixed_covariance[idx] -= mixed_covariance[jdx] @ self._data[idx][jdx].T

            mixed_covariance[idx] = (
                self._data[idx][idx].solve(mixed_covariance[idx].T, inplace=True).T
            )

        self._data.append(mixed_covariance)
        return mixed_covariance

    def append_corner(self, auto_covariance):
        nat_ce = self._data[-1]
        for block in nat_ce:
            auto_covariance -= block @ block.T
        self._data.append(auto_covariance.cholesky(overwrite_self=True))


class LinIsotropicGRF:
    """Object which represents a lazily evaluated linear Isotropic Gaussian Random Function"""

    __slots__ = (
        "mean",
        "kernel",
        "dim",
        "_adapted_span",
        "_cholesky",
        "_randomness",
        "_coeffs",
    )

    def __init__(self, *, mean, kernel: LinearIsotropicKernel, dim, rng=None) -> None:
        if isinstance(rng, Number) or rng is None:
            self._rng = np.random.default_rng(rng) 
        elif isinstance(rng, np.random.Generator):
            self._rng = rng
        else:
            raise ValueError("rng must be a seed number, a numpy random generator or None")
        

        self.dim = dim
        self.mean = mean
        self.kernel = kernel
        self._adapted_span = LazyAdaptedSpan()
        self._cholesky = BlockCholesky()
        self._randomness = sp.sparse.lil_array((0, dim))
        self._coeffs = sp.sparse.csr_array((0, dim))

    def __call__(self, vec):
        new_coeff = self._adapted_span.into_basis(vec).coeffs
        c_E = self._conditional_expectation(new_coeff)

        d = self._coeffs.shape[0]
        c_std_old, c_std_new = self._conditional_std(new_coeff)

        self._coeffs.resize(d + 1, self._coeffs.shape[1])
        self._coeffs[d, : len(new_coeff)] = new_coeff

        new_randomenss = self._rng.normal(size=self.dim)
        z_row = c_E + c_std_old @ new_randomenss



    def _conditional_expectation(self, new_coeff):
        mixed_cov = self.kernel.covariance(new_coeff, self._coeffs, derivatives=1, basis=self._adapted_span)
        self._cholesky.append_row(mixed_cov)
        return sum(
            block @ Y for block, Y in zip(self._cholesky._data[-1], self._randomness)
        )

    def _conditional_std(self, new_coeff):
        auto_cov = self.kernel.auto_covariance(new_coeff, derivatives=1)

    def gradient(self, x):
        pass
