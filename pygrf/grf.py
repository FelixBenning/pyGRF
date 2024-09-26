from enum import auto
from typing import List
from numbers import Number

import scipy as sp
import numpy as np

from .matrices import KiteMatrix, ScaledIdentity
from .adapted_span import LazyAdaptedSpan
from .kernels import IsotropicKernel


class BlockCholesky:
    __slots__ = ("data",)

    def __init__(self):
        self.data: List[List[KiteMatrix]] = []
    
    def solve_inplace(self, mixed_covariance):
        """ solve the equation
            self @ x = mixed_covariance for x
        and return x.transpose()

        Note that mixed_covariance is overwritten with x in the process!
        """
        n = len(mixed_covariance)
        for idx in range(n):  # cholesky
            for jdx in range(idx):
                mixed_covariance[idx] -= self.data[idx][jdx] @ mixed_covariance[jdx].T

            mixed_covariance[idx] = (
                self.data[idx][idx].solve(mixed_covariance[idx], inplace=True).T
            )

        return mixed_covariance

    def append_row(self, mixed_covariance):
        self.data.append(mixed_covariance)

 

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
        "_rng",
    )

    def __init__(self, *, mean, kernel: IsotropicKernel, dim, rng=None) -> None:
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
        self._coeffs = sp.sparse.lil_array((0, dim))

    def __call__(self, vec):
        new_coeff = self._adapted_span.into_basis(vec).coeffs
        c_E, natural_ce = self._conditional_expectation(new_coeff, return_natural_ce=True)

        c_std = self._conditional_std(new_coeff, natural_ce)

        nn = self._randomness.shape[0]
        span_len = len(self._adapted_span)
        self._randomness.resize(nn + 1, self.dim)
        self._randomness[nn,:span_len] = self._rng.normal(size=span_len)

        # only sample the norm of the component orthogonal to the existing span
        self._randomness[nn,span_len] = np.sqrt(self._rng.chisquare(df=self.dim-span_len))
        self._adapted_span.add_random_orthogonal() # lazy new direction

        result = c_E + c_std @ self._randomness[nn]

        # === BOOKKEEPING ===
        # append cholesky
        natural_ce.append(c_std)
        self._cholesky.append_row(natural_ce)

        # append coefficient
        dd = self._coeffs.shape[0]
        self._coeffs.resize(dd + 1, self._coeffs.shape[1])
        self._coeffs[dd, : len(new_coeff)] = new_coeff


        # === return result ===
        return result



    def _conditional_expectation(self, new_coeff, return_natural_ce=False):
        mixed_cov, new_dir = self.kernel.covariance(
            self._coeffs,
            new_coeff,
            derivatives=1,
            basis=self._adapted_span,
            new_dir=True,
        )
        natural_ce = self._cholesky.solve_inplace([
            KiteMatrix(
                dense= coeff.reshape(coeff.shape[0],coeff.shape[2]),
                diag= ScaledIdentity(scale, self.dim - coeff.shape[0])
            )
            for coeff, scale in zip(mixed_cov, new_dir.reshape(-1))
        ])

        result = np.zeros(new_coeff.shape[1])
        for idx, block in enumerate(natural_ce):
            dense = block.dense
            result[:dense.shape[0]] += dense @ self._randomness[idx,:dense.shape[1]]
 
        if return_natural_ce:
            return result, natural_ce
        return result

    def _conditional_std(self, new_coeff, natural_ce):
        auto_cov, new_dir = self.kernel.covariance(
            new_coeff, new_coeff, derivatives=1, new_dir=True
        )
        auto_covariance = KiteMatrix(
            dense= auto_cov.reshape(auto_cov.shape[1], auto_cov.shape[3]),
            diag= ScaledIdentity(new_dir.item(), self.dim - auto_cov.shape[1])
        )
        for block in natural_ce:
            auto_covariance -= block @ block.T
        return auto_covariance.cholesky(overwrite_self=True)

    def gradient(self, x):
        pass
