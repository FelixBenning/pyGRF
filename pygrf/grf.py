from typing import List
from numbers import Number

import scipy as sp
import numpy as np

from pygrf.basis import CoordinateVec

from pygrf.matrices import KiteMatrix, ScaledIdentity
from pygrf.adapted_span import LazyAdaptedSpan
from pygrf.kernels import IsotropicKernel, SquaredExponentialKernel, partial_derivatives


class BlockCholesky:
    __slots__ = ("data",)

    def __init__(self):
        self.data: List[List[KiteMatrix]] = []

    def __array__(self):
        max_row_len = len(self.data[-1])
        zero_fill = np.zeros(self.data[-1][-1].shape)
        return np.block(
            [
                [
                    np.asarray(row[idx]) if idx < len(row) else zero_fill
                    for idx in range(max_row_len)
                ]
                for row in self.data
            ]
        )

    def solve_inplace(self, mixed_covariance):
        """solve the equation
            self @ x = mixed_covariance for x
        and return x.transpose()

        Note that mixed_covariance is overwritten with x in the process!
        """
        n = len(mixed_covariance)
        for idx in range(n):  # cholesky
            for jdx in range(idx):
                mixed_covariance[idx] -= self.data[idx][jdx] @ mixed_covariance[jdx].T

            lower_block = self.data[idx][idx]
            if isinstance(lower_block, KiteMatrix):
                mixed_covariance[idx] = lower_block.solve_triangular(
                    mixed_covariance[idx], inplace=True
                ).T
            else:
                mixed_covariance[idx] = sp.linalg.solve_triangular(
                    lower_block, mixed_covariance[idx], lower=True, overwrite_b=True
                ).T

        return mixed_covariance

    def append_row(self, mixed_covariance):
        self.data.append(mixed_covariance)


class IsotropicGRF:
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
        "_noise",
    )

    def __init__(
        self, *, kernel: IsotropicKernel, dim, mean=0, rng=None, noise=1e-6
    ) -> None:
        if isinstance(rng, Number) or rng is None:
            self._rng = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            self._rng = rng
        else:
            raise ValueError(
                "rng must be a seed number, a numpy random generator or None"
            )

        self.dim = dim
        if isinstance(mean, Number):
            self.mean = lambda x: mean
        else:
            self.mean = mean
        self.kernel = kernel
        self._adapted_span = LazyAdaptedSpan(dim=dim)
        self._cholesky = BlockCholesky()
        self._randomness = []
        self._coeffs = sp.sparse.lil_array((0, dim))
        self._noise = noise

    def into_adapted_span(self, vec):
        """ convert vec into a Coordinate Vector of the adapted span"""
        return self._adapted_span.into_basis(vec)

    def __call__(self, vec, /, *, with_gradient=False):
        if vec.ndim > 1:
            if with_gradient:
                return [self(x) for x in vec]
            else:
                return np.array([self(x) for x in vec])
        new_coeff = self._adapted_span.into_basis(vec).coeffs
        cond_exp, natural_ce = self._conditional_expectation(
            new_coeff, return_natural_ce=True
        )

        c_std = self._conditional_std(new_coeff, natural_ce)

        new_randomness = np.empty(min(len(cond_exp) + 1, self.dim + 1))
        new_randomness[: len(cond_exp)] = self._rng.normal(
            size=len(cond_exp),
            scale=1/np.sqrt(self.dim)
        )

        if len(cond_exp) < self.dim + 1:
            # only sample the norm of the component orthogonal to the existing span
            df = self.dim + 1 - len(cond_exp) # remaining dimensions
            new_randomness[len(cond_exp)] = np.sqrt(
                self._rng.gamma(shape=df/2, scale=2/self.dim)
            )
            # lazy new direction
            self._adapted_span.add_random_orthogonal()

            noise_on_cE = c_std.dense @ new_randomness[: len(cond_exp)]
            new_v_len = c_std.diag.scale * new_randomness[len(cond_exp)]
            z_result = np.append(cond_exp + noise_on_cE, new_v_len)
        else:
            z_result = cond_exp + c_std @ new_randomness

        # === BOOKKEEPING ===
        # append randomness
        self._randomness.append(new_randomness)

        # append cholesky
        natural_ce.append(c_std)
        self._cholesky.append_row(natural_ce)

        # append coefficient
        dd = self._coeffs.shape[0]
        self._coeffs.resize(dd + 1, self._coeffs.shape[1])
        self._coeffs[dd, : len(new_coeff)] = new_coeff

        # === return result ===
        ## VALUE
        sq_norm_half = np.einsum("i,i->", new_coeff, new_coeff) / 2
        val = self.mean(sq_norm_half) + z_result[0] # the first entry is the val

        ## GRADIENT
        grad_coeff = z_result[1:] # the remaining entries are the gradient
        # since the coordinate vector might have fewer entries as one new direction is added
        # add mean with len range
        grad_coeff[:len(new_coeff)] = partial_derivatives(self.mean, 1, dim=1)(sq_norm_half) * new_coeff
        gradient = CoordinateVec(basis_ref=self._adapted_span, coeffs=grad_coeff)

        if with_gradient:
            return val, gradient
        return val

    def _conditional_expectation(self, new_coeff, return_natural_ce=False):
        new_coeff_2d = np.atleast_2d(new_coeff)
        old_coeffs = self._coeffs[:, : len(self._adapted_span)].toarray()
        if old_coeffs.shape[0] == 0:
            result = np.zeros(len(new_coeff_2d) + 1)
            if return_natural_ce:
                return result, []
            return result
        mixed_cov, new_dir = self.kernel.covariance(
            old_coeffs,
            new_coeff_2d,
            derivatives=1,
            basis=self._adapted_span,
            new_dir=True,
        )
        if (residual_dim := self.dim - len(self._adapted_span)) > 0:
            mixed_covariance = [
                KiteMatrix(
                    dense=block.reshape(block.shape[0], block.shape[2]),
                    diag=ScaledIdentity(scale, residual_dim),
                )
                for block, scale in zip(mixed_cov, new_dir.reshape(-1))
            ]
        else:
            mixed_covariance = [
                block.reshape(block.shape[0], block.shape[2]) for block in mixed_cov
            ]
        natural_ce = self._cholesky.solve_inplace(mixed_covariance)

        result = np.zeros(new_coeff_2d.shape[1] + 1)
        for idx, block in enumerate(natural_ce):
            dense = block if not isinstance(block, KiteMatrix) else block.dense
            rand = self._randomness[idx]
            result[: dense.shape[0]] += dense[:, : len(rand)] @ rand

        if return_natural_ce:
            return result, natural_ce
        return result

    def _conditional_std(self, new_coeff, natural_ce):
        new_coeff_2d = np.atleast_2d(new_coeff)
        auto_cov, new_dir = self.kernel.covariance(
            new_coeff_2d, new_coeff_2d, derivatives=1, new_dir=True
        )
        coeff_dim = new_coeff_2d.shape[1]
        auto_covariance = auto_cov.reshape(coeff_dim + 1, coeff_dim + 1)
        auto_covariance += self._noise * np.eye(coeff_dim + 1)
        if self.dim > coeff_dim:
            auto_covariance = KiteMatrix(
                dense=auto_covariance,
                diag=ScaledIdentity(new_dir.item(), self.dim - coeff_dim),
            )
        for block in natural_ce:
            auto_covariance -= block @ block.T

        if isinstance(auto_covariance, KiteMatrix):
            return auto_covariance.cholesky(overwrite_self=True, lower=True)
        return sp.linalg.cholesky(auto_covariance, overwrite_a=True, lower=True)

    def gradient(self, x):
        """return the gradient at x"""
        return self(x, with_gradient=True)[1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # plot tests

    # %% 1D
    f1 = IsotropicGRF(dim=1, kernel=SquaredExponentialKernel())
    x = np.arange(start=0, stop=10, step=0.1).reshape((-1, 1))
    y = f1(x)

    plt.plot(x.reshape(-1), y)
    plt.show()

    # %% 2D
    f2 = IsotropicGRF(dim=2, kernel=SquaredExponentialKernel())
    X,Y = np.mgrid[-5:5:0.4, -5:5:0.4]
    points = [np.array([x,y]) for x,y in zip(X.flatten(),Y.flatten())]
    z = np.array([f2(point) for point in points]).reshape(X.shape)
    plt.contour(X,Y, z)
    plt.show()

    # %% 10D Gradient Descent
    dim= 100
    f10 = IsotropicGRF(dim=dim, kernel=SquaredExponentialKernel())
    x0 = np.random.rand(dim)
    x0 = f10.into_adapted_span(x0)
    X = [x0]
    Y = []
    ts = range(20)
    for _ in ts:
        x = X[-1]
        f, g = f10(x, with_gradient=True)
        Y.append(f)
        X.append(x - g)
    
    plt.plot(ts, Y)
    plt.show()

    # %% 100D Gradient Descent
