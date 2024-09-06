from hmac import new
import scipy as sp

from .adapted_span import LazyAdaptedSpan
from .kernels import LinearIsotropicKernel


class BlockCholesky:
    __slots__ = "_data",

    def __init__(self):
        self._data = []

    def append_row(self, mixed_covariance):
        n = len(mixed_covariance)
        for idx in range(n): # cholesky
            for jdx in range(idx):
                mixed_covariance[idx] -= mixed_covariance[jdx] @ self._data[idx][jdx].T

            mixed_covariance[idx] = self._data[idx][idx].solve(mixed_covariance[idx].T, inplace=True).T

        self._data.append(mixed_covariance)
        return mixed_covariance
    
    def append_corner(self, auto_covariance):
        nat_ce = self._data[-1]
        for block in nat_ce:
            auto_covariance -= block @ block.T
        self._data.append(auto_covariance.cholesky(overwrite_self=True))



class LinIsotropicGRF():
    """ Object which represents a lazily evaluated linear Isotropic Gaussian Random Function """

    __slots__ = 'mean', 'kernel', 'dim', '_adapted_span', "_cholesky", "_block_randomness", '_coeffs'

    def __init__(self, * , mean, kernel:LinearIsotropicKernel, dim) -> None:
        self.dim = dim
        self.mean = mean
        self.kernel = kernel
        self._adapted_span = LazyAdaptedSpan()
        self._cholesky = BlockCholesky()
        self._block_randomness = []
        self._coeffs = sp.sparse.csr_matrix((0, dim))

    def __call__(self, vec):
        new_coeff = self._adapted_span.coeff_from_std_basis(vec)
        c_E = self._conditional_expectation(new_coeff)

        d = self._coeffs.shape[0]
        c_std_old, c_std_new  = self._conditional_std(new_coeff)

        self._coeffs.resize(d+1, self._coeffs.shape[1])
        self._coeffs[d,:len(new_coeff)] = new_coeff


    def _conditional_expectation(self, new_coeff):
        mixed_cov = self.kernel.covariance(new_coeff, self._coeffs, derivatives=1)
        self._cholesky.append_row(mixed_cov)
        return sum(block @ Y for block, Y in zip(self._cholesky._data[-1], self._block_randomness))


    def _conditional_std(self, new_coeff):
        auto_cov = self.kernel.auto_covariance(new_coeff, derivatives=1)


    def gradient(self, x):
        pass
