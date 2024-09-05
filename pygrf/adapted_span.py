""" The adapted span V_t from the paper """

import numpy as np

from .basis import OrthogonalBasis, CoordinateVec

def last_nonzero_col(matrix):
    return np.flatnonzero(matrix.T)[-1] // np.atleast_2d(matrix).T.shape[0]

class LazyAdaptedSpan(OrthogonalBasis):
    """A lazy version of the Adapted Span V_t"""

    __solts__ = "_dim_t", "_frozen", "rng"

    def __init__(self, *, dim=None, row_basis=None, rng=np.random.default_rng()) -> None:
        if dim is None and row_basis is None:
            raise ValueError("Either dim or initial basis must be provided")

        init_basis = np.empty((0, dim)) if row_basis is None else row_basis

        super().__init__(init_basis)

        self._dim_t = init_basis.shape[0]
        self._frozen = False
        self.rng = rng

    def coeff_into_std_basis(self, coeffs):
        needed_basis_len = last_nonzero_col(coeffs)+1
        self.ensure_eager(needed_basis_len)
        return coeffs[:, :needed_basis_len] @ self.basis_matrix[:needed_basis_len]

    def coeff_from_std_basis(self, row_vecs):
        if len(row_vecs.shape) == 1:
            return self.coeff_from_vec(row_vecs)

        # effecitvely gram-schmidt
        coeff_list = [self.coeff_from_vec(row) for row in row_vecs]
        n = len(coeff_list)
        m = len(coeff_list[-1])
        result= np.empty((n, m))
        for i, coeff in enumerate(coeff_list):
            result[i, :len(coeff)] = coeff
        return result

    def _project_to_current(self, row_vecs):
        """ project a row vector to the currently available basis """
        coeff = row_vecs @ self.basis_matrix.T
        residual = row_vecs - coeff @ self.basis_matrix
        return coeff, residual


    def coeff_from_vec(self, vec):
        """ Convert a single vector to coefficients with respect to the current basis """
        assert len(vec.shape) == 1, "vec must be a 1D array"

        self.ensure_eager(self._dim_t)
        prelim_coeff, residual = self._project_to_current(vec)
        if np.allclose(residual, 0):  # check if residual is zero
            return prelim_coeff

        # residual is not zero
        if self._frozen:
            raise ValueError("Cannot extend the basis after it was frozen")

        # add residual to the basis
        coeff = np.linalg.norm(residual)

        n,m = self.basis_matrix.shape
        self.basis_matrix.resize(n+1, m)
        self._dim_t += 1
        self.basis_matrix[-1] = residual / coeff

        prelim_coeff.resize(len(prelim_coeff) + 1)
        prelim_coeff[-1] = coeff

        return prelim_coeff

    def ensure_eager(self, needed_basis_len):
        if needed_basis_len > self._dim_t:
            raise ValueError("Cannot reference basis vectors of the future")

        d_eager, dim = self.basis_matrix.shape
        self.basis_matrix.resize(max(d_eager, needed_basis_len), dim)
        for idx in range(d_eager, needed_basis_len):
            _, residual = self._project_to_current(self.rng.random(dim))
            residual /= np.linalg.norm(residual)
            self.basis_matrix[idx] = self.rng.random(residual)
