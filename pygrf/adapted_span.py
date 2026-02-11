""" The adapted span V_t from the paper """

import numpy as np

from .basis import OrthogonalBasis


class LazyAdaptedSpan(OrthogonalBasis):
    """A lazy version of the Adapted Span V_t"""

    __solts__ = "_dim_t", "rng"

    def __init__(
        self, *, dim=None, row_basis=None, rng=np.random.default_rng()
    ) -> None:
        if dim is None and row_basis is None:
            raise ValueError("Either dim or initial basis must be provided")

        init_basis = np.empty((0, dim)) if row_basis is None else row_basis

        super().__init__(init_basis)

        self._dim_t = init_basis.shape[0]
        self._eager = self._dim_t
        # in essence we keep track of the rows of basis_matrix here, why not use shape?
        # because we essentially want to use the basis_matrix as a dynamic array
        # where the allocated memory is sometimes larger than the actual number of rows
        self.rng = rng

    def __len__(self):
        return self._dim_t

    def coeff_into_std_basis(self, coeffs):
        coeff_2d = np.atleast_2d(coeffs)
        dim = min(coeff_2d.shape[1], self._dim_t)

        # if dim = shape[1], then the following will be empty and skipped
        # if dim = self._dim_t then there can not be larger coefficients
        if np.any(coeff_2d[:, dim:].nonzero()):
            raise ValueError("Cannot reference basis vectors of the future")

        self.ensure_eager(dim)
        result = coeff_2d[:, :dim] @ self._basis_matrix[:dim]
        if len(coeffs.shape) == 2:
            return result
        return result.reshape(-1)

    def coeff_from_std_basis(self, row_vecs):
        self.ensure_eager(self._dim_t)

        row_vecs_2d = np.atleast_2d(row_vecs)
        needed_mat_size = min(self._eager + row_vecs_2d.shape[0], self.dim)
        if needed_mat_size > self._basis_matrix.shape[0]:
            self._basis_matrix.resize(
                needed_mat_size, self._basis_matrix.shape[1], refcheck=False
            )

        coeffs = np.zeros((row_vecs_2d.shape[0], needed_mat_size))

        # effectively do gram-schmidt orthogonalization here
        for idx, row_vec in enumerate(row_vecs_2d):
            self._coeff_from_vec(row_vec, out=coeffs[idx])

        return (
            coeffs[:, : self._dim_t] if row_vecs.ndim > 1 else coeffs[0, : self._dim_t]
        )

    def _project(self, row_vecs, up_to_dim):
        """project row vectors onto the first up_to_dim vectors of the basis

        return the coefficients and the residuals
        """
        coeff = row_vecs @ self._basis_matrix[:up_to_dim].T
        residual = row_vecs - coeff @ self._basis_matrix[:up_to_dim]
        return coeff, residual

    def _coeff_from_vec(self, vec, out):
        """Convert a single vector to coefficients with respect to the current basis.
           If the vec is not already in the span, add its orthogonal component
           to the basis.
        """
        out[: self._dim_t], residual = self._project(vec, up_to_dim=self._dim_t)
        if np.allclose(residual, 0):  # check if residual is zero
            return out

        # add residual to the basis
        new_coeff = np.linalg.norm(residual)
        self._basis_matrix[self._dim_t, :] = residual / new_coeff
        out[self._dim_t] = new_coeff
        self._dim_t += 1
        self._eager += 1

        return out

    def add_random_orthogonal(self):
        """Add a random orthogonal vector to the basis"""
        if self._dim_t < self.dim:
            self._dim_t += 1
            # lazy, only actually generate vec if needed
        return self._dim_t

    def ensure_eager(self, needed_basis_len):
        """Ensure that the eagerly calculated part of matrix
        is at least of size needed_basis_len"""
        if needed_basis_len > self._dim_t:
            raise ValueError("Cannot reference basis vectors of the future")

        n, dim = self._basis_matrix.shape
        self._basis_matrix.resize(max(n, needed_basis_len), dim, refcheck=False)

        for idx in range(self._eager, needed_basis_len):
            # generate random vector, uniformly distributed on the orthogonal sphere
            _, residual = self._project(self.rng.random(dim), up_to_dim=idx)
            residual /= np.linalg.norm(residual)
            self._basis_matrix[idx] = residual

        self._eager = max(self._eager, needed_basis_len)
