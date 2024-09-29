""" Linar algebra module for basis and coordinate vectors """

from numbers import Number
import abc

import scipy as sp
import numpy as np


class Basis(abc.ABC):
    """Basis of a vector space"""

    __slots__ = ("_basis_matrix",)

    def __init__(self, row_basis) -> None:
        """The rows of the basis matrix are the basis vectors"""
        self._basis_matrix = row_basis

    def into_basis(self, vec):
        """Convert a vector to this basis"""
        if isinstance(vec, CoordinateVec):
            if vec.basis == self:
                return vec

            _vec = vec.in_std_basis()
        else:
            _vec = vec

        return CoordinateVec(basis_ref=self, coeffs=self.coeff_from_std_basis(_vec))

    def coeff_from_std_basis(self, row_vecs):
        """Convert standard basis representation into rows of coefficients"""
        # col_basis @ col_coeff = col_vecs -> row_basis.T @ row_coeff.T = row_vecs.T
        return sp.linalg.solve(self._basis_matrix, row_vecs.T, transposed=True).T

    def coeff_into_std_basis(self, coeffs):
        """Convert coefficients into standard basis representation"""
        return coeffs @ self._basis_matrix

    @property
    def dim(self):
        """Return the dimension of the vector space"""
        return self._basis_matrix.shape[1]


class OrthogonalBasis(Basis):
    """An orthogonal basis whose vectors are orthonormal

    i.e. the inverse of its col_matrix is its transpose
    """

    def coeff_from_std_basis(self, row_vecs):
        # row_basis.T @ row_coeff.T = row_vecs.T
        # -> row_coeff.T = row_basis @ row_vecs.T
        # -> row_coeff = row_vecs @ row_basis.T
        return row_vecs @ self._basis_matrix.T


class StandardBasis(OrthogonalBasis):
    """The standard basis"""

    def __init__(self, dim) -> None:
        super().__init__(sp.sparse.eye_array(dim))


class CoordinateVec:
    """A coordinate vector with respect to a basis"""

    __slots__ = "basis", "coeffs"

    def __init__(self, basis_ref: Basis, coeffs) -> None:
        assert coeffs.ndim == 1, "vector should not have more than one dim"
        self.basis = basis_ref
        self.coeffs = coeffs

    @property
    def ndim(self):
        """ number of dimensions """
        return 1

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("copy=False is not supported. A copy is always made.")

        return np.array(self.in_std_basis(), dtype=dtype)

    def in_std_basis(self):
        """Translate to standard basis"""
        return self.basis.coeff_into_std_basis(self.coeffs)

    def __add__(self, other):
        return CoordinateVec(
            basis_ref=self.basis,
            coeffs=self.coeffs + self.basis.into_basis(other).coeffs,
        )

    def __radd__(self, other):
        return other + self.in_std_basis()

    def __mul__(self, other):
        if isinstance(other, Number):
            return CoordinateVec(self.basis, other * self.coeffs)

        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return CoordinateVec(
            basis_ref=self.basis,
            coeffs=self.coeffs - self.basis.into_basis(other).coeffs,
        )

    def __rsub__(self, other):
        return other - self.in_std_basis()
