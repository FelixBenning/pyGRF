from abc import abstractmethod
from numbers import Number
from functools import cache

import sympy as sym
import numpy as np

from pygrf.basis import OrthogonalBasis


@cache
def partial_derivatives(func, indices, dim=3):
    """Compute (higher order) partial derivatives of the function

    Example:
        indices = (1,2,2)
        dim = 3
        return ∂1 ∂2 ∂2 func(t_1, t_2, t_3)
    """
    print(f"cache miss: partial_derivatives({func}, {indices}, {dim})")  # logging
    if isinstance(indices, tuple):
        _key = indices
    elif isinstance(indices, Number):
        _key = (indices,)
    else:
        raise NotImplementedError("indices should be Number or tuple of Numbers")

    t = sym.symbols(f"t1:{dim}")
    expr = func(*t)
    for idx in _key:
        expr = sym.diff(t[idx], expr)

    return sym.lambdify(t, expr)


class Kernel:

    @abstractmethod
    def covariance(self, c1, c2, /, *, derivatives=0, basis=None):
        """Compute the covariance between coordinates c1 and c2 with
        respect to the basis basis.

        if derivatives>0, also compute the derivatives
        """
        if not isinstance(basis, OrthogonalBasis):
            raise NotImplementedError("Only OrthogonalBasis are supported")


class LinearIsotropicKernel(Kernel):

    def __init__(self, kernel_expression):
        self._kernel_func_expr = kernel_expression

    def covariance(self, c1, c2, /, *, derivatives=0, basis=None):
        """Compute the covariance between coordinates c1 and c2 with
        respect to `basis'.

        returns tensor:

        [i1, c1, i2, c2] = cov(D_{v_1} c_1[i1], D_{v_2} c_2[i2])

        for i1 in c_1
            for v1 in derivatives
                for i2 in c2
                    for v2 in derivatives
                        cov(D_{v_1} c_1[i1], D_{v_2} c_2[i2])

        if derivatives>0, also compute the derivatives
        """
        if not isinstance(basis, OrthogonalBasis):
            raise NotImplementedError("Only OrthogonalBasis are supported")

    def covariance_new_dir_derivative(self, c1, c2, /, *, basis=None):
        c1_norms = np.einsum("ij,ij->i", c1, c1)  # row-wise squared norms
        c2_norms = np.einsum("ij,ij->i", c2, c2)  # row-wise squared norms
        c1_dot_c2 = np.einsum("ij,jk->ik", c1, c2)  # c1 @ c2.T (row-wise dot products)

        joined_tensor = np.stack(
            (
                np.broadcast_to(c1_norms, (len(c2_norms), len(c1_norms))).T,
                np.broadcast_to(c2_norms, (len(c1_norms), len(c2_norms))),
                c1_dot_c2,
            ),
            axis=2,
        )
        return np.apply_along_axis(self[3], axis=2, arr=joined_tensor)

    def __getitem__(self, indices):
        return partial_derivatives(self._kernel_func_expr, indices)
