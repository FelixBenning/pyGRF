from abc import abstractmethod
from numbers import Number
from functools import cache

import sympy as sym
import numpy as np

from pygrf.basis import OrthogonalBasis


@cache
def partial_derivatives(func, indices, dim=3, index_start=1):
    """Compute (higher order) partial derivatives of the function

    Example:
        indices = (1,2,2)
        dim = 3
        return ∂1 ∂2 ∂2 func(t_1, t_2, t_3)
    """
    # print(f"cache miss: partial_derivatives({func}, {indices}, {dim})")  # logging
    if isinstance(indices, tuple):
        _key = indices
    elif isinstance(indices, Number):
        _key = (indices,)
    else:
        raise NotImplementedError("indices should be Number or tuple of Numbers")

    t = sym.symbols(f"t{index_start}:{dim+index_start}")
    expr = func(*t)
    for idx in _key:
        expr = sym.diff(expr, t[idx - index_start])

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


class IsotropicKernel(Kernel):
    """(Non-stationary) isotropic kernel of the form

    k(x,y) = kernel_expression(norm(x)**2/2, norm(y)**2/2, dot(x,y))
    """

    def __init__(self, kernel_expression):
        self._kernel_func_expr = kernel_expression

    def covariance(self, c1, c2, /, *, derivatives=0, basis=None, new_dir=False):
        """Compute the covariance between coordinates c1 and c2 with
        respect to `basis', assume standard basis if basis=None.

        returns tensor:

        [i1, c1, i2, c2] = cov(D_{v_1} c_1[i1], D_{v_2} c_2[i2])

        for i1 in c_1
            for v1 in derivatives
                for i2 in c2
                    for v2 in derivatives
                        cov(D_{v_1} c_1[i1], D_{v_2} c_2[i2])

        if derivatives>0, also compute the derivatives
        """
        if (basis is not None) and (not isinstance(basis, OrthogonalBasis)):
            raise NotImplementedError("Only OrthogonalBasis are supported")

        if derivatives > 1:
            raise NotImplementedError("Only first order derivatives are supported")

        kernel_inputs = self.prepare_kernel_inputs(c1, c2)
        _c1 = np.atleast_2d(c1)
        _c2 = np.atleast_2d(c2)

        basis_len = _c1.shape[1]
        assert basis_len == _c2.shape[1]
        if basis:
            assert len(basis) == basis_len

        dloc = np.cumsum([basis_len**order for order in range(derivatives + 1)])
        # [1, 1+d, 1+d d^2, ...]
        # => dloc[ord-1]:dloc[ord] is the location of derivatives of order ord
        # e.g. dloc[0]:dloc[1] = 1:(1+d) is the location of the 1-th order derivatives

        result = np.empty((_c1.shape[0], dloc[-1], _c2.shape[0], dloc[-1]))

        # no derivative
        result[:, 0, :, 0] = np.apply_along_axis(self[()], axis=0, arr=kernel_inputs)
        if derivatives == 0:
            # pylint: disable=unexpected-keyword-arg
            # copy is a valid keyword argument (https://github.com/numpy/numpy/issues/27373)
            return result.reshape(_c1.shape[0], _c2.shape[0], copy=False)

        # derivatives == 1:

        ## ---- mix derivative and no derivative -----
        k_1 = np.apply_along_axis(self[1], axis=0, arr=kernel_inputs)
        k_2 = np.apply_along_axis(self[2], axis=0, arr=kernel_inputs)
        k_3 = np.apply_along_axis(self[3], axis=0, arr=kernel_inputs)
        result[:, 0, :, dloc[0] : dloc[1]] = np.einsum(
            "kl,lj->klj", k_2, _c2
        ) + np.einsum("kl,kj->klj", k_3, _c1)

        result[:, dloc[0] : dloc[1], :, 0] = np.einsum(
            "kl,ki->kil", k_1, _c1
        ) + np.einsum("kl,li->kil", k_3, _c2)
        ## ---- cov derivative with derivative -----

        k_12 = np.apply_along_axis(self[1, 2], axis=0, arr=kernel_inputs)
        k_13 = np.apply_along_axis(self[1, 3], axis=0, arr=kernel_inputs)
        k_23 = np.apply_along_axis(self[2, 3], axis=0, arr=kernel_inputs)
        k_33 = np.apply_along_axis(self[3, 3], axis=0, arr=kernel_inputs)

        result[:, dloc[0] : dloc[1], :, dloc[0] : dloc[1]] = (
            # derivative c1, c1 points, derivative c2, c2 points
            np.einsum("kl,ki,lj->kilj", k_12, _c1, _c2)
            + np.einsum("kl,ki,kj->kilj", k_13, _c1, _c1)
            + np.einsum("kl,li,lj->kilj", k_23, _c2, _c2)
            + np.einsum("kl,li,kj->kilj", k_33, _c2, _c1)
        ) + np.einsum("kl,ij->kilj", k_3, np.identity(basis_len))

        if new_dir:
            return result, k_3
        return result

    def prepare_kernel_inputs(self, c1, c2):
        _c1 = np.atleast_2d(c1)
        _c2 = np.atleast_2d(c2)
        c1_norms = np.einsum("ki,ki,->k", _c1, _c1, 0.5)  # row-wise squared norms
        c2_norms = np.einsum("lj,lj,->l", _c2, _c2, 0.5)  # row-wise squared norms
        c1_dot_c2 = np.einsum(
            "ki,li->kl", _c1, _c2
        )  # c1 @ c2.T (row-wise dot products)

        return np.stack(
            (
                np.broadcast_to(c1_norms, (len(c2_norms), len(c1_norms))).T,
                np.broadcast_to(c2_norms, (len(c1_norms), len(c2_norms))),
                c1_dot_c2,
            ),
            axis=0,
        )

    def covariance_new_dir_derivative(self, c1, c2, /, *, basis=None):
        joined_tensor = self.prepare_kernel_inputs(c1, c2)
        return np.apply_along_axis(self[3], axis=0, arr=joined_tensor)

    def __getitem__(self, indices):
        return lambda x: partial_derivatives(self._kernel_func_expr, indices)(*x)


class StationaryIsotropicKernel(IsotropicKernel):
    """Stationary isotropic kernel of the form

    k(x,y) = kernel_expression(-norm(x-y)**2/2)
    """

    def __init__(self, kernel_expression):
        def isotropic_kernel(sq_norm_half_1, sq_norm_half_2, dot):
            neg_sq_norm_dist_half = dot - sq_norm_half_1 - sq_norm_half_2
            return kernel_expression(neg_sq_norm_dist_half)

        super().__init__(isotropic_kernel)


class SquaredExponentialKernel(StationaryIsotropicKernel):
    """Squared Exponential Kernel

    k(x,y) = variance * exp(-norm(x-y)**2/(2*length_scale**2))
    """

    def __init__(self, variance=1.0, length_scale=1.0):
        self._variance = variance
        self._length_scale = length_scale
        super().__init__(lambda x: variance * sym.exp(x / length_scale**2))

    def __repr__(self):
        return f"SquaredExponentialKernel(variance={self._variance}, length_scale={self._length_scale})"
