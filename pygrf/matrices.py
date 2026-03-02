"""Implement special matrices"""

from numbers import Number
import numpy as np
import scipy as sp
from scipy import linalg


class ScaledIdentity:
    """A scaled square identity matrix"""

    __slots__ = "scale", "dim"

    def __init__(self, scale, dim):
        self.scale = scale
        self.dim = dim

    def __repr__(self):
        return f"ScaledIdentity(scale={self.scale}, dim={self.dim})"

    @property
    def shape(self):
        """Returns the shape of the matrix"""
        return self.dim, self.dim

    @property
    def T(self):  # pylint: disable=invalid-name
        """same as `self.transpose()'"""
        return self

    def transpose(self):
        """Transpose of the matrix"""
        return self

    def tosparse(self):
        """Convert to sparse matrix"""
        return sp.sparse.diags(np.full(self.dim, self.scale))

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("copy=False is not supported. A copy is always made.")
        return self.scale * np.eye(self.dim, dtype=dtype)

    def toarray(self):
        """Convert to dense array"""
        return self.__array__()

    def cholesky(self):
        """Cholesky decomposition of itself"""
        return ScaledIdentity(np.sqrt(self.scale), self.dim)

    def solve(self, other):
        return other / self.scale

    def __truediv__(self, other):
        if np.isscalar(other):
            return ScaledIdentity(self.scale / other, self.dim)

        raise NotImplementedError

    def __matmul__(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Matmul of shapes {self.shape} and {other.shape} not possible"
            )
        if isinstance(other, ScaledIdentity):
            return ScaledIdentity(self.scale * other.scale, self.dim)
        return self.scale * other

    def __rmatmul__(self, other):
        if self.shape[0] != other.shape[1]:
            raise ValueError(
                f"Matmul of shapes {other.shape} and {self.shape} not possible"
            )
        return self.scale * other

    def split_blocks(self, split_point):
        """Split the matrix into two blocks at split_point"""
        return (
            ScaledIdentity(self.scale, split_point),
            ScaledIdentity(self.scale, self.dim - split_point),
        )


class KiteMatrix:
    """A Matrix looking like a kite on a string,

    i.e. a dense upper left corner and a diagonal bottom right with identical elements
    """

    __slots__ = "dense", "diag", "lower"

    def __init__(self, dense, diag: ScaledIdentity, *, lower=None):
        self.dense = dense
        self.diag = diag
        self.lower = lower  # True: lower triangular, False: upper triangular, None: not triangular

    def __repr__(self):
        return f"KiteMatrix(dense={self.dense}, diag={self.diag}, lower={self.lower})"

    @property
    def shape(self):
        """Returns the shape of the matrix"""
        return (
            self.dense.shape[0] + self.diag.shape[0],
            self.dense.shape[1] + self.diag.shape[1],
        )

    @property
    def T(self):  # pylint: disable=invalid-name
        """same as self.transpose()"""
        return self.transpose()

    def transpose(self):
        """Transpose of the matrix"""
        return KiteMatrix(
            dense=self.dense.T,
            diag=self.diag.T,
            lower=not self.lower if self.lower is not None else None,
        )

    def __isub__(self, b):
        if isinstance(b, Number):
            self.dense -= b
            self.diag.scale -= b
            return self

        if not self.shape == b.shape:
            raise ValueError(
                f"can not subtract subtrahend with shape {b.shape} "
                + f"from minuent with shape {self.shape}"
            )

        if isinstance(b, KiteMatrix):
            b_shape = b.dense.shape
            self.ensure_dense_size(b_shape[0])  # assume square
            self.dense[: b_shape[0], : b_shape[1]] -= b.dense
            if (pad := self.dense.shape[0] - b_shape[0]) > 0:
                self.dense[-pad:, -pad:] -= b.diag.scale * np.eye(pad)
            self.diag.scale -= b.diag.scale
            return self

        raise NotImplementedError()

    def ensure_dense_size(self, dim):
        """ensure the size of the dense matrix is at least dim

        assuming a square dense matrix
        """
        pad = dim - self.dense.shape[0]
        if pad > 0:
            self.dense = np.pad(self.dense, pad_width=(0, pad), mode="constant")
            self.dense[-pad:, -pad:] = self.diag.scale * np.eye(pad)
            self.diag.dim -= pad

    def tosparse(self):
        """Convert to sparse matrix"""
        return sp.sparse.block_array([[self.dense, None], [None, self.diag.tosparse()]])

    def __array__(self, dtype=None, copy=None):
        return np.block(
            [
                [
                    np.asarray(self.dense, dtype=dtype, copy=copy),
                    np.zeros((self.dense.shape[0], self.diag.shape[1])),
                ],
                [
                    np.zeros((self.diag.shape[0], self.dense.shape[1])),
                    np.asarray(self.diag, dtype=dtype, copy=copy),
                ],
            ]
        )

    def toarray(self):
        """Convert to dense array"""
        return self.tosparse().toarray()

    def __matmul__(self, other):
        if not self.shape[1] == other.shape[0]:
            raise ValueError(
                f"Matmul of shapes {self.shape} and {other.shape} not possible"
            )

        if isinstance(other, KiteMatrix):

            dcols_l = self.dense.shape[1]  # dense columns of left
            drows_r = other.dense.shape[0]  # dense rows of right

            if dcols_l == drows_r:
                return KiteMatrix(
                    dense=self.dense @ other.dense,
                    diag=self.diag @ other.diag,
                )

            elif dcols_l > drows_r:
                # padd smaller dense matrix by using some of the sparse part
                middle, rest_diag = other.diag.split_blocks(dcols_l - drows_r)
                return KiteMatrix(
                    dense=np.concatenate(
                        (
                            self.dense[:, 0:drows_r] @ other.dense,  # fully dense
                            # unfortunately, dense matrices do not seem to raise NotImplemented
                            # such that __rmatmul__ is called %TODO: better solution
                            # pylint: disable=unnecessary-dunder-call
                            middle.__rmatmul__(
                                self.dense[:, drows_r:]
                            ),  # dense * sparse
                        ),
                        axis=1,
                    ),
                    diag=self.diag @ rest_diag,
                )
            else:  # dcols_l < drows_r:
                middle, rest_diag = self.diag.split_blocks(drows_r - dcols_l)
                return KiteMatrix(
                    dense=np.concatenate(
                        (
                            self.dense @ other.dense[0:dcols_l, :],  # fully dense
                            middle @ other.dense[dcols_l:, :],  # sparse * dense
                        ),
                        axis=0,
                    ),
                    diag=rest_diag @ other.diag,
                )

        # other is not KiteMatrix, split in blocks and multiply block wise
        n = self.dense.shape[1]
        return np.concatenate((self.dense @ other[0:n], self.diag @ other[n:]), axis=0)

    def cholesky(
        self, lower=True, overwrite_self=False, check_finite=True
    ) -> "KiteMatrix":
        """Cholesky decomposition of the matrix, assuming it is square"""
        if overwrite_self:
            self.dense = linalg.cholesky(
                self.dense,
                lower=lower,
                overwrite_a=True,
                check_finite=check_finite,
            )
            self.diag = self.diag.cholesky()
            self.lower = lower
            return self

        return KiteMatrix(
            dense=linalg.cholesky(
                self.dense,
                lower=lower,
                check_finite=check_finite,
            ),
            diag=self.diag.cholesky(),
            lower=lower,
        )

    def standard_deviation_mat(
        self, overwrite_self=False, check_finite=True
    ) -> "KiteMatrix":
        """Returns a matrix R such that Cov(R @ Y) == self for Y ~ N(0, I),
        assuming self is positive semi-definite."""
        u, s, _ = linalg.svd(
            self.dense, check_finite=check_finite, overwrite_a=overwrite_self
        )
        s_sqrt = np.sqrt(s)
        return KiteMatrix(dense=(u * s_sqrt), diag=self.diag.cholesky(), lower=None)

    def cho_factor(self, lower=True, overwrite_self=False, check_finite=True):
        """Cholesky decomposition

        Warning: does not zero the lower/upper triangle. For that use cholesky

        calls scipy.linalg.cho_factor under the hood, which is a wrapper of
        lapack's POTRF, https://oneapi-src.github.io/oneMKL/domains/lapack/potrf.html

        Important:
            POTRF guarantees that the upper triangular part is not referenced if lower=True
            and the lower triangular part is not referenced if lower=False
        """
        if overwrite_self:
            sp.linalg.cho_factor(
                self.dense, overwrite_a=True, lower=lower, check_finite=check_finite
            )
            self.diag = self.diag.cholesky()
            self.lower = lower
            return self

        return KiteMatrix(
            dense=sp.linalg.cho_factor(
                self.dense, overwrite_a=False, lower=lower, check_finite=check_finite
            ),
            diag=self.diag.cholesky(),
            lower=lower,
        )

    def solve_triangular(self, other, trans="N", inplace=False, check_finite=True):
        """Solve the linear system self @ x = b"""
        if self.lower is None:
            raise ValueError("Matrix is not triangular")

        if isinstance(other, KiteMatrix):
            dcols_l = self.dense.shape[1]  # dense columns of left
            drows_r = other.dense.shape[0]  # dense rows of right

            if dcols_l == drows_r:
                return KiteMatrix(
                    dense=sp.linalg.solve_triangular(
                        self.dense,
                        other.dense,
                        lower=self.lower,
                        trans=trans,
                        check_finite=check_finite,
                        overwrite_b=inplace,
                    ),
                    diag=self.diag.solve(other.diag),
                )

            if dcols_l > drows_r:
                # since this will be rarely used anyway, simply enlarge dense matrix on the right
                middle, rest_diag = other.diag.split_blocks(dcols_l - drows_r)
                return KiteMatrix(
                    dense=sp.linalg.solve_triangular(
                        self.dense,
                        KiteMatrix(dense=other.dense, diag=middle).toarray(),
                        lower=self.lower,
                        trans=trans,
                        check_finite=check_finite,
                        overwrite_b=inplace,
                    ),
                    diag=self.diag.solve(rest_diag),
                )
            # dcols_l < drows_r:
            middle, rest_diag = self.diag.split_blocks(drows_r - dcols_l)
            return KiteMatrix(
                dense=np.concatenate(
                    (
                        sp.linalg.solve_triangular(
                            self.dense,
                            other.dense[0:dcols_l],
                            lower=self.lower,
                            trans=trans,
                            check_finite=check_finite,
                            overwrite_b=inplace,
                        ),
                        middle.solve(other.dense[dcols_l:]),
                    ),
                    axis=0,
                ),
                diag=rest_diag.solve(other.diag),
            )

        return np.concatenate(
            (
                sp.linalg.solve_triangular(
                    self.dense,
                    other[0 : self.dense.shape[1]],
                    lower=self.lower,
                    trans=trans,
                    check_finite=check_finite,
                ),
                self.diag.solve(other[self.dense.shape[1] :]),
            ),
            axis=0,
        )

    def __rmatmul__(self, other):
        # other is not KiteMatrix (otherwise __matmul__ would have been used),
        # split in blocks and multiply block wise
        n = self.dense.shape[0]
        return np.concatenate(
            (other[:, 0:n] @ self.dense, other[:, n:] @ self.diag), axis=0
        )
