""" Implement special matrices """

import numpy as np
import scipy.sparse as sp


class ScaledIdentity:
    """A scaled square identity matrix"""

    __slots__ = "scale", "dim"

    def __init__(self, scale, dim):
        self.scale = scale
        self.dim = dim

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
        return sp.diags(np.full(self.dim, self.scale))

    def toarray(self):
        """Convert to dense array"""
        return self.tosparse().toarray()

    def cholesky(self):
        """Cholesky decomposition of itself"""
        return ScaledIdentity(np.sqrt(self.scale), self.dim)

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


def _matmul_kitematrix(left: "KiteMatrix", right: "KiteMatrix"):
    """Matrix multiplication of two KiteMatrices

    i.e. left @ right

    used to implement __matmul__ and __rmatmul__ of KiteMatrix
    """

    if not left.shape[1] == right.shape[0]:
        raise ValueError(f"Matmul of shapes {left.shape} and {right.shape} not possible")

    dcols_l = left.dense.shape[1]  # dense columns of left
    drows_r = right.dense.shape[0]  # dense rows of right

    if dcols_l == drows_r:
        return KiteMatrix(
            dense=left.dense @ right.dense,
            diag=left.diag @ right.diag,
        )

    # padd smaller dense matrix by using some of the sparse part
    elif dcols_l > drows_r:
        middle, lower = right.diag.split_blocks(dcols_l - drows_r)
        return KiteMatrix(
            dense=np.concatenate(
                (
                    left.dense[:, 0:drows_r] @ right.dense,  # fully dense
                    # unfortunately, dense matrices do not seem to raise NotImplemented
                    # such that __rmatmul__ is called %TODO: better solution
                    # pylint: disable=unnecessary-dunder-call
                    middle.__rmatmul__(left.dense[:, drows_r:]),  # dense * sparse
                ),
                axis=1,
            ),
            diag=left.diag @ lower,
        )
    else:  # dcols_l < drows_r:
        middle, lower = left.diag.split_blocks(drows_r - dcols_l)
        return KiteMatrix(
            dense=np.concatenate(
                (
                    left.dense @ right.dense[0:dcols_l, :],  # fully dense
                    middle @ right.dense[dcols_l:, :],  # sparse * dense
                ),
                axis=0,
            ),
            diag=lower @ right.diag,
        )


class KiteMatrix:
    """A Matrix looking like a kite on a string,

    i.e. a dense upper left corner and a diagonal bottom right with identical elements
    """

    __slots__ = "dense", "diag"

    def __init__(self, dense, diag: ScaledIdentity):
        self.dense = dense
        self.diag = diag

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
        )

    def tosparse(self):
        """Convert to sparse matrix"""
        return sp.block_array([[self.dense, None], [None, self.diag.tosparse()]])

    def toarray(self):
        """Convert to dense array"""
        return self.tosparse().toarray()

    def __matmul__(self, other):
        if isinstance(other, KiteMatrix):
            return _matmul_kitematrix(self, other)

        # other is not KiteMatrix, split in blocks and multiply block wise
        n = self.dense.shape[1]
        return np.concatenate(
            (self.dense @ other[0:n, :], self.diag @ other[n:, :]), axis=0
        )

    def cholesky(self):
        """Cholesky decomposition of the matrix, assuming it is square"""
        return KiteMatrix(
            dense=np.linalg.cholesky(self.dense),
            diag=self.diag.cholesky(),
        )

    def __rmatmul__(self, other):
        # other is not KiteMatrix (otherwise __matmul__ would have been used),
        # split in blocks and multiply block wise
        n = self.dense.shape[0]
        return np.concatenate(
            (other[:, 0:n] @ self.dense, other[:, n:] @ self.diag), axis=0
        )
