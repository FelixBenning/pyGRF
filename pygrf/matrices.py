""" Implement special matrices """

import numpy as np
import scipy.sparse as sp


def _matmul_kitematrix(left, right):
    """Matrix multiplication of two KiteMatrices

    i.e. left @ right

    used to implement __matmul__ and __rmatmul__ of KiteMatrix
    """
    dcols_l = left.dense.shape[1]  # dense columns of left
    drows_r = right.dense.shape[0]  # dense rows of right

    if not (dcols_l + left.sparse_dim) == (drows_r + right.sparse_dim):
        raise ValueError("Matrix dimensions do not match")

    if dcols_l == drows_r:
        return KiteMatrix(
            dense=left.dense @ right.dense,
            sparse=(left.sparse_val * right.sparse_val, left.sparse_dim),
        )

    # padd smaller dense matrix by using some of the sparse part
    elif dcols_l > drows_r:
        # right is smaller
        return KiteMatrix(
            dense=np.concatenate(
                (
                    left.dense[:, 0:drows_r] @ right.dense,  # fully dense
                    right.sparse_val * left.dense[:, drows_r:],  # dense * sparse
                ),
                axis=1,
            ),
            sparse=(left.sparse_val * right.sparse_val, left.sparse_dim),
        )
    else:  # dcols_l < drows_r:
        return KiteMatrix(
            dense=np.concatenate(
                (
                    left.dense @ right.dense[0:dcols_l, :],  # fully dense
                    left.sparse_val * right.dense[dcols_l:, :],  # sparse * dense
                ),
                axis=0,
            ),
            sparse=(left.sparse_val * right.sparse_val, right.sparse_dim),
        )


class KiteMatrix:
    """A Matrix looking like a kite on a string,

    i.e. a dense upper left corner and a diagonal bottom right with identical elements
    """

    __slots__ = "dense", "sparse_val", "sparse_dim"

    def __init__(self, dense, sparse):
        self.dense = dense
        self.sparse_val, self.sparse_dim = sparse

    def tosparse(self):
        """Convert to sparse matrix"""
        return sp.block_array(
            [
                [self.dense, None],
                [None, sp.diags(np.repeat(self.sparse_val, self.sparse_dim))],
            ]
        )

    def toarray(self):
        """Convert to dense array"""
        return self.tosparse().toarray()

    def __matmul__(self, other):
        if isinstance(other, KiteMatrix):
            return _matmul_kitematrix(self, other)

        # other is not KiteMatrix, split in blocks and multiply block wise
        n = self.dense.shape[1]
        return np.concatenate(
            (self.dense @ other[0:n, :], self.sparse_val * other[n:, :]), axis=0
        )

    def __rmatmul__(self, other):
        # other is not KiteMatrix (otherwise __matmul__ would have been used),
        # split in blocks and multiply block wise
        n = self.dense.shape[0]
        return np.concatenate(
            (other[:, 0:n] @ self.dense, self.sparse_val * other[:, n:]), axis=0
        )
