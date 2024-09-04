""" Tests for matrices.py """

# pylint: disable=redefined-outer-name
from typing import Tuple
import pytest

import scipy as sp
import numpy as np

from pygrf.matrices import KiteMatrix, ScaledIdentity

def _random_kitematrix(rows, cols, rng=np.random.default_rng()):
    """Return a random KiteMatrix of shape rows x cols"""
    diag_dim = rng.integers(low=0, high=min(cols, rows))
    return KiteMatrix(
        dense=rng.random((rows-diag_dim, cols-diag_dim)),
        diag=ScaledIdentity(rng.random(), diag_dim),
    )

@pytest.fixture
def random_kitematrix_tuple(request):
    """
    Return two random Kite Matrices

    one with dimensions k x n and the other with dimensions n x m 
    for random (k,n,m)

    generated using request.param as a seed
    """
    rng = np.random.default_rng(request.param)
    k, n, m = rng.integers(low=1, high=100, size=3)
    return _random_kitematrix(k,n, rng), _random_kitematrix(n,m, rng)

@pytest.fixture
def random_kitematrix_chol(request):
    """
    Return a random, positive definite Kite Matrix
    with dimensions n x n for random n

    generated using request.param as a seed
    """
    rng = np.random.default_rng(request.param)
    n, k = rng.integers(low=1, high=100, size=2)
    mat_a = _random_kitematrix(n,n, rng)
    mat_b = _random_kitematrix(n,k, rng)
    return mat_a @ mat_a.T, mat_b

@pytest.mark.parametrize("random_kitematrix_tuple", range(100), indirect=True)
def test_random_matrix_fixture(random_kitematrix_tuple):
    """Test the fixture for generating random matrices and shape"""
    matl, matr = random_kitematrix_tuple
    # check if the dimensions match
    assert matl.shape[1] == matr.shape[0]

def test_kitematrix_matmul_handcrafted():
    """Test matrix multiplication with handcrafted examples"""
    mat1 = KiteMatrix(dense=np.array([[1, 2], [3, 4]]), diag=ScaledIdentity(1,2))
    mat2 = KiteMatrix(dense=np.array([[-1, 5], [6, 7]]), diag=ScaledIdentity(3,2))
    assert np.allclose(mat1.toarray() @ mat2.toarray(), (mat1 @ mat2).toarray())

    mat1 = KiteMatrix(dense=np.array([[1, 2], [3, 4]]), diag=ScaledIdentity(1, 3))
    mat2 = KiteMatrix(dense=np.array([[-1, 5, 2], [6, 7, 2], [1,2,3]]), diag=ScaledIdentity(3, 2))
    assert np.allclose(mat1.toarray() @ mat2.toarray(), (mat1 @ mat2).toarray())

@pytest.mark.parametrize("random_kitematrix_tuple", range(100), indirect=True)
def test_kitematrix_matmul_random(random_kitematrix_tuple):
    """Test matrix multiplication with random input"""
    matl, matr = random_kitematrix_tuple
    res1 = matl.toarray() @ matr.toarray()
    res2= (matl @ matr).toarray()
    assert np.allclose(res1, res2)

@pytest.mark.parametrize("random_kitematrix_chol", range(100), indirect=True)
def test_kitematrix_cholesky_random(random_kitematrix_chol):
    """Test cholesky decomposition with random input"""
    mat_sq, mat_b = random_kitematrix_chol
    lower = True
    chol_expected = sp.linalg.cholesky(mat_sq.toarray(), lower=lower)
    solve_expected = sp.linalg.cho_solve((chol_expected, lower), mat_b.toarray())

    chol_actual:KiteMatrix = mat_sq.cholesky()
    assert np.allclose(chol_expected, chol_actual.toarray())

    solve_actual = chol_actual.cho_solve(mat_b)
    assert np.allclose(solve_expected, solve_actual.toarray())


    # chol_expected = sp.linalg.cho_factor(mat_sq.toarray())
    # chol_actual = mat_sq.cho_factor().toarray()
    # assert np.allclose(chol_expected, chol_actual)



    res3 = np.linalg.cholesky(np.tril(mat_sq.toarray()))
    assert np.allclose(chol_expected,res3)



