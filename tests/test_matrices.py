# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from pygrf.matrices import KiteMatrix


@pytest.fixture
def random_kitematrix_tuple(request):
    """Return a random KiteMatrix"""
    rng = np.random.default_rng(request.param)
    rcol, joint, lrow = rng.integers(low=2, high=100, size=3)

    rdiag = rng.integers(low=1, high=min(rcol, joint))
    ldiag = rng.integers(low=1, high=min(joint, lrow))

    rmat = KiteMatrix(
        dense=rng.random((rcol-rdiag, joint-rdiag)),
        sparse=(rng.random(), rdiag),
    )
    lmat = KiteMatrix(
        dense=rng.random((joint-ldiag, lrow-ldiag)),
        sparse=(rng.random(), ldiag),
    )

    return rmat, lmat

def test_kitematrix_matmul_handcrafted():
    """Test matrix multiplication with handcrafted examples"""
    mat1 = KiteMatrix(dense=np.array([[1, 2], [3, 4]]), sparse=(1,2))
    mat2 = KiteMatrix(dense=np.array([[-1, 5], [6, 7]]), sparse=(3,2))
    assert (mat1.toarray() @ mat2.toarray() == (mat1 @ mat2).toarray()).all()

    mat1 = KiteMatrix(dense=np.array([[1, 2], [3, 4]]), sparse=(1, 3))
    mat2 = KiteMatrix(dense=np.array([[-1, 5, 2], [6, 7, 2], [1,2,3]]), sparse=(3, 2))
    assert (mat1.toarray() @ mat2.toarray() == (mat1 @ mat2).toarray()).all()

@pytest.mark.parametrize("random_kitematrix_tuple", range(100), indirect=True)
def test_kitematrix_matmul_random(random_kitematrix_tuple):
    """Test matrix multiplication with random input"""
    mat1 = KiteMatrix(dense=np.array([[1, 2], [3, 4]]), sparse=(1,2))
    mat2 = KiteMatrix(dense=np.array([[-1, 5], [6, 7]]), sparse=(3,2))
    assert (mat1.toarray() @ mat2.toarray() == (mat1 @ mat2).toarray()).all()

    mat1 = KiteMatrix(dense=np.array([[1, 2], [3, 4]]), sparse=(1, 3))
    mat2 = KiteMatrix(dense=np.array([[-1, 5, 2], [6, 7, 2], [1,2,3]]), sparse=(3, 2))
    assert (mat1.toarray() @ mat2.toarray() == (mat1 @ mat2).toarray()).all()

    matl, matr = random_kitematrix_tuple
    res1 = matl.toarray() @ matr.toarray()
    res2= (matl @ matr).toarray()
    assert np.linalg.matrix_norm(res1 - res2) == pytest.approx(0)
