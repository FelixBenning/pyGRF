""" Tests of basis.py """

# pylint: disable=redefined-outer-name
import pytest

import scipy as sp
import numpy as np

from pygrf.basis import Basis, CoordinateVec, OrthogonalBasis


def random_basis(dim, rng=np.random.default_rng(), orthogonal=False):
    """Return a random basis matrix of shape dim x dim"""
    mat = rng.random((dim, dim))
    while np.linalg.det(mat) == 0:
        mat = rng.random((dim, dim))

    if orthogonal:
        return OrthogonalBasis(sp.linalg.qr(mat)[0])
    return Basis(mat)


@pytest.mark.parametrize("seed", range(100))
def test_basis_change_random(seed):
    """ Test round-trip basis change and vector ops """
    rng = np.random.default_rng(seed)
    dim = rng.integers(low=1, high=100)

    for orthogonal in [False, True]:
        basis = random_basis(dim, rng, orthogonal)
        vec1, vec2 = rng.random(dim), rng.random(dim)

        # simple round trip test
        coord_vec1 = basis.into_basis(vec1)
        assert np.allclose(vec1, coord_vec1.in_std_basis())

        # test operations
        assert np.allclose(vec1, np.zeros(dim) + coord_vec1)

        # addition/subtraction
        coord_sum = coord_vec1 + vec2
        assert isinstance(coord_sum, CoordinateVec)
        assert np.allclose(vec1 + vec2, coord_sum.in_std_basis())
        assert np.allclose(vec1 - vec2, (coord_vec1-vec2).in_std_basis())

        # scalar multiplication
        coeff = rng.random()
        assert np.allclose(coeff * vec1, (coeff * coord_vec1).in_std_basis())
        assert np.allclose(coeff * vec1, (coord_vec1 * coeff).in_std_basis())

 
