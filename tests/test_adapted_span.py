""" Test adapted_span.py """

# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from pygrf.adapted_span import LazyAdaptedSpan

from .test_basis import random_basis


@pytest.mark.parametrize("seed", range(100))
def test_adapted_span(seed):
    """Test the LazyAdaptedSpan class"""
    rng = np.random.default_rng(seed)

    dim = rng.integers(low=1, high=100)
    vecs = rng.random((dim, dim))

    adapted_span = LazyAdaptedSpan(dim=dim, rng=rng)
    assert len(adapted_span) == 0

    # vector into coordinates should be added to lazy span
    c1 = adapted_span.coeff_from_std_basis(vecs[0])
    assert c1[0] == pytest.approx(np.linalg.norm(vecs[0]))
    assert len(adapted_span) == 1

    # getting the coefficients again should not change eager length 
    adapted_span.coeff_from_std_basis(vecs[0])
    assert len(adapted_span) == 1

    coeff_mat = adapted_span.coeff_from_std_basis(vecs)
    assert np.allclose(vecs, adapted_span.coeff_into_std_basis(coeff_mat))
