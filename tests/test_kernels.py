""" Test kernels.py """

# pylint: disable=redefined-outer-name
import pytest
import numpy as np
import scipy as sp

from pygrf.kernels import SquaredExponentialKernel

# regtest
# https://gitlab.com/uweschmitt/pytest-regtest/-/blob/main/docs/index.md


def test_squared_exponential(regtest):
    """Test the Squared Exponential Kernel"""
    k = SquaredExponentialKernel()
    std_basis = np.array([[1, 0], [0, 1]])
    res = k.covariance(std_basis, std_basis)
    print(res, file=regtest)

    res = k.covariance(std_basis, std_basis, derivatives=1)
    res = res.reshape(res.shape[0] * res.shape[1], res.shape[2] * res.shape[3])
    print(res, file=regtest)


@pytest.mark.parametrize("seed", range(100))
def test_squared_exponential_random(seed):
    rng = np.random.default_rng(seed)
    c1_len = rng.integers(low=1, high=100)
    c2_len = rng.integers(low=1, high=100)
    dim = rng.integers(low=1, high=100)
    c1 = rng.normal((c1_len, dim))
    c2 = rng.normal((c2_len, dim))

    var = rng.normal() ** 2
    length_scale = rng.normal() ** 2

    k = SquaredExponentialKernel(variance=var, length_scale=length_scale)
    k_std = SquaredExponentialKernel()

    autocov = k.covariance(c1, c1, derivatives=1)
    autocov = autocov.reshape(autocov.shape[0] * autocov.shape[1], -1)
    assert sp.linalg.issymmetric(autocov)

    try:
        np.linalg.cholesky(autocov)
    except np.linalg.LinAlgError:
        pytest.fail("Autocovariance not positive (semi-)definite")

    cov = k.covariance(c1, c2)
    cov_std = k_std.covariance(c1 / length_scale, c2 / length_scale)
    assert np.allclose(cov, var * cov_std)
