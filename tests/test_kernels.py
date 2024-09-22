""" Test kernels.py """

# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from pygrf.kernels import SquaredExponentialKernel

# regtest
# https://gitlab.com/uweschmitt/pytest-regtest/-/blob/main/docs/index.md

def test_squared_exponential(regtest):
    """Test the Squared Exponential Kernel"""
    k = SquaredExponentialKernel() 
    std_basis = np.array([[1,0],[0,1]])
    res = k.covariance(std_basis, std_basis)
    print(res, file=regtest)

    res = k.covariance(std_basis, std_basis, derivatives=1)
    res = res.reshape(res.shape[0]*res.shape[1], res.shape[2]*res.shape[3])
    print(res, file=regtest)
