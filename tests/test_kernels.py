"""tests/test_kernels.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.svm.kernels import (linear_kernel, polynomial_kernel, rbf_kernel,
                              sigmoid_kernel, compute_kernel_matrix_fast, get_kernel)

class TestKernels:
    def test_linear_symmetric(self):
        a, b = np.array([1., 2., 3.]), np.array([4., 5., 6.])
        assert linear_kernel(a, b) == pytest.approx(linear_kernel(b, a))

    def test_rbf_self_is_one(self):
        a = np.array([1., 2., 3.])
        assert rbf_kernel(a, a, gamma=0.5) == pytest.approx(1.0)

    def test_rbf_decreases_with_distance(self):
        a = np.array([0., 0.])
        b = np.array([1., 0.])
        c = np.array([10., 0.])
        assert rbf_kernel(a, b) > rbf_kernel(a, c)

    def test_polynomial_degree_1_equals_linear(self):
        a, b = np.array([1., 2.]), np.array([3., 4.])
        poly1 = polynomial_kernel(a, b, degree=1, coef0=0, gamma=1)
        lin   = linear_kernel(a, b)
        assert poly1 == pytest.approx(lin)

    def test_fast_kernel_matrix_shape(self):
        X = np.random.randn(10, 4)
        K = compute_kernel_matrix_fast(X, X, kernel="rbf")
        assert K.shape == (10, 10)

    def test_fast_kernel_matrix_symmetric_rbf(self):
        X = np.random.randn(8, 3)
        K = compute_kernel_matrix_fast(X, X, kernel="rbf")
        assert np.allclose(K, K.T, atol=1e-10)

    def test_get_kernel_invalid_raises(self):
        with pytest.raises(ValueError):
            get_kernel("unknown_kernel")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])