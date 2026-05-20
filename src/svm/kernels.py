"""
src/svm/kernels.py
------------------
Manual implementations of all SVM kernel functions.
Kernels map input data into a higher-dimensional feature space,
enabling non-linear classification without explicit transformation.
No Scikit-learn used.
"""

import numpy as np


# ──────────────────────────────────────────────
# Kernel Functions
# ──────────────────────────────────────────────

def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Linear Kernel — standard dot product.
    Formula: K(x1, x2) = x1 · x2

    Best for: Linearly separable data.
    """
    return float(np.dot(x1, x2))


def polynomial_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    degree: int   = 3,
    coef0:  float = 1.0,
    gamma:  float = 1.0,
) -> float:
    """
    Polynomial Kernel.
    Formula: K(x1, x2) = (γ·x1·x2 + c)^d

    Parameters
    ----------
    degree : int   — polynomial degree d              (default: 3)
    coef0  : float — independent term c               (default: 1.0)
    gamma  : float — scale factor γ                   (default: 1.0)

    Best for: Problems with polynomial feature interactions.
    """
    return float((gamma * np.dot(x1, x2) + coef0) ** degree)


def rbf_kernel(
    x1:    np.ndarray,
    x2:    np.ndarray,
    gamma: float = None,
) -> float:
    """
    Radial Basis Function (RBF / Gaussian) Kernel.
    Formula: K(x1, x2) = exp( -γ · ||x1 - x2||² )

    Parameters
    ----------
    gamma : float — bandwidth parameter γ.
                    If None, uses 1 / n_features.

    Best for: General non-linear problems; most popular kernel.
    """
    if gamma is None:
        gamma = 1.0 / len(x1)
    diff = x1 - x2
    return float(np.exp(-gamma * np.dot(diff, diff)))


def sigmoid_kernel(
    x1:    np.ndarray,
    x2:    np.ndarray,
    gamma: float = 0.01,
    coef0: float = 0.0,
) -> float:
    """
    Sigmoid (tanh) Kernel — inspired by neural network activation.
    Formula: K(x1, x2) = tanh(γ·x1·x2 + c)

    Note: Not always a valid Mercer kernel; use with caution.
    """
    return float(np.tanh(gamma * np.dot(x1, x2) + coef0))


def laplacian_kernel(
    x1:    np.ndarray,
    x2:    np.ndarray,
    gamma: float = 1.0,
) -> float:
    """
    Laplacian Kernel — uses L1 norm instead of L2.
    Formula: K(x1, x2) = exp( -γ · ||x1 - x2||₁ )

    More robust to outliers than RBF.
    """
    return float(np.exp(-gamma * np.sum(np.abs(x1 - x2))))


# ──────────────────────────────────────────────
# Kernel Matrix (Gram Matrix)
# ──────────────────────────────────────────────

def compute_kernel_matrix(
    X1:          np.ndarray,
    X2:          np.ndarray,
    kernel_fn,
    **kernel_params,
) -> np.ndarray:
    """
    Compute the full kernel (Gram) matrix K where K[i,j] = kernel(X1[i], X2[j]).

    Parameters
    ----------
    X1, X2     : arrays, shapes (n1, d) and (n2, d)
    kernel_fn  : one of the kernel functions above
    **kernel_params : passed directly to kernel_fn

    Returns
    -------
    K : np.ndarray, shape (n1, n2)
    """
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j], **kernel_params)
    return K


def compute_kernel_matrix_fast(
    X1:     np.ndarray,
    X2:     np.ndarray,
    kernel: str  = "rbf",
    gamma:  float = None,
    degree: int   = 3,
    coef0:  float = 1.0,
) -> np.ndarray:
    """
    Vectorised kernel matrix computation for linear, polynomial, and RBF.
    Significantly faster than the loop version for large datasets.

    Parameters
    ----------
    X1, X2  : arrays
    kernel  : 'linear' | 'polynomial' | 'rbf'
    """
    if kernel == "linear":
        return X1 @ X2.T

    elif kernel == "polynomial":
        return (gamma * (X1 @ X2.T) + coef0) ** degree if gamma else ((X1 @ X2.T) + coef0) ** degree

    elif kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / X1.shape[1]
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1·x2
        sq_dist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1)
            - 2 * (X1 @ X2.T)
        )
        sq_dist = np.maximum(sq_dist, 0)  # numerical safety
        return np.exp(-gamma * sq_dist)

    else:
        raise ValueError(f"Fast mode not available for kernel='{kernel}'. "
                         "Use compute_kernel_matrix() with a custom function.")


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

_KERNEL_REGISTRY = {
    "linear":     linear_kernel,
    "polynomial": polynomial_kernel,
    "poly":       polynomial_kernel,
    "rbf":        rbf_kernel,
    "gaussian":   rbf_kernel,
    "sigmoid":    sigmoid_kernel,
    "laplacian":  laplacian_kernel,
}


def get_kernel(name: str):
    """
    Returns the kernel function for the given name.

    Parameters
    ----------
    name : str — 'linear' | 'polynomial' | 'rbf' | 'sigmoid' | 'laplacian'

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, ...], float]
    """
    key = name.strip().lower()
    if key not in _KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel '{name}'. "
            f"Available: {list(_KERNEL_REGISTRY.keys())}"
        )
    return _KERNEL_REGISTRY[key]


def list_kernels() -> list:
    """Returns all available kernel names."""
    return list(_KERNEL_REGISTRY.keys())


# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    print("Kernel Function Tests")
    print("─" * 40)
    print(f"  linear      K(a,b) = {linear_kernel(a, b):.4f}")
    print(f"  polynomial  K(a,b) = {polynomial_kernel(a, b, degree=3):.4f}")
    print(f"  rbf         K(a,b) = {rbf_kernel(a, b, gamma=0.1):.4f}")
    print(f"  sigmoid     K(a,b) = {sigmoid_kernel(a, b):.4f}")
    print(f"  laplacian   K(a,b) = {laplacian_kernel(a, b):.4f}")

    print("\nKernel Matrix (RBF, fast vectorised):")
    X = np.random.randn(4, 3)
    K = compute_kernel_matrix_fast(X, X, kernel="rbf", gamma=0.5)
    print(K.round(4))