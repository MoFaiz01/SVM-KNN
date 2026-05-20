"""
src/knn/distance_metrics.py
----------------------------
Manual implementations of all distance metrics used in k-NN.
No Scikit-learn or external ML libraries.
"""

import numpy as np


# ──────────────────────────────────────────────
# Core Distance Functions
# ──────────────────────────────────────────────

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    L2 (Euclidean) distance.
    Formula: sqrt( Σ (x1_i - x2_i)² )
    """
    return float(np.sqrt(np.sum((x1 - x2) ** 2)))


def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    L1 (Manhattan / City-block) distance.
    Formula: Σ |x1_i - x2_i|
    """
    return float(np.sum(np.abs(x1 - x2)))


def minkowski_distance(x1: np.ndarray, x2: np.ndarray, p: int = 3) -> float:
    """
    Minkowski distance — generalises Euclidean (p=2) and Manhattan (p=1).
    Formula: ( Σ |x1_i - x2_i|^p )^(1/p)
    """
    return float(np.sum(np.abs(x1 - x2) ** p) ** (1.0 / p))


def chebyshev_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Chebyshev (L∞) distance — maximum element-wise absolute difference.
    Formula: max( |x1_i - x2_i| )
    """
    return float(np.max(np.abs(x1 - x2)))


def hamming_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Hamming distance — proportion of positions where values differ.
    Intended for categorical / binary features.
    Formula: (1/n) * Σ 𝟙[x1_i ≠ x2_i]
    """
    return float(np.sum(x1 != x2) / len(x1))


def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Cosine distance = 1 − cosine_similarity.
    Measures the angle between two vectors.
    Formula: 1 − (x1·x2) / (||x1|| * ||x2||)
    """
    norm = np.linalg.norm(x1) * np.linalg.norm(x2)
    if norm == 0:
        return 1.0
    return float(1.0 - np.dot(x1, x2) / norm)


def mahalanobis_distance(x1: np.ndarray, x2: np.ndarray,
                          cov_inv: np.ndarray = None) -> float:
    """
    Mahalanobis distance — accounts for feature correlations.
    Formula: sqrt( (x1-x2)^T * Σ^{-1} * (x1-x2) )
    If cov_inv is None, falls back to Euclidean distance.
    """
    diff = x1 - x2
    if cov_inv is None:
        return float(np.sqrt(np.dot(diff, diff)))
    return float(np.sqrt(diff @ cov_inv @ diff))


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

DISTANCE_FUNCTIONS = {
    "euclidean":  euclidean_distance,
    "manhattan":  manhattan_distance,
    "minkowski":  minkowski_distance,
    "chebyshev":  chebyshev_distance,
    "hamming":    hamming_distance,
    "cosine":     cosine_distance,
    "mahalanobis": mahalanobis_distance,
}


def get_distance_function(name: str):
    """
    Returns the distance callable for the given metric name.

    Parameters
    ----------
    name : str  — one of: euclidean, manhattan, minkowski,
                  chebyshev, hamming, cosine, mahalanobis

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]

    Raises
    ------
    ValueError  — if metric name is unrecognised
    """
    key = name.strip().lower()
    if key not in DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Unknown metric '{name}'. "
            f"Available: {list(DISTANCE_FUNCTIONS.keys())}"
        )
    return DISTANCE_FUNCTIONS[key]


def list_metrics() -> list:
    """Returns all available metric names."""
    return list(DISTANCE_FUNCTIONS.keys())


# ──────────────────────────────────────────────
# Vectorised pairwise distance (performance helper)
# ──────────────────────────────────────────────

def pairwise_distances(X_train: np.ndarray,
                       x_test: np.ndarray,
                       metric: str = "euclidean") -> np.ndarray:
    """
    Computes distances from a single test point to all training points.
    Uses vectorised NumPy operations — much faster than a Python loop.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    x_test  : np.ndarray, shape (n_features,)
    metric  : str

    Returns
    -------
    distances : np.ndarray, shape (n_train,)
    """
    if metric == "euclidean":
        diff = X_train - x_test
        return np.sqrt(np.sum(diff ** 2, axis=1))

    elif metric == "manhattan":
        return np.sum(np.abs(X_train - x_test), axis=1)

    elif metric == "minkowski":
        p = 3
        return np.sum(np.abs(X_train - x_test) ** p, axis=1) ** (1.0 / p)

    elif metric == "chebyshev":
        return np.max(np.abs(X_train - x_test), axis=1)

    elif metric == "cosine":
        norms = np.linalg.norm(X_train, axis=1) * np.linalg.norm(x_test)
        norms[norms == 0] = 1e-10
        dots = X_train @ x_test
        return 1.0 - dots / norms

    else:
        # Fallback: element-wise Python loop for other metrics
        fn = get_distance_function(metric)
        return np.array([fn(X_train[i], x_test) for i in range(len(X_train))])


# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    print("Distance Metrics Test")
    print("─" * 35)
    for name, fn in DISTANCE_FUNCTIONS.items():
        if name == "mahalanobis":
            result = fn(a, b, cov_inv=None)
        else:
            result = fn(a, b)
        print(f"  {name:<15} → {result:.6f}")

    print("\nVectorised pairwise (euclidean):")
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    x = np.array([2.0, 3.0])
    print("  Distances:", pairwise_distances(X, x, "euclidean"))