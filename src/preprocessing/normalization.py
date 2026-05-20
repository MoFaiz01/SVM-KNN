"""
src/preprocessing/normalization.py
------------------------------------
Feature scaling methods implemented from scratch.
Scaling is critical for both k-NN (distance-based) and SVM (margin-based).
No Scikit-learn used.
"""

import numpy as np


# ──────────────────────────────────────────────
# Min-Max Normalization
# ──────────────────────────────────────────────

def min_max_normalize(X: np.ndarray, feature_range=(0, 1),
                       X_min=None, X_max=None):
    """
    Scale features to [a, b] range.
    Formula: X_scaled = a + (X - X_min) / (X_max - X_min) * (b - a)

    Parameters
    ----------
    X             : np.ndarray, shape (n, d)
    feature_range : tuple (a, b)  — target range, default (0, 1)
    X_min, X_max  : if provided, use these stats (for applying to test data)

    Returns
    -------
    X_scaled : np.ndarray
    X_min    : np.ndarray — per-feature minima  (store for test scaling)
    X_max    : np.ndarray — per-feature maxima
    """
    X = np.array(X, dtype=float)
    if X_min is None:
        X_min = X.min(axis=0)
    if X_max is None:
        X_max = X.max(axis=0)

    a, b = feature_range
    denom = X_max - X_min
    denom[denom == 0] = 1.0  # avoid division by zero for constant features

    X_scaled = a + (X - X_min) / denom * (b - a)
    return X_scaled, X_min, X_max


# ──────────────────────────────────────────────
# Z-Score Standardisation
# ──────────────────────────────────────────────

def z_score_normalize(X: np.ndarray, mean=None, std=None):
    """
    Standardise features to zero mean, unit variance.
    Formula: X_scaled = (X - μ) / σ

    Parameters
    ----------
    X         : np.ndarray, shape (n, d)
    mean, std : if provided, apply these stats (for test data)

    Returns
    -------
    X_scaled : np.ndarray
    mean     : np.ndarray — per-feature means
    std      : np.ndarray — per-feature standard deviations
    """
    X = np.array(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)

    std_safe = np.where(std == 0, 1.0, std)  # avoid /0
    X_scaled = (X - mean) / std_safe
    return X_scaled, mean, std


# ──────────────────────────────────────────────
# L2 (Unit) Normalisation (per sample)
# ──────────────────────────────────────────────

def l2_normalize(X: np.ndarray):
    """
    Normalise each sample to unit L2 norm.
    Formula: x_i = x_i / ||x_i||₂

    Useful for cosine-distance-based models.
    """
    X = np.array(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ──────────────────────────────────────────────
# L1 (Manhattan) Normalisation (per sample)
# ──────────────────────────────────────────────

def l1_normalize(X: np.ndarray):
    """
    Normalise each sample to unit L1 norm.
    Formula: x_i = x_i / ||x_i||₁
    """
    X = np.array(X, dtype=float)
    norms = np.abs(X).sum(axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ──────────────────────────────────────────────
# Robust Scaler (using median and IQR)
# ──────────────────────────────────────────────

def robust_normalize(X: np.ndarray, median=None, iqr=None):
    """
    Scale using median and interquartile range — robust to outliers.
    Formula: X_scaled = (X - median) / IQR

    Returns
    -------
    X_scaled : np.ndarray
    median   : np.ndarray
    iqr      : np.ndarray
    """
    X = np.array(X, dtype=float)
    if median is None:
        median = np.median(X, axis=0)
    if iqr is None:
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25

    iqr_safe = np.where(iqr == 0, 1.0, iqr)
    return (X - median) / iqr_safe, median, iqr


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

def normalize(X: np.ndarray, method: str = "minmax", **kwargs):
    """
    Convenience dispatcher.

    Parameters
    ----------
    X      : np.ndarray
    method : 'minmax' | 'zscore' | 'l2' | 'l1' | 'robust'

    Returns
    -------
    X_scaled  : np.ndarray
    stats     : dict with scaling parameters (for reuse on test data)
    """
    method = method.lower().strip()

    if method == "minmax":
        Xs, mn, mx = min_max_normalize(X, **kwargs)
        return Xs, {"min": mn, "max": mx}

    elif method == "zscore":
        Xs, mu, sigma = z_score_normalize(X, **kwargs)
        return Xs, {"mean": mu, "std": sigma}

    elif method == "l2":
        return l2_normalize(X), {}

    elif method == "l1":
        return l1_normalize(X), {}

    elif method == "robust":
        Xs, med, iqr_ = robust_normalize(X, **kwargs)
        return Xs, {"median": med, "iqr": iqr_}

    else:
        raise ValueError(f"Unknown normalisation method '{method}'. "
                         f"Choose: minmax, zscore, l2, l1, robust")


def apply_normalization(X: np.ndarray, method: str, stats: dict) -> np.ndarray:
    """
    Apply precomputed scaling stats to new data (e.g. test set).

    Parameters
    ----------
    X      : np.ndarray — test data
    method : str        — same method used on training data
    stats  : dict       — stats dict returned by normalize()
    """
    method = method.lower().strip()

    if method == "minmax":
        Xs, _, _ = min_max_normalize(X, X_min=stats["min"], X_max=stats["max"])
        return Xs
    elif method == "zscore":
        Xs, _, _ = z_score_normalize(X, mean=stats["mean"], std=stats["std"])
        return Xs
    elif method in ("l2", "l1"):
        return l2_normalize(X) if method == "l2" else l1_normalize(X)
    elif method == "robust":
        Xs, _, _ = robust_normalize(X, median=stats["median"], iqr=stats["iqr"])
        return Xs
    else:
        raise ValueError(f"Unknown method '{method}'")


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    X = np.array([[1, 200, 0.1],
                  [2, 400, 0.5],
                  [3, 600, 0.9],
                  [4, 800, 1.3]], dtype=float)

    print("Original X:\n", X)

    Xs_mm, stats_mm = normalize(X, "minmax")
    print("\nMin-Max Normalised:\n", Xs_mm.round(4))

    Xs_z, stats_z = normalize(X, "zscore")
    print("\nZ-Score Standardised:\n", Xs_z.round(4))

    Xs_l2 = l2_normalize(X)
    print("\nL2 Normalised:\n", Xs_l2.round(4))

    Xs_r, stats_r = normalize(X, "robust")
    print("\nRobust Scaled:\n", Xs_r.round(4))