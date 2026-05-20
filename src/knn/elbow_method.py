"""
src/knn/elbow_method.py
------------------------
Elbow Method to find the optimal k for k-NN classification.
Evaluates error rate for each k on a validation split and plots
the curve — the "elbow" indicates the best trade-off point.
No Scikit-learn used.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from .distance_metrics import pairwise_distances


# ──────────────────────────────────────────────
# Core evaluation loop
# ──────────────────────────────────────────────

def _predict_one(X_train, y_train, x_test, k, metric):
    """Predict class label for a single point given k and metric."""
    dists  = pairwise_distances(X_train, x_test, metric)
    k_idx  = np.argsort(dists)[:k]
    labels = y_train[k_idx]
    return Counter(labels).most_common(1)[0][0]


def compute_error_rates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    k_range=range(1, 31),
    metric:  str = "euclidean",
    verbose: bool = True,
):
    """
    Evaluate error rate on validation data for each k in k_range.

    Parameters
    ----------
    X_train, y_train : training data
    X_val,   y_val   : validation data
    k_range          : iterable of k values to test (default 1–30)
    metric           : distance metric name
    verbose          : print per-k results

    Returns
    -------
    k_values    : list[int]
    error_rates : list[float]
    accuracy_rates : list[float]
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train)
    X_val   = np.array(X_val,   dtype=float)
    y_val   = np.array(y_val)

    k_values       = list(k_range)
    error_rates    = []
    accuracy_rates = []

    if verbose:
        print(f"\n{'k':>4}  {'Error Rate':>12}  {'Accuracy':>12}")
        print("─" * 33)

    for k in k_values:
        errors = sum(
            1 for i in range(len(X_val))
            if _predict_one(X_train, y_train, X_val[i], k, metric) != y_val[i]
        )
        err = errors / len(y_val)
        acc = 1.0 - err
        error_rates.append(err)
        accuracy_rates.append(acc)

        if verbose:
            print(f"{k:>4}  {err:>12.4f}  {acc:>12.4f}")

    return k_values, error_rates, accuracy_rates


# ──────────────────────────────────────────────
# Optimal k selection
# ──────────────────────────────────────────────

def find_optimal_k(k_values: list, error_rates: list):
    """
    Returns the k with the lowest validation error rate.
    In case of ties, returns the smallest k (simpler model).

    Returns
    -------
    optimal_k  : int
    min_error  : float
    """
    min_error = min(error_rates)
    optimal_k = k_values[error_rates.index(min_error)]
    return optimal_k, min_error


def find_elbow_k(k_values: list, error_rates: list):
    """
    Finds the 'elbow point' using maximum curvature (second derivative).
    Useful when the minimum-error k is too large.

    Returns
    -------
    elbow_k : int
    """
    errors = np.array(error_rates, dtype=float)
    # Normalise to [0,1] for stable curvature computation
    e_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
    k_norm = np.linspace(0, 1, len(k_values))
    # Second derivative (finite differences)
    d2 = np.gradient(np.gradient(e_norm, k_norm), k_norm)
    elbow_idx = int(np.argmax(np.abs(d2)))
    return k_values[elbow_idx]


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_elbow_curve(
    k_values:    list,
    error_rates: list,
    accuracy_rates: list = None,
    optimal_k:   int  = None,
    elbow_k:     int  = None,
    metric:      str  = "euclidean",
    save_path:   str  = None,
    show:        bool = True,
):
    """
    Plots Error Rate (and optionally Accuracy) vs. k.

    Parameters
    ----------
    k_values, error_rates : results from compute_error_rates()
    accuracy_rates        : optional; also plots accuracy curve
    optimal_k             : if set, marks best k with a red dashed line
    elbow_k               : if set, marks elbow k with an orange dashed line
    metric                : used in plot title
    save_path             : optional path to save PNG
    show                  : whether to call plt.show()
    """
    has_acc = accuracy_rates is not None
    fig, axes = plt.subplots(1, 2 if has_acc else 1,
                             figsize=(14 if has_acc else 8, 5))

    if not has_acc:
        axes = [axes]

    # ── Error Rate ──
    ax = axes[0]
    ax.plot(k_values, error_rates, marker='o', linewidth=2,
            color='steelblue', markersize=5, label='Error Rate')

    for k_mark, color, label in [
        (optimal_k, 'crimson',   f'Optimal k={optimal_k}'),
        (elbow_k,   'darkorange', f'Elbow k={elbow_k}'),
    ]:
        if k_mark is not None and k_mark in k_values:
            idx = k_values.index(k_mark)
            ax.axvline(k_mark, color=color, linestyle='--', linewidth=1.8, label=label)
            ax.scatter([k_mark], [error_rates[idx]], color=color, s=100, zorder=5)

    ax.set_title(f'Elbow Method — Error Rate vs k\n[metric: {metric}]', fontsize=12)
    ax.set_xlabel('k (Number of Neighbours)', fontsize=11)
    ax.set_ylabel('Error Rate', fontsize=11)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(k_values[::2])

    # ── Accuracy (optional) ──
    if has_acc:
        ax2 = axes[1]
        ax2.plot(k_values, accuracy_rates, marker='s', linewidth=2,
                 color='seagreen', markersize=5, label='Accuracy')
        if optimal_k is not None and optimal_k in k_values:
            idx = k_values.index(optimal_k)
            ax2.axvline(optimal_k, color='crimson', linestyle='--',
                        linewidth=1.8, label=f'Optimal k={optimal_k}')
            ax2.scatter([optimal_k], [accuracy_rates[idx]],
                        color='crimson', s=100, zorder=5)
        ax2.set_title(f'Accuracy vs k\n[metric: {metric}]', fontsize=12)
        ax2.set_xlabel('k (Number of Neighbours)', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.set_xticks(k_values[::2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Elbow] Plot saved → {save_path}")
    if show:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────

def run_elbow_analysis(
    X_train, y_train, X_val, y_val,
    k_range=range(1, 31),
    metric="euclidean",
    save_path=None,
):
    """
    Full pipeline: compute error rates → find optimal k → plot.

    Returns
    -------
    optimal_k : int
    """
    print(f"\n[Elbow] Running for k={list(k_range)[0]}..{list(k_range)[-1]}, "
          f"metric={metric}")
    k_vals, errs, accs = compute_error_rates(
        X_train, y_train, X_val, y_val, k_range, metric
    )
    opt_k, min_err = find_optimal_k(k_vals, errs)
    elbow_k = find_elbow_k(k_vals, errs)
    print(f"\n✅ Optimal k = {opt_k}  (error={min_err:.4f})")
    print(f"   Elbow   k = {elbow_k}")
    plot_elbow_curve(k_vals, errs, accs,
                     optimal_k=opt_k, elbow_k=elbow_k,
                     metric=metric, save_path=save_path)
    return opt_k


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    np.random.seed(7)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_tr, y_tr = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    opt = run_elbow_analysis(X_tr, y_tr, X_val, y_val,
                              k_range=range(1, 25), metric="euclidean")
    print(f"\nBest k for this dataset: {opt}")