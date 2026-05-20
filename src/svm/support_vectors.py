"""
src/svm/support_vectors.py
---------------------------
Utilities to identify, analyse, and visualise support vectors.
No Scikit-learn used.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def find_support_vectors(X: np.ndarray, y: np.ndarray,
                          alphas: np.ndarray, threshold: float = 1e-5):
    """
    Identify support vectors from dual coefficients (alphas).

    A training point is a support vector if its alpha > threshold,
    meaning it lies on or inside the margin.

    Parameters
    ----------
    X         : np.ndarray, shape (n_samples, n_features)
    y         : np.ndarray, shape (n_samples,) — labels {-1, +1}
    alphas    : np.ndarray, shape (n_samples,) — dual coefficients
    threshold : float — minimum alpha to be considered a support vector

    Returns
    -------
    sv_X      : np.ndarray — support vector feature arrays
    sv_y      : np.ndarray — support vector labels
    sv_idx    : np.ndarray — indices in original X
    sv_alphas : np.ndarray — their alpha values
    """
    sv_idx    = np.where(alphas > threshold)[0]
    sv_X      = X[sv_idx]
    sv_y      = y[sv_idx]
    sv_alphas = alphas[sv_idx]

    print(f"[SVs] Found {len(sv_idx)} support vectors "
          f"({np.sum(sv_y == 1)} positive, {np.sum(sv_y == -1)} negative)")
    return sv_X, sv_y, sv_idx, sv_alphas


def compute_margin(w: np.ndarray) -> float:
    """
    Compute the geometric margin of a linear SVM.
    Formula: margin = 2 / ||w||

    Parameters
    ----------
    w : np.ndarray — weight vector (from SGD/primal SVM)

    Returns
    -------
    margin : float
    """
    norm = np.linalg.norm(w)
    if norm == 0:
        return float("inf")
    return 2.0 / norm


def compute_functional_margins(X: np.ndarray, y: np.ndarray,
                                w: np.ndarray, b: float) -> np.ndarray:
    """
    Compute functional margin  yᵢ(w·xᵢ + b)  for each training point.
    Values < 1  → margin violation.
    Values = 1  → on the margin (support vectors).
    Values > 1  → correctly classified with margin.
    """
    return y * (X @ w + b)


def sv_summary(X: np.ndarray, y: np.ndarray,
               w: np.ndarray, b: float) -> dict:
    """
    Classify all training points into:
      - support vectors (functional margin ≈ 1)
      - margin violators (functional margin < 1)
      - correctly classified (functional margin > 1)

    Returns a dict with counts and indices.
    """
    margins = compute_functional_margins(X, y, w, b)
    tol = 0.1

    sv_idx          = np.where(np.abs(margins - 1.0) <= tol)[0]
    violators_idx   = np.where(margins < 1.0)[0]
    correct_idx     = np.where(margins > 1.0 + tol)[0]

    summary = {
        "support_vectors":    {"count": len(sv_idx),        "indices": sv_idx},
        "margin_violators":   {"count": len(violators_idx), "indices": violators_idx},
        "correctly_classified": {"count": len(correct_idx), "indices": correct_idx},
        "geometric_margin":   compute_margin(w),
    }

    print("\n[SV Summary]")
    print(f"  Support Vectors      : {summary['support_vectors']['count']}")
    print(f"  Margin Violators     : {summary['margin_violators']['count']}")
    print(f"  Correctly Classified : {summary['correctly_classified']['count']}")
    print(f"  Geometric Margin     : {summary['geometric_margin']:.6f}")
    return summary


def plot_support_vectors(X: np.ndarray, y: np.ndarray,
                          w: np.ndarray, b: float,
                          sv_indices: np.ndarray = None,
                          title: str = "SVM — Support Vectors & Margin",
                          save_path: str = None,
                          show: bool = True):
    """
    2D visualisation of SVM decision boundary, margins, and support vectors.
    Works only when X has exactly 2 features.

    Parameters
    ----------
    X          : np.ndarray, shape (n_samples, 2)
    y          : np.ndarray, shape (n_samples,)  labels {-1, +1}
    w          : np.ndarray, shape (2,)          weight vector
    b          : float                           bias
    sv_indices : np.ndarray | None               indices of support vectors
    save_path  : str | None
    show       : bool
    """
    if X.shape[1] != 2:
        raise ValueError("plot_support_vectors requires exactly 2 features.")

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Data points ──
    for label, color, marker in [(1, "steelblue", "o"), (-1, "tomato", "s")]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=color, marker=marker, edgecolors="k",
                   s=60, zorder=3, label=f"Class {label}")

    # ── Highlight support vectors ──
    if sv_indices is not None and len(sv_indices) > 0:
        ax.scatter(X[sv_indices, 0], X[sv_indices, 1],
                   s=200, facecolors="none", edgecolors="gold",
                   linewidths=2.5, zorder=4, label="Support Vectors")

    # ── Decision boundary and margins ──
    x_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)

    if abs(w[1]) > 1e-9:
        # Hyperplane:  w[0]*x + w[1]*y + b = 0  →  y = (-w[0]*x - b) / w[1]
        decision  = (-w[0] * x_vals - b)             / w[1]
        margin_p  = (-w[0] * x_vals - b + 1)         / w[1]
        margin_n  = (-w[0] * x_vals - b - 1)         / w[1]

        ax.plot(x_vals, decision, "k-",  linewidth=2,   label="Decision Boundary")
        ax.plot(x_vals, margin_p, "g--", linewidth=1.5, label="+1 Margin")
        ax.plot(x_vals, margin_n, "r--", linewidth=1.5, label="−1 Margin")
        ax.fill_between(x_vals, margin_n, margin_p, alpha=0.08, color="grey")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SVs] Plot saved → {save_path}")
    if show:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.svm.optimizer import SGDOptimizer

    np.random.seed(3)
    pos = np.random.randn(40, 2) + [2, 2]
    neg = np.random.randn(40, 2) + [-2, -2]
    X   = np.vstack([pos, neg])
    y   = np.array([1]*40 + [-1]*40, dtype=float)

    opt = SGDOptimizer(C=1.0, lr=0.01, n_iters=500, decay=True)
    opt.fit(X, y)

    summary = sv_summary(X, y, opt.w, opt.b)
    sv_idx  = summary["support_vectors"]["indices"]

    plot_support_vectors(X, y, opt.w, opt.b, sv_indices=sv_idx,
                         title="Linear SVM — Support Vectors")