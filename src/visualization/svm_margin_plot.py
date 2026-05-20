"""
src/visualization/svm_margin_plot.py
--------------------------------------
SVM margin and support vector visualisation for linear SVMs.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_svm_margin(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    sv_indices:  np.ndarray = None,
    title:       str  = "SVM — Margin & Support Vectors",
    class_names: list = None,
    save_path:   str  = None,
    show:        bool = True,
):
    """
    Plot SVM hyperplane, ±1 margin lines, and support vectors.
    Requires exactly 2 features.

    Parameters
    ----------
    X          : np.ndarray, shape (n, 2)
    y          : np.ndarray, labels {-1, +1}
    w          : np.ndarray, shape (2,)  — weight vector
    b          : float                   — bias
    sv_indices : indices of support vectors (highlighted)
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Data points
    for label, color, marker, name in [
        ( 1, "steelblue", "o", class_names[0] if class_names else "+1"),
        (-1, "tomato",    "s", class_names[1] if class_names else "−1"),
    ]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                   edgecolors="k", s=70, zorder=3, label=name)

    # Highlight support vectors
    if sv_indices is not None and len(sv_indices) > 0:
        ax.scatter(X[sv_indices, 0], X[sv_indices, 1],
                   s=220, facecolors="none", edgecolors="gold",
                   linewidths=2.5, zorder=5, label="Support Vectors")

    # Decision boundary and margins
    x_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)

    if abs(w[1]) > 1e-10:
        def hyperplane(x, offset=0):
            return (-w[0] * x - b + offset) / w[1]

        ax.plot(x_vals, hyperplane(x_vals, 0), "k-",  lw=2.5, label="Decision Boundary")
        ax.plot(x_vals, hyperplane(x_vals, +1), "g--", lw=1.8, label="+1 Margin")
        ax.plot(x_vals, hyperplane(x_vals, -1), "r--", lw=1.8, label="−1 Margin")
        ax.fill_between(x_vals,
                         hyperplane(x_vals, -1),
                         hyperplane(x_vals, +1),
                         alpha=0.08, color="grey", label="Margin zone")

    # Margin width annotation
    margin_width = 2.0 / (np.linalg.norm(w) + 1e-12)
    ax.set_title(f"{title}\nGeometric Margin = {margin_width:.4f}", fontsize=12)
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    if show:
        plt.show()
    plt.close()