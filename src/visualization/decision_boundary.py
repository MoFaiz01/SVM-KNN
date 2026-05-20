"""
src/visualization/decision_boundary.py
----------------------------------------
Decision boundary visualisations for k-NN and SVM classifiers.
Works on 2D feature spaces (first 2 features used if d > 2).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


COLORS_10 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
             "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
             "#CCB974", "#64B5CD"]


def plot_decision_boundary_knn(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list = None,
    class_names:   list = None,
    title:         str  = "k-NN Decision Boundary",
    resolution:    float = 0.05,
    save_path:     str  = None,
    show:          bool = True,
):
    """
    Plot k-NN decision boundary over 2D data.

    Parameters
    ----------
    model        : fitted KNNClassifier with predict() method
    X            : np.ndarray, shape (n, 2)  — 2 features only
    y            : np.ndarray, shape (n,)
    resolution   : grid step size (smaller = finer but slower)
    """
    X = np.array(X[:, :2], dtype=float)
    classes = np.unique(y)
    n_cls   = len(classes)

    color_map  = {cls: COLORS_10[i % 10] for i, cls in enumerate(classes)}
    cmap       = mcolors.ListedColormap([color_map[c] for c in classes])
    cls_to_int = {c: i for i, c in enumerate(classes)}

    # Build mesh grid
    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                          np.arange(y_min, y_max, resolution))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_raw = model.predict(grid_points)
    Z     = np.array([cls_to_int[z] for z in Z_raw]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap, levels=n_cls - 1)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, alpha=0.3)

    # Plot data points
    y_int = np.array([cls_to_int[yi] for yi in y])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_int,
                         cmap=cmap, edgecolors="k", s=60, zorder=3)

    # Legend
    if class_names is None:
        class_names = [str(c) for c in classes]
    legend_elements = [
        Patch(facecolor=color_map[cls], edgecolor="k", label=name)
        for cls, name in zip(classes, class_names)
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9)

    feat = feature_names if feature_names else ["Feature 1", "Feature 2"]
    ax.set_xlabel(feat[0], fontsize=11)
    ax.set_ylabel(feat[1], fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_decision_boundary_svm(
    model,
    X: np.ndarray,
    y: np.ndarray,
    class_names:  list  = None,
    title:        str   = "SVM Decision Boundary",
    resolution:   float = 0.05,
    show_margin:  bool  = True,
    save_path:    str   = None,
    show:         bool  = True,
):
    """
    Plot SVM decision boundary and margin for 2D data.
    Draws filled regions with contour lines for boundaries.
    """
    X = np.array(X[:, :2], dtype=float)
    classes  = np.unique(y)
    n_cls    = len(classes)
    cls_int  = {c: i for i, c in enumerate(classes)}

    x_min = X[:, 0].min() - 0.5;  x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5;  y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                          np.arange(y_min, y_max, resolution))

    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Background colour
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
                alpha=0.2, colors=["#4C72B0", "#DD8452"])

    # Decision boundary (Z=0) and margins (Z=±1)
    cs = ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                    linestyles=["--", "-", "--"],
                    colors=["royalblue", "black", "tomato"],
                    linewidths=[1.5, 2.5, 1.5])
    ax.clabel(cs, fmt={-1: "−1", 0: "DB", 1: "+1"}, fontsize=9)

    # Data points
    for cls, color in zip(classes, ["#4C72B0", "#DD8452"]):
        mask = y == cls
        lbl  = class_names[cls_int[cls]] if class_names else str(cls)
        ax.scatter(X[mask, 0], X[mask, 1], c=color,
                   edgecolors="k", s=70, zorder=3, label=lbl)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()