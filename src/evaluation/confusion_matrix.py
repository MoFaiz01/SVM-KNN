"""
src/evaluation/confusion_matrix.py
------------------------------------
Confusion matrix — computation and visualisation.
No Scikit-learn used.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute confusion matrix manually.

    Rows    = Actual class
    Columns = Predicted class

    Returns
    -------
    matrix  : np.ndarray, shape (n_classes, n_classes)
    classes : np.ndarray — unique class labels in sorted order
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    classes = np.unique(y_true)
    n       = len(classes)
    idx_map = {c: i for i, c in enumerate(classes)}

    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[idx_map[t]][idx_map[p]] += 1

    return matrix, classes


def print_confusion_matrix(matrix: np.ndarray, classes: np.ndarray,
                            title: str = "Confusion Matrix"):
    """Pretty-print the confusion matrix as a table."""
    n = len(classes)
    col_w = max(10, max(len(str(c)) for c in classes) + 2)

    print(f"\n{title}")
    print("─" * (col_w * (n + 1) + 2))

    # Header
    header = f"{'Actual \\ Pred':<{col_w}}" + "".join(f"{str(c):>{col_w}}" for c in classes)
    print(header)
    print("─" * (col_w * (n + 1) + 2))

    for i, cls in enumerate(classes):
        row = f"{str(cls):<{col_w}}" + "".join(f"{matrix[i][j]:>{col_w}}" for j in range(n))
        print(row)

    print("─" * (col_w * (n + 1) + 2))

    # Per-class accuracy
    for i, cls in enumerate(classes):
        total = matrix[i].sum()
        correct = matrix[i][i]
        print(f"  Class {cls}: {correct}/{total} correct "
              f"({100*correct/total:.1f}%)" if total > 0 else f"  Class {cls}: no samples")


def plot_confusion_matrix(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    class_names:  list = None,
    title:        str  = "Confusion Matrix",
    cmap:         str  = "Blues",
    normalise:    bool = False,
    save_path:    str  = None,
    show:         bool = True,
):
    """
    Plot confusion matrix as a colour-coded heatmap.

    Parameters
    ----------
    y_true, y_pred : arrays of labels
    class_names    : display names for classes
    title          : plot title
    cmap           : matplotlib colormap
    normalise      : if True, show row-normalised percentages
    save_path      : optional PNG path to save figure
    show           : whether to call plt.show()
    """
    matrix, classes = confusion_matrix(y_true, y_pred)

    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display_matrix = matrix.astype(float) / row_sums
        fmt = ".2f"
    else:
        display_matrix = matrix
        fmt = "d"

    if class_names is None:
        class_names = [str(c) for c in classes]

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), max(4, n * 1.3)))

    im = ax.imshow(display_matrix, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Threshold for text colour
    thresh = display_matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = display_matrix[i, j]
            txt = f"{val:{fmt}}"
            colour = "white" if val > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, color=colour, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[CM] Saved → {save_path}")
    if show:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 1, 2, 0, 2])

    mat, cls = confusion_matrix(y_true, y_pred)
    print_confusion_matrix(mat, cls)

    plot_confusion_matrix(y_true, y_pred,
                          class_names=["Setosa", "Versicolor", "Virginica"],
                          title="Demo Confusion Matrix")