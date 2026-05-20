"""
src/visualization/elbow_plot.py
---------------------------------
Standalone elbow curve plotting utility.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_elbow(
    k_values:      list,
    error_rates:   list,
    accuracy_rates: list = None,
    optimal_k:     int  = None,
    metric:        str  = "euclidean",
    save_path:     str  = None,
    show:          bool = True,
):
    """
    Plot the Elbow Curve: Error Rate (and optionally Accuracy) vs k.

    Parameters
    ----------
    k_values, error_rates : output of compute_error_rates()
    accuracy_rates        : optional parallel accuracy list
    optimal_k             : if set, marks the best k on both plots
    metric                : distance metric name (for title)
    """
    ncols = 2 if accuracy_rates else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Error Rate plot
    ax = axes[0]
    ax.plot(k_values, error_rates, "o-", color="steelblue",
            linewidth=2, markersize=5, label="Error Rate")
    if optimal_k and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        ax.axvline(optimal_k, color="crimson", linestyle="--", lw=1.8,
                   label=f"Best k={optimal_k}")
        ax.scatter([optimal_k], [error_rates[idx]], color="crimson", s=100, zorder=5)
    ax.set_title(f"Elbow Method — Error Rate vs k\n[{metric}]", fontsize=12)
    ax.set_xlabel("k  (Number of Neighbours)", fontsize=11)
    ax.set_ylabel("Error Rate", fontsize=11)
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(k_values[::2])

    # Accuracy plot (optional)
    if accuracy_rates:
        ax2 = axes[1]
        ax2.plot(k_values, accuracy_rates, "s-", color="seagreen",
                 linewidth=2, markersize=5, label="Accuracy")
        if optimal_k and optimal_k in k_values:
            idx = k_values.index(optimal_k)
            ax2.axvline(optimal_k, color="crimson", linestyle="--", lw=1.8,
                        label=f"Best k={optimal_k}")
            ax2.scatter([optimal_k], [accuracy_rates[idx]],
                        color="crimson", s=100, zorder=5)
        ax2.set_title(f"Accuracy vs k\n[{metric}]", fontsize=12)
        ax2.set_xlabel("k  (Number of Neighbours)", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.set_xticks(k_values[::2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Elbow curve saved → {save_path}")
    if show:
        plt.show()
    plt.close()