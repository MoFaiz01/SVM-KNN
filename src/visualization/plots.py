"""
src/visualization/plots.py
---------------------------
General-purpose plotting utilities for classification experiments.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_comparison(
    model_names: list,
    accuracies:  list,
    title:       str  = "Model Accuracy Comparison",
    ylabel:      str  = "Accuracy (%)",
    colors:      list = None,
    save_path:   str  = None,
    show:        bool = True,
):
    """
    Bar chart comparing accuracy across models or configurations.

    Parameters
    ----------
    model_names : list of str — x-axis labels
    accuracies  : list of float — values in [0, 1]
    """
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.4), 5))
    bars = ax.bar(model_names, [a * 100 for a in accuracies],
                  color=colors, edgecolor="white", linewidth=1.2)

    # Value labels on bars
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Model / Configuration", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_loss_curve(
    loss_history: list,
    title:        str  = "Training Loss Curve",
    save_path:    str  = None,
    show:         bool = True,
):
    """Plot training loss over epochs (for SGD-based SVM)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(loss_history, color="steelblue", linewidth=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_accuracy_bar(
    k_values:   list,
    accuracies: list,
    metric:     str = "euclidean",
    save_path:  str = None,
    show:       bool = True,
):
    """Bar chart of k-NN accuracy for different k values."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(k_values, [a * 100 for a in accuracies],
           color="steelblue", edgecolor="white")
    ax.set_title(f"k-NN Accuracy vs k  [{metric}]", fontsize=12)
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(k_values)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_cv_results(cv_results: dict, save_path: str = None, show: bool = True):
    """
    Box-plot style visualisation of cross-validation fold accuracies
    for multiple models.

    Parameters
    ----------
    cv_results : dict {model_name: {'fold_accuracies': [...], ...}}
    """
    names  = list(cv_results.keys())
    data   = [np.array(cv_results[n]["fold_accuracies"]) * 100 for n in names]
    means  = [d.mean() for d in data]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False)

    colours = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)

    for i, mean in enumerate(means):
        ax.text(i + 1, mean + 0.3, f"{mean:.1f}%",
                ha="center", fontsize=9, color="navy")

    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Cross-Validation Accuracy Comparison", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()