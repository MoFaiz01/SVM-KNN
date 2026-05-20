"""
src/evaluation/metrics.py
--------------------------
Classification metrics implemented manually from scratch.
Accuracy, Precision, Recall, F1-Score, and full report.
No Scikit-learn used.
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Classification accuracy.
    Formula: (TP + TN) / (TP + TN + FP + FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))


def precision(y_true: np.ndarray, y_pred: np.ndarray,
              cls, zero_division: float = 0.0) -> float:
    """
    Precision for a single class.
    Formula: TP / (TP + FP)

    Parameters
    ----------
    cls            : the class label to compute precision for
    zero_division  : value to return if denominator is 0
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else zero_division


def recall(y_true: np.ndarray, y_pred: np.ndarray,
           cls, zero_division: float = 0.0) -> float:
    """
    Recall (Sensitivity / True Positive Rate) for a single class.
    Formula: TP / (TP + FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else zero_division


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             cls, zero_division: float = 0.0) -> float:
    """
    F1-Score (harmonic mean of precision and recall) for a single class.
    Formula: 2 * P * R / (P + R)
    """
    p = precision(y_true, y_pred, cls, zero_division)
    r = recall(y_true, y_pred, cls, zero_division)
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else zero_division


def specificity(y_true: np.ndarray, y_pred: np.ndarray, cls) -> float:
    """
    Specificity (True Negative Rate) for a single class.
    Formula: TN / (TN + FP)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = np.sum((y_pred != cls) & (y_true != cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                           target_names: list = None,
                           average: str = "macro") -> str:
    """
    Full per-class report with overall averages.

    Parameters
    ----------
    y_true       : np.ndarray — ground truth labels
    y_pred       : np.ndarray — predicted labels
    target_names : list — optional class name strings
    average      : 'macro' | 'weighted' — how to aggregate overall metrics

    Returns
    -------
    report : str — formatted table
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)

    if target_names is None:
        target_names = [str(c) for c in classes]

    lines = [
        f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} "
        f"{'F1-Score':>10} {'Support':>10}",
        "─" * 58,
    ]

    supports    = []
    precisions  = []
    recalls     = []
    f1s         = []

    for cls, name in zip(classes, target_names):
        p = precision(y_true, y_pred, cls)
        r = recall(y_true, y_pred, cls)
        f = f1_score(y_true, y_pred, cls)
        s = int(np.sum(y_true == cls))
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(s)
        lines.append(f"{name:<15} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10}")

    lines.append("─" * 58)

    total = sum(supports)
    if average == "weighted":
        weights   = np.array(supports) / total
        avg_p = float(np.dot(weights, precisions))
        avg_r = float(np.dot(weights, recalls))
        avg_f = float(np.dot(weights, f1s))
    else:  # macro
        avg_p = float(np.mean(precisions))
        avg_r = float(np.mean(recalls))
        avg_f = float(np.mean(f1s))

    acc = accuracy(y_true, y_pred)
    lines.append(f"{'Accuracy':<15} {'':>10} {'':>10} {acc:>10.4f} {total:>10}")
    lines.append(f"{average+' avg':<15} {avg_p:>10.4f} {avg_r:>10.4f} {avg_f:>10.4f} {total:>10}")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report


def error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - accuracy."""
    return 1.0 - accuracy(y_true, y_pred)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 1, 0, 1, 2])

    print(f"Accuracy : {accuracy(y_true, y_pred):.4f}")
    for c in [0, 1, 2]:
        print(f"Class {c} — P={precision(y_true,y_pred,c):.3f} "
              f"R={recall(y_true,y_pred,c):.3f} "
              f"F1={f1_score(y_true,y_pred,c):.3f}")

    classification_report(y_true, y_pred,
                          target_names=["setosa", "versicolor", "virginica"])