"""
src/knn/knn.py
--------------
k-Nearest Neighbors Classifier built entirely from scratch.
Supports multiple distance metrics, majority & weighted voting,
full evaluation suite, and decision-boundary visualisation.
No Scikit-learn used.
"""

import numpy as np
from collections import Counter
from .distance_metrics import pairwise_distances, get_distance_function


class KNNClassifier:
    """
    k-Nearest Neighbors Classifier from scratch.

    Parameters
    ----------
    k        : int   — neighbours to consider             (default: 3)
    metric   : str   — 'euclidean' | 'manhattan' |
                       'minkowski' | 'chebyshev' |
                       'hamming'   | 'cosine'             (default: 'euclidean')
    weighted : bool  — inverse-distance weighted voting    (default: False)

    Examples
    --------
    >>> clf = KNNClassifier(k=5, metric='euclidean')
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)
    >>> print(clf.score(X_test, y_test))
    """

    def __init__(self, k: int = 3, metric: str = "euclidean", weighted: bool = False):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.classes_: np.ndarray = None
        self._fitted = False

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Store training data (lazy learner — no computation at fit time).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        self.classes_ = np.unique(self.y_train)
        self._fitted = True
        print(f"[KNN] Fitted — {len(X)} samples | "
              f"k={self.k} | metric={self.metric} | weighted={self.weighted} | "
              f"classes={list(self.classes_)}")
        return self

    # ──────────────────────────────────────────
    # Single-point prediction
    # ──────────────────────────────────────────

    def _predict_one(self, x: np.ndarray):
        """Predict class for a single test vector."""
        self._check_fitted()

        # Vectorised distances to all training points
        distances = pairwise_distances(self.X_train, x, self.metric)

        # Indices of k nearest neighbours (ascending distance)
        k_idx = np.argsort(distances)[:self.k]
        k_dists  = distances[k_idx]
        k_labels = self.y_train[k_idx]

        if self.weighted:
            # Inverse-distance weighted voting
            weights = {}
            for dist, label in zip(k_dists, k_labels):
                w = 1.0 / (dist + 1e-9)
                weights[label] = weights.get(label, 0.0) + w
            return max(weights, key=weights.get)
        else:
            # Simple majority vote
            return Counter(k_labels).most_common(1)[0][0]

    # ──────────────────────────────────────────
    # Batch prediction
    # ──────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for multiple samples.

        Parameters
        ----------
        X : array-like, shape (n_test, n_features)

        Returns
        -------
        predictions : np.ndarray, shape (n_test,)
        """
        X = np.array(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities as vote fractions.

        Returns
        -------
        proba : np.ndarray, shape (n_test, n_classes)
        """
        self._check_fitted()
        X = np.array(X, dtype=float)
        n_classes = len(self.classes_)
        class_idx = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((len(X), n_classes))

        for i, x in enumerate(X):
            distances = pairwise_distances(self.X_train, x, self.metric)
            k_idx = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_idx]
            for lbl in k_labels:
                proba[i][class_idx[lbl]] += 1
            proba[i] /= self.k

        return proba

    # ──────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns classification accuracy on (X, y)."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.array(y)))

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray):
        """
        Computes confusion matrix manually.

        Returns
        -------
        matrix  : np.ndarray, shape (n_classes, n_classes)
                  Rows = actual, Columns = predicted
        classes : np.ndarray — class labels in order
        """
        y_true = np.array(y)
        y_pred = self.predict(X)
        classes = np.unique(y_true)
        idx_map = {c: i for i, c in enumerate(classes)}
        n = len(classes)
        matrix = np.zeros((n, n), dtype=int)
        for true, pred in zip(y_true, y_pred):
            matrix[idx_map[true]][idx_map[pred]] += 1
        return matrix, classes

    def classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Returns a formatted string with per-class Precision, Recall,
        F1-Score and Support — all computed manually.
        """
        y_true = np.array(y)
        y_pred = self.predict(X)
        classes = np.unique(y_true)

        lines = []
        lines.append(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} "
                     f"{'F1-Score':>10} {'Support':>10}")
        lines.append("─" * 55)

        for cls in classes:
            tp = int(np.sum((y_pred == cls) & (y_true == cls)))
            fp = int(np.sum((y_pred == cls) & (y_true != cls)))
            fn = int(np.sum((y_pred != cls) & (y_true == cls)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            support   = int(np.sum(y_true == cls))

            lines.append(f"{str(cls):<12} {precision:>10.4f} {recall:>10.4f} "
                         f"{f1:>10.4f} {support:>10}")

        acc = float(np.mean(y_pred == y_true))
        lines.append("─" * 55)
        lines.append(f"{'Accuracy':<12} {acc:>10.4f}  "
                     f"(total samples: {len(y_true)})\n")
        report = "\n".join(lines)
        print(report)
        return report

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    def get_params(self) -> dict:
        return {"k": self.k, "metric": self.metric, "weighted": self.weighted}

    def __repr__(self):
        return (f"KNNClassifier(k={self.k}, metric='{self.metric}', "
                f"weighted={self.weighted})")


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    np.random.seed(0)
    c0 = np.random.randn(60, 2) + [2, 2]
    c1 = np.random.randn(60, 2) + [-2, -2]
    c2 = np.random.randn(60, 2) + [2, -2]
    X  = np.vstack([c0, c1, c2])
    y  = np.array([0]*60 + [1]*60 + [2]*60)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    clf = KNNClassifier(k=5, metric="euclidean")
    clf.fit(X_tr, y_tr)
    print(f"\nAccuracy: {clf.score(X_te, y_te)*100:.2f}%")
    clf.classification_report(X_te, y_te)
    cm, cls = clf.confusion_matrix(X_te, y_te)
    print("Confusion Matrix:\n", cm)