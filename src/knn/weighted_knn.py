"""
src/knn/weighted_knn.py
------------------------
Weighted k-NN Classifier from scratch.
Neighbours closer to the test point get higher voting weight
via inverse-distance or Gaussian kernel weighting.
No Scikit-learn used.
"""

import numpy as np
from collections import defaultdict
from .distance_metrics import pairwise_distances


class WeightedKNNClassifier:
    """
    Weighted k-Nearest Neighbors Classifier.

    Two weighting schemes:
      - 'inverse'  : weight = 1 / (distance + ε)
      - 'gaussian' : weight = exp( -distance² / (2σ²) )

    Parameters
    ----------
    k        : int    — number of neighbours            (default: 5)
    metric   : str    — distance metric                 (default: 'euclidean')
    weight   : str    — 'inverse' or 'gaussian'         (default: 'inverse')
    sigma    : float  — bandwidth for Gaussian kernel   (default: 1.0)

    Examples
    --------
    >>> clf = WeightedKNNClassifier(k=7, weight='gaussian', sigma=0.5)
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)
    """

    WEIGHT_SCHEMES = ("inverse", "gaussian")

    def __init__(
        self,
        k:      int   = 5,
        metric: str   = "euclidean",
        weight: str   = "inverse",
        sigma:  float = 1.0,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        if weight not in self.WEIGHT_SCHEMES:
            raise ValueError(f"weight must be one of {self.WEIGHT_SCHEMES}")

        self.k      = k
        self.metric = metric
        self.weight = weight
        self.sigma  = sigma

        self.X_train  = None
        self.y_train  = None
        self.classes_ = None
        self._fitted  = False

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedKNNClassifier":
        """Store training data (lazy learner)."""
        self.X_train  = np.array(X, dtype=float)
        self.y_train  = np.array(y)
        self.classes_ = np.unique(self.y_train)
        self._fitted  = True
        print(f"[WeightedKNN] Fitted — {len(X)} samples | k={self.k} | "
              f"metric={self.metric} | weight={self.weight}")
        return self

    # ──────────────────────────────────────────
    # Weight computation
    # ──────────────────────────────────────────

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Convert k nearest distances to voting weights.

        Parameters
        ----------
        distances : np.ndarray, shape (k,) — distances of k neighbours

        Returns
        -------
        weights : np.ndarray, shape (k,) — non-negative voting weights
        """
        if self.weight == "inverse":
            # Inverse distance: closer = higher weight
            return 1.0 / (distances + 1e-9)

        elif self.weight == "gaussian":
            # Gaussian kernel: smooth decay with bandwidth sigma
            return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        else:
            raise ValueError(f"Unknown weight scheme: {self.weight}")

    # ──────────────────────────────────────────
    # Single prediction
    # ──────────────────────────────────────────

    def _predict_one(self, x: np.ndarray):
        """Predict class for a single test vector using weighted voting."""
        distances = pairwise_distances(self.X_train, x, self.metric)

        # Get k nearest indices and their distances
        k_idx    = np.argsort(distances)[:self.k]
        k_dists  = distances[k_idx]
        k_labels = self.y_train[k_idx]

        # Compute weights
        weights = self._compute_weights(k_dists)

        # Aggregate weights per class
        class_weights = defaultdict(float)
        for label, w in zip(k_labels, weights):
            class_weights[label] += w

        return max(class_weights, key=class_weights.get)

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
        self._check_fitted()
        X = np.array(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict weighted class probabilities (normalised per sample).

        Returns
        -------
        proba : np.ndarray, shape (n_test, n_classes)
        """
        self._check_fitted()
        X = np.array(X, dtype=float)
        n_classes = len(self.classes_)
        cls_idx   = {c: i for i, c in enumerate(self.classes_)}
        proba     = np.zeros((len(X), n_classes))

        for i, x in enumerate(X):
            distances = pairwise_distances(self.X_train, x, self.metric)
            k_idx     = np.argsort(distances)[:self.k]
            k_dists   = distances[k_idx]
            k_labels  = self.y_train[k_idx]
            weights   = self._compute_weights(k_dists)

            for lbl, w in zip(k_labels, weights):
                proba[i][cls_idx[lbl]] += w

            total = proba[i].sum()
            if total > 0:
                proba[i] /= total

        return proba

    # ──────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.array(y)))

    # ──────────────────────────────────────────
    # Comparison utility
    # ──────────────────────────────────────────

    @staticmethod
    def compare_weights(X_train, y_train, X_test, y_test,
                        k: int = 5, metric: str = "euclidean"):
        """
        Compare both weighting schemes on the same data.
        Prints accuracy for inverse, gaussian, and unweighted majority vote.
        """
        from .knn import KNNClassifier

        results = {}

        # Unweighted baseline
        plain = KNNClassifier(k=k, metric=metric, weighted=False)
        plain.fit(X_train, y_train)
        results["majority_vote"] = plain.score(X_test, y_test)

        for scheme in ("inverse", "gaussian"):
            clf = WeightedKNNClassifier(k=k, metric=metric, weight=scheme)
            clf.fit(X_train, y_train)
            results[f"weighted_{scheme}"] = clf.score(X_test, y_test)

        print(f"\n{'Scheme':<20} {'Accuracy':>10}")
        print("─" * 32)
        for name, acc in results.items():
            print(f"{name:<20} {acc*100:>9.2f}%")

        return results

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    def get_params(self) -> dict:
        return {
            "k": self.k, "metric": self.metric,
            "weight": self.weight, "sigma": self.sigma,
        }

    def __repr__(self):
        return (f"WeightedKNNClassifier(k={self.k}, metric='{self.metric}', "
                f"weight='{self.weight}', sigma={self.sigma})")


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    np.random.seed(42)
    c0 = np.random.randn(80, 2) + [2, 2]
    c1 = np.random.randn(80, 2) + [-2, -2]
    X  = np.vstack([c0, c1])
    y  = np.array([0]*80 + [1]*80)

    perm   = np.random.permutation(len(X))
    X, y   = X[perm], y[perm]
    split  = 130
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    print("=== Weighted k-NN Demo ===\n")

    clf_inv = WeightedKNNClassifier(k=7, metric="euclidean", weight="inverse")
    clf_inv.fit(X_tr, y_tr)
    print(f"Inverse-distance accuracy : {clf_inv.score(X_te, y_te)*100:.2f}%")

    clf_gauss = WeightedKNNClassifier(k=7, metric="euclidean", weight="gaussian", sigma=0.8)
    clf_gauss.fit(X_tr, y_tr)
    print(f"Gaussian-kernel accuracy  : {clf_gauss.score(X_te, y_te)*100:.2f}%")

    print("\n--- Weight Scheme Comparison ---")
    WeightedKNNClassifier.compare_weights(X_tr, y_tr, X_te, y_te, k=7)