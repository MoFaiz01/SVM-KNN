"""
src/svm/svm.py
--------------
Support Vector Machine Classifier built entirely from scratch.
Supports linear, polynomial, and RBF kernels.
Uses SGD (linear) or SMO (kernel) optimisation.
No Scikit-learn used.
"""

import numpy as np
from .kernels import get_kernel, linear_kernel, rbf_kernel, polynomial_kernel
from .optimizer import SGDOptimizer, SMOOptimizer


class SVMClassifier:
    """
    Binary SVM Classifier from scratch.

    Parameters
    ----------
    kernel  : str   — 'linear' | 'polynomial' | 'rbf'   (default: 'linear')
    C       : float — regularisation / margin penalty     (default: 1.0)
    lr      : float — learning rate (SGD only)            (default: 0.001)
    n_iters : int   — epochs (SGD) or max passes (SMO)    (default: 1000)
    gamma   : float — RBF/poly kernel parameter           (default: None → 1/n_features)
    degree  : int   — polynomial degree                   (default: 3)
    coef0   : float — polynomial independent term         (default: 1.0)
    optimizer: str  — 'sgd' | 'smo'                      (default: 'sgd' for linear,
                                                                    'smo' for kernel)

    Examples
    --------
    >>> clf = SVMClassifier(kernel='rbf', C=1.0, gamma=0.1)
    >>> clf.fit(X_train, y_train)     # labels: {-1, +1}
    >>> preds = clf.predict(X_test)
    >>> print(clf.score(X_test, y_test))
    """

    def __init__(
        self,
        kernel:    str   = "linear",
        C:         float = 1.0,
        lr:        float = 0.001,
        n_iters:   int   = 1000,
        gamma:     float = None,
        degree:    int   = 3,
        coef0:     float = 1.0,
        optimizer: str   = None,
    ):
        self.kernel    = kernel
        self.C         = C
        self.lr        = lr
        self.n_iters   = n_iters
        self.gamma     = gamma
        self.degree    = degree
        self.coef0     = coef0

        # Default optimiser choice
        if optimizer is None:
            self.optimizer = "sgd" if kernel == "linear" else "smo"
        else:
            self.optimizer = optimizer

        self._opt       = None     # fitted optimiser object
        self._fitted    = False
        self.classes_   = None
        self._label_map = {}       # original → {-1, +1}
        self._inv_map   = {}       # {-1, +1} → original

    # ──────────────────────────────────────────
    # Label encoding helpers (any 2-class labels → {-1, +1})
    # ──────────────────────────────────────────

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"SVMClassifier supports binary classification only. "
                f"Got {len(self.classes_)} classes."
            )
        self._label_map = {self.classes_[0]: -1, self.classes_[1]: +1}
        self._inv_map   = {-1: self.classes_[0], +1: self.classes_[1]}
        return np.array([self._label_map[yi] for yi in y], dtype=float)

    def _decode_labels(self, y_enc: np.ndarray) -> np.ndarray:
        return np.array([self._inv_map[int(yi)] for yi in y_enc])

    # ──────────────────────────────────────────
    # Kernel function builder
    # ──────────────────────────────────────────

    def _build_kernel_fn(self, n_features: int):
        gamma = self.gamma if self.gamma is not None else 1.0 / n_features

        if self.kernel == "linear":
            return linear_kernel

        elif self.kernel in ("polynomial", "poly"):
            def poly_fn(x1, x2):
                return polynomial_kernel(x1, x2,
                                         degree=self.degree,
                                         coef0=self.coef0,
                                         gamma=gamma)
            return poly_fn

        elif self.kernel == "rbf":
            def rbf_fn(x1, x2):
                return rbf_kernel(x1, x2, gamma=gamma)
            return rbf_fn

        else:
            return get_kernel(self.kernel)

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """
        Train the SVM.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) — any two distinct labels
        """
        X   = np.array(X, dtype=float)
        y   = np.array(y)
        y_e = self._encode_labels(y)

        kernel_fn = self._build_kernel_fn(X.shape[1])

        print(f"\n[SVM] Training — kernel={self.kernel} | C={self.C} | "
              f"optimizer={self.optimizer} | samples={len(X)}")

        if self.optimizer == "sgd":
            self._opt = SGDOptimizer(C=self.C, lr=self.lr, n_iters=self.n_iters)
            self._opt.fit(X, y_e)

        elif self.optimizer == "smo":
            self._opt = SMOOptimizer(
                C=self.C, tol=1e-3,
                max_iter=self.n_iters,
                kernel_fn=kernel_fn,
            )
            self._opt.fit(X, y_e)

        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'. Use 'sgd' or 'smo'.")

        self._fitted = True
        print(f"[SVM] Training complete.\n")
        return self

    # ──────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw signed distance from the hyperplane."""
        self._check_fitted()
        return self._opt.decision_function(np.array(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (returns original label type)."""
        self._check_fitted()
        raw = self._opt.predict(np.array(X, dtype=float))
        return self._decode_labels(raw)

    # ──────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        return float(np.mean(self.predict(X) == np.array(y)))

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray):
        """Manually computed confusion matrix."""
        y_true = np.array(y)
        y_pred = self.predict(X)
        classes = self.classes_
        idx     = {c: i for i, c in enumerate(classes)}
        n       = len(classes)
        mat     = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            mat[idx[t]][idx[p]] += 1
        return mat, classes

    def classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        """Per-class Precision, Recall, F1 — all computed manually."""
        y_true = np.array(y)
        y_pred = self.predict(X)
        classes = self.classes_

        lines = [f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} "
                 f"{'F1-Score':>10} {'Support':>10}", "─" * 56]

        for cls in classes:
            tp = int(np.sum((y_pred == cls) & (y_true == cls)))
            fp = int(np.sum((y_pred == cls) & (y_true != cls)))
            fn = int(np.sum((y_pred != cls) & (y_true == cls)))
            prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            support = int(np.sum(y_true == cls))
            lines.append(f"{str(cls):<12} {prec:>10.4f} {rec:>10.4f} "
                         f"{f1:>10.4f} {support:>10}")

        acc = float(np.mean(y_pred == y_true))
        lines += ["─" * 56,
                  f"{'Accuracy':<12} {acc:>10.4f}  (n={len(y_true)})\n"]
        report = "\n".join(lines)
        print(report)
        return report

    # ──────────────────────────────────────────
    # Support vector access (SMO only)
    # ──────────────────────────────────────────

    @property
    def support_vectors_(self):
        """Returns support vectors (SMO mode only)."""
        if isinstance(self._opt, SMOOptimizer):
            return self._opt.X_train[self._opt.support_indices_]
        elif isinstance(self._opt, SGDOptimizer):
            # For SGD: approximate SVs as points near the margin
            X  = self._opt.__dict__.get("_X", None)
            return None  # Not stored in SGD mode
        return None

    @property
    def n_support_(self):
        if isinstance(self._opt, SMOOptimizer):
            return len(self._opt.support_indices_)
        return None

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in
                ("kernel", "C", "lr", "n_iters", "gamma", "degree", "coef0", "optimizer")}

    def __repr__(self):
        return (f"SVMClassifier(kernel='{self.kernel}', C={self.C}, "
                f"optimizer='{self.optimizer}')")


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    pos = np.random.randn(50, 2) + [2, 2]
    neg = np.random.randn(50, 2) + [-2, -2]
    X   = np.vstack([pos, neg])
    y   = np.array([1]*50 + [-1]*50)

    perm = np.random.permutation(100)
    X, y = X[perm], y[perm]
    X_tr, X_te = X[:80], X[80:]
    y_tr, y_te = y[:80], y[80:]

    for kern in ["linear", "rbf", "polynomial"]:
        clf = SVMClassifier(kernel=kern, C=1.0, n_iters=200)
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        print(f"[{kern:<12}] Accuracy: {acc*100:.2f}%")