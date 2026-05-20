"""
src/svm/optimizer.py
---------------------
Two optimisation strategies for SVM from scratch:
  1. SGDOptimizer  — Stochastic Gradient Descent (primal, linear SVM)
  2. SMOOptimizer  — Sequential Minimal Optimisation (dual, any kernel)
No Scikit-learn used.
"""

import numpy as np


# ══════════════════════════════════════════════
# 1. SGD Optimiser — Primal Soft-Margin Linear SVM
# ══════════════════════════════════════════════

class SGDOptimizer:
    """
    Trains a linear soft-margin SVM via Stochastic Gradient Descent.

    Objective:  min  ½||w||²  +  C·Σ max(0, 1 − yᵢ(w·xᵢ + b))

    Parameters
    ----------
    C       : regularisation strength (penalty for margin violations)
    lr      : learning rate
    n_iters : training epochs
    decay   : whether to decay learning rate over epochs
    """

    def __init__(self, C=1.0, lr=0.001, n_iters=1000, decay=True):
        self.C        = C
        self.lr       = lr
        self.n_iters  = n_iters
        self.decay    = decay
        self.w        = None
        self.b        = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train linear SVM. Labels must be {-1, +1}.
        Returns self.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.n_iters):
            lr_t = self.lr / (1.0 + 0.01 * epoch) if self.decay else self.lr

            # Shuffle
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for i in idx:
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                hinge  = max(0.0, 1.0 - margin)
                epoch_loss += 0.5 * np.dot(self.w, self.w) + self.C * hinge

                if margin < 1:
                    # Misclassified or inside margin — update both w and b
                    self.w -= lr_t * (self.w - self.C * y[i] * X[i])
                    self.b += lr_t * self.C * y[i]
                else:
                    # Correctly classified — only regularise w
                    self.w -= lr_t * self.w

            self.loss_history.append(epoch_loss / n_samples)

            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:>5}/{self.n_iters}  "
                      f"Loss: {self.loss_history[-1]:.6f}")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X)).astype(int)


# ══════════════════════════════════════════════
# 2. SMO Optimiser — Dual Kernel SVM
# ══════════════════════════════════════════════

class SMOOptimizer:
    """
    Sequential Minimal Optimisation (Platt, 1998) for kernel SVM.

    Solves the dual:
        max  Σᵢ αᵢ  −  ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ,xⱼ)
        s.t. 0 ≤ αᵢ ≤ C,  Σᵢ αᵢ yᵢ = 0

    Parameters
    ----------
    C         : regularisation (box constraint on αᵢ)
    tol       : KKT tolerance
    max_iter  : maximum passes over training set without alpha change
    kernel_fn : callable K(x1, x2) → float
    """

    def __init__(self, C=1.0, tol=1e-3, max_iter=100, kernel_fn=None):
        from .kernels import linear_kernel
        self.C         = C
        self.tol       = tol
        self.max_iter  = max_iter
        self.kernel_fn = kernel_fn if kernel_fn is not None else linear_kernel

        self.alphas    = None
        self.b         = 0.0
        self.X_train   = None
        self.y_train   = None
        self.K         = None   # kernel matrix cache

    def _kernel(self, i, j):
        return self.kernel_fn(self.X_train[i], self.X_train[j])

    def _decision(self, i):
        return float(np.sum(
            self.alphas * self.y_train *
            np.array([self._kernel(k, i) for k in range(len(self.X_train))])
        ) + self.b)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train kernel SVM via SMO. Labels must be {-1, +1}.
        Returns self.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n = len(X)

        self.X_train = X
        self.y_train = y
        self.alphas  = np.zeros(n)
        self.b       = 0.0

        passes = 0
        while passes < self.max_iter:
            num_changed = 0

            for i in range(n):
                Ei = self._decision(i) - y[i]

                # Check KKT condition
                if ((y[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                        (y[i] * Ei > self.tol and self.alphas[i] > 0)):

                    # Pick j ≠ i randomly
                    j = i
                    while j == i:
                        j = np.random.randint(0, n)

                    Ej = self._decision(j) - y[j]

                    ai_old = self.alphas[i]
                    aj_old = self.alphas[j]

                    # Compute L and H (box constraints)
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L >= H:
                        continue

                    # Compute eta (second derivative of objective)
                    eta = (2 * self._kernel(i, j)
                           - self._kernel(i, i)
                           - self._kernel(j, j))
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j]  = np.clip(self.alphas[j], L, H)

                    if abs(self.alphas[j] - aj_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alphas[i] += y[i] * y[j] * (aj_old - self.alphas[j])

                    # Update bias b
                    b1 = (self.b - Ei
                          - y[i] * (self.alphas[i] - ai_old) * self._kernel(i, i)
                          - y[j] * (self.alphas[j] - aj_old) * self._kernel(i, j))
                    b2 = (self.b - Ej
                          - y[i] * (self.alphas[i] - ai_old) * self._kernel(i, j)
                          - y[j] * (self.alphas[j] - aj_old) * self._kernel(j, j))

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0
            print(f"  SMO pass — alphas changed: {num_changed}")

        self.support_indices_ = np.where(self.alphas > 1e-5)[0]
        print(f"[SMO] Done — {len(self.support_indices_)} support vectors found.")
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        scores = np.zeros(len(X))
        for i, x in enumerate(X):
            s = 0.0
            for j in range(len(self.X_train)):
                s += (self.alphas[j] * self.y_train[j] *
                      self.kernel_fn(self.X_train[j], x))
            scores[i] = s + self.b
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X)).astype(int)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(1)
    X = np.vstack([np.random.randn(40, 2) + [2, 2],
                   np.random.randn(40, 2) + [-2, -2]])
    y = np.array([1]*40 + [-1]*40)

    print("=== SGD Optimizer ===")
    sgd = SGDOptimizer(C=1.0, lr=0.01, n_iters=500)
    sgd.fit(X, y)
    preds = sgd.predict(X)
    print(f"SGD Accuracy: {np.mean(preds == y)*100:.2f}%")

    print("\n=== SMO Optimizer ===")
    from kernels import linear_kernel
    smo = SMOOptimizer(C=1.0, max_iter=20, kernel_fn=linear_kernel)
    smo.fit(X, y)
    preds2 = smo.predict(X)
    print(f"SMO Accuracy: {np.mean(preds2 == y)*100:.2f}%")