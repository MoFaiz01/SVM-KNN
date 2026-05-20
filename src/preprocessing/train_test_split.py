"""
src/preprocessing/train_test_split.py
---------------------------------------
Manual train-test splitting utilities.
Includes random split, stratified split, and k-fold CV indices.
No Scikit-learn used.
"""

import numpy as np


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     shuffle: bool = True,
                     seed: int = 42):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X         : np.ndarray, shape (n, d)
    y         : np.ndarray, shape (n,)
    test_size : float — fraction for test set  (default: 0.2)
    shuffle   : bool  — shuffle before split   (default: True)
    seed      : int   — random seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = np.array(X)
    y = np.array(y)
    n = len(X)

    if shuffle:
        np.random.seed(seed)
        idx = np.random.permutation(n)
        X, y = X[idx], y[idx]

    split = int(n * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


def stratified_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     seed: int = 42):
    """
    Stratified train-test split — preserves class proportions.

    Parameters
    ----------
    X, y      : arrays
    test_size : float — fraction for test set
    seed      : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    np.random.seed(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    # Shuffle the final indices
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def k_fold_indices(n: int, k: int = 5, shuffle: bool = True, seed: int = 42):
    """
    Generate train/validation index pairs for k-fold cross-validation.

    Parameters
    ----------
    n       : int  — total number of samples
    k       : int  — number of folds          (default: 5)
    shuffle : bool — shuffle before folding
    seed    : int

    Yields
    ------
    (train_idx, val_idx) for each fold
    """
    idx = np.arange(n)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(idx)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1  # distribute remainder

    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(idx[current: current + size])
        current += size

    for fold_id in range(k):
        val_idx   = folds[fold_id]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != fold_id])
        yield train_idx, val_idx


def validation_split(X: np.ndarray, y: np.ndarray,
                     val_size: float = 0.1, seed: int = 42):
    """
    Split training data into train + validation sets.
    Wraps train_test_split for convenience.
    """
    return train_test_split(X, y, test_size=val_size, seed=seed)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = np.array(["A"]*40 + ["B"]*35 + ["C"]*25)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    print(f"Random split    — Train: {len(X_tr)}, Test: {len(X_te)}")

    X_tr2, X_te2, y_tr2, y_te2 = stratified_split(X, y, test_size=0.2)
    print(f"Stratified split — Train: {len(X_tr2)}, Test: {len(X_te2)}")
    for cls in np.unique(y):
        print(f"  Class {cls}: train={np.sum(y_tr2==cls)}, test={np.sum(y_te2==cls)}")

    print("\n5-Fold CV indices:")
    for fold, (ti, vi) in enumerate(k_fold_indices(100, k=5)):
        print(f"  Fold {fold+1}: train={len(ti)}, val={len(vi)}")