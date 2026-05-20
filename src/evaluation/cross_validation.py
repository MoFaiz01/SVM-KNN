"""
src/evaluation/cross_validation.py
------------------------------------
K-Fold Cross-Validation implemented from scratch.
Works with any classifier that has fit() and predict() methods.
No Scikit-learn used.
"""

import numpy as np
from .metrics import accuracy, classification_report
from ..preprocessing.train_test_split import k_fold_indices


def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    k:         int  = 5,
    shuffle:   bool = True,
    seed:      int  = 42,
    verbose:   bool = True,
    fit_params: dict = None,
):
    """
    Perform k-fold cross-validation on any classifier.

    Parameters
    ----------
    model      : object with fit(X, y) and predict(X) methods
    X, y       : full dataset
    k          : number of folds (default: 5)
    shuffle    : shuffle before folding
    seed       : random seed
    verbose    : print per-fold results
    fit_params : extra keyword arguments for model.fit()

    Returns
    -------
    results : dict with keys:
        'fold_accuracies' : list[float]
        'mean_accuracy'   : float
        'std_accuracy'    : float
        'best_fold'       : int
        'worst_fold'      : int
    """
    X = np.array(X)
    y = np.array(y)
    if fit_params is None:
        fit_params = {}

    fold_accuracies = []

    if verbose:
        print(f"\n[CV] {k}-Fold Cross-Validation — {type(model).__name__}")
        print("─" * 45)

    for fold_id, (train_idx, val_idx) in enumerate(
        k_fold_indices(len(X), k=k, shuffle=shuffle, seed=seed)
    ):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model.fit(X_tr, y_tr, **fit_params)
        y_pred = model.predict(X_val)

        fold_acc = accuracy(y_val, y_pred)
        fold_accuracies.append(fold_acc)

        if verbose:
            print(f"  Fold {fold_id+1}/{k}  "
                  f"Train: {len(X_tr)}  Val: {len(X_val)}  "
                  f"Accuracy: {fold_acc*100:.2f}%")

    fold_accuracies = np.array(fold_accuracies)
    mean_acc = float(fold_accuracies.mean())
    std_acc  = float(fold_accuracies.std())
    best     = int(np.argmax(fold_accuracies)) + 1
    worst    = int(np.argmin(fold_accuracies)) + 1

    if verbose:
        print("─" * 45)
        print(f"  Mean Accuracy : {mean_acc*100:.2f}%  (±{std_acc*100:.2f}%)")
        print(f"  Best Fold     : Fold {best}  ({fold_accuracies.max()*100:.2f}%)")
        print(f"  Worst Fold    : Fold {worst}  ({fold_accuracies.min()*100:.2f}%)\n")

    return {
        "fold_accuracies": list(fold_accuracies),
        "mean_accuracy":   mean_acc,
        "std_accuracy":    std_acc,
        "best_fold":       best,
        "worst_fold":      worst,
    }


def kfold_cross_validate(
    model_class,
    model_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    k:   int  = 5,
    seed: int = 42,
):
    """
    Cross-validate by re-instantiating the model for each fold.
    Guarantees a fresh model per fold (no state leakage).

    Parameters
    ----------
    model_class  : class — e.g. KNNClassifier
    model_params : dict  — constructor arguments
    X, y         : full dataset
    k, seed      : fold settings

    Returns
    -------
    Same structure as cross_validate()
    """
    class _Wrapper:
        def __init__(self):
            self._model = None

        def fit(self, X, y):
            self._model = model_class(**model_params)
            self._model.fit(X, y)

        def predict(self, X):
            return self._model.predict(X)

    return cross_validate(_Wrapper(), X, y, k=k, seed=seed)


def compare_models(models: dict, X: np.ndarray, y: np.ndarray,
                   k: int = 5, seed: int = 42):
    """
    Run k-fold CV for multiple models and compare.

    Parameters
    ----------
    models : dict {model_name: model_instance}
    X, y   : full dataset

    Returns
    -------
    comparison : dict {model_name: cv_results}
    """
    print(f"\n{'='*55}")
    print(f"  Model Comparison — {k}-Fold Cross-Validation")
    print(f"{'='*55}")

    comparison = {}
    for name, model in models.items():
        print(f"\n[Model] {name}")
        results = cross_validate(model, X, y, k=k, seed=seed, verbose=True)
        comparison[name] = results

    # Summary table
    print(f"\n{'Model':<25} {'Mean Acc':>10} {'Std':>8}")
    print("─" * 45)
    for name, res in comparison.items():
        print(f"{name:<25} {res['mean_accuracy']*100:>9.2f}% "
              f"{res['std_accuracy']*100:>7.2f}%")

    return comparison


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

    from src.knn.knn import KNNClassifier

    np.random.seed(0)
    X = np.random.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = KNNClassifier(k=5, metric="euclidean")
    results = cross_validate(clf, X, y, k=5)
    print("CV Results:", results)