"""
main.py
-------
End-to-end experiment runner for KNN and SVM models.
Loads dataset from config.py, preprocesses data, trains models,
evaluates metrics, and saves outputs under results/.
"""

from __future__ import annotations

import os
import sys
import io
from dataclasses import dataclass
from typing import Dict, List
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import config as cfg
from src.preprocessing import (
    load_iris,
    load_breast_cancer,
    load_wine,
    stratified_split,
    train_test_split,
    normalize,
)
from src.preprocessing.normalization import apply_normalization
from src.knn import KNNClassifier, compute_error_rates, find_optimal_k
from src.svm.svm import SVMClassifier
from src.evaluation import accuracy, classification_report, cross_validate, plot_confusion_matrix
from src.visualization.plots import plot_metric_comparison
from src.utils.helpers import save_results_txt, set_seed


@dataclass
class RunResult:
    name: str
    accuracy: float
    report: str


class OneVsRestSVM:
    """Simple One-vs-Rest multiclass adapter using binary SVMClassifier."""

    def __init__(self, **svm_params):
        self.svm_params = svm_params
        self.classes_: np.ndarray | None = None
        self.models_: Dict[object, SVMClassifier] = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_.clear()

        for cls in self.classes_:
            y_bin = np.where(y == cls, 1, -1)
            model = SVMClassifier(**self.svm_params)
            model.fit(X, y_bin)
            self.models_[cls] = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = []
        for cls in self.classes_:
            score = self.models_[cls].decision_function(X)
            scores.append(score)
        score_matrix = np.column_stack(scores)
        idx = np.argmax(score_matrix, axis=1)
        return self.classes_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


def load_selected_dataset(name: str):
    name = name.lower().strip()
    if name == "iris":
        return load_iris()
    if name == "breast_cancer":
        return load_breast_cancer()
    if name == "wine":
        return load_wine()
    raise ValueError(f"Unknown dataset '{name}'. Choose iris, breast_cancer, or wine.")


def ensure_dirs():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_KNN, exist_ok=True)
    os.makedirs(cfg.RESULTS_SVM, exist_ok=True)
    os.makedirs(cfg.RESULTS_CMP, exist_ok=True)


def run_knn(X_train, y_train, X_test, y_test) -> RunResult:
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=True, seed=cfg.RANDOM_SEED
    )
    k_values, err_rates, _ = compute_error_rates(
        X_sub_tr,
        y_sub_tr,
        X_sub_val,
        y_sub_val,
        k_range=cfg.K_RANGE,
        metric=cfg.KNN_METRIC,
        verbose=False,
    )
    best_k, best_err = find_optimal_k(k_values, err_rates)
    chosen_k = best_k if cfg.KNN_K is None else cfg.KNN_K

    knn = KNNClassifier(k=chosen_k, metric=cfg.KNN_METRIC, weighted=cfg.KNN_WEIGHTED)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    save_results_txt(
        {
            "dataset": cfg.DATASET,
            "k": chosen_k,
            "validation_best_k": best_k,
            "validation_best_error": f"{best_err:.6f}",
            "metric": cfg.KNN_METRIC,
            "weighted": cfg.KNN_WEIGHTED,
            "test_accuracy": f"{acc:.6f}",
        },
        os.path.join(cfg.RESULTS_KNN, "accuracy_results.txt"),
        header="KNN Results",
    )
    plot_confusion_matrix(
        y_test,
        y_pred,
        title=f"KNN Confusion Matrix ({cfg.DATASET})",
        save_path=os.path.join(cfg.RESULTS_KNN, "confusion_matrix.png"),
        show=False,
    )
    return RunResult(name="KNN", accuracy=acc, report=report)


def run_svm(X_train, y_train, X_test, y_test) -> RunResult:
    classes = np.unique(y_train)
    n_iters = cfg.SVM_ITERS
    if len(classes) > 2 and (cfg.SVM_OPTIMIZER in (None, "smo")):
        n_iters = min(cfg.SVM_ITERS, 200)

    svm_params = dict(
        kernel=cfg.SVM_KERNEL,
        C=cfg.SVM_C,
        lr=cfg.SVM_LR,
        n_iters=n_iters,
        gamma=cfg.SVM_GAMMA,
        degree=cfg.SVM_DEGREE,
        coef0=cfg.SVM_COEF0,
        optimizer=cfg.SVM_OPTIMIZER,
    )

    if len(classes) == 2:
        model = SVMClassifier(**svm_params)
    else:
        model = OneVsRestSVM(**svm_params)

    # SMO training can emit thousands of lines; keep terminal output manageable.
    with redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    save_results_txt(
        {
            "dataset": cfg.DATASET,
            "kernel": cfg.SVM_KERNEL,
            "optimizer": cfg.SVM_OPTIMIZER if cfg.SVM_OPTIMIZER else "auto",
            "C": cfg.SVM_C,
            "gamma": cfg.SVM_GAMMA,
            "degree": cfg.SVM_DEGREE,
            "coef0": cfg.SVM_COEF0,
            "test_accuracy": f"{acc:.6f}",
        },
        os.path.join(cfg.RESULTS_SVM, f"{cfg.SVM_KERNEL}_kernel_results.txt"),
        header="SVM Results",
    )
    if cfg.SVM_KERNEL == "linear":
        plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"SVM Confusion Matrix ({cfg.DATASET})",
            save_path=os.path.join(cfg.RESULTS_SVM, "support_vectors_plot.png"),
            show=False,
        )
    return RunResult(name="SVM", accuracy=acc, report=report)


def maybe_run_cv(model_name: str, model_obj, X: np.ndarray, y: np.ndarray):
    if not cfg.RUN_CV:
        return None
    return cross_validate(
        model_obj,
        X,
        y,
        k=cfg.CV_FOLDS,
        seed=cfg.RANDOM_SEED,
        verbose=cfg.VERBOSE,
    )


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ensure_dirs()
    set_seed(cfg.RANDOM_SEED)

    X, y, feature_names = load_selected_dataset(cfg.DATASET)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=cfg.TEST_SIZE, seed=cfg.RANDOM_SEED
    )
    X_train_norm, stats = normalize(X_train, method=cfg.NORMALIZATION)
    X_test_norm = apply_normalization(X_test, method=cfg.NORMALIZATION, stats=stats)

    print(f"[Data] Dataset={cfg.DATASET} | train={len(X_train)} | test={len(X_test)}")
    print(f"[Data] Features={len(feature_names)} | classes={len(np.unique(y))}")
    print(f"[Data] Normalization={cfg.NORMALIZATION}")

    run_results: List[RunResult] = []

    if cfg.RUN_KNN:
        run_results.append(run_knn(X_train_norm, y_train, X_test_norm, y_test))

    if cfg.RUN_SVM:
        run_results.append(run_svm(X_train_norm, y_train, X_test_norm, y_test))

    if run_results:
        comp_df = pd.DataFrame(
            {"model": [r.name for r in run_results], "accuracy": [r.accuracy for r in run_results]}
        )
        comp_path = os.path.join(cfg.RESULTS_CMP, "model_comparison.csv")
        comp_df.to_csv(comp_path, index=False)
        print(f"[Results] Saved -> {comp_path}")

        plot_metric_comparison(
            comp_df["model"].tolist(),
            comp_df["accuracy"].tolist(),
            title=f"Model Accuracy Comparison ({cfg.DATASET})",
            save_path=os.path.join(cfg.RESULTS_CMP, "accuracy_comparison_graph.png"),
            show=False,
        )

    print("\n[Summary]")
    for r in run_results:
        print(f"  {r.name}: {r.accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
