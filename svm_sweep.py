"""Run a simple SVM hyperparameter sweep and save ranked results."""

from __future__ import annotations

import io
import os
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import config as cfg
from main import OneVsRestSVM
from src.evaluation.metrics import accuracy
from src.preprocessing import (
    load_breast_cancer,
    load_iris,
    load_wine,
    normalize,
    stratified_split,
)
from src.preprocessing.normalization import apply_normalization
from src.svm.svm import SVMClassifier


def load_selected_dataset(name: str):
    name = name.lower().strip()
    if name == "iris":
        return load_iris()
    if name == "breast_cancer":
        return load_breast_cancer()
    if name == "wine":
        return load_wine()
    raise ValueError(f"Unknown dataset '{name}'")


def build_model(y_train: np.ndarray, params: dict):
    if len(np.unique(y_train)) == 2:
        return SVMClassifier(**params)
    return OneVsRestSVM(**params)


def run_sweep():
    X, y, _ = load_selected_dataset(cfg.DATASET)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=cfg.TEST_SIZE, seed=cfg.RANDOM_SEED
    )
    X_train_norm, stats = normalize(X_train, method=cfg.NORMALIZATION)
    X_test_norm = apply_normalization(X_test, method=cfg.NORMALIZATION, stats=stats)

    # Keep the sweep practical for from-scratch SMO implementations.
    kernels = ["linear", "rbf"]
    c_values = [0.5, 1.0, 2.0]
    gamma_values = [None, 0.1]
    iter_values = [80, 120]

    results = []
    for kernel in kernels:
        for c_val in c_values:
            for n_iters in iter_values:
                gamma_grid = [None] if kernel == "linear" else gamma_values
                for gamma in gamma_grid:
                    params = dict(
                        kernel=kernel,
                        C=c_val,
                        lr=cfg.SVM_LR,
                        n_iters=n_iters,
                        gamma=gamma,
                        degree=cfg.SVM_DEGREE,
                        coef0=cfg.SVM_COEF0,
                        optimizer="sgd" if kernel == "linear" else "smo",
                    )
                    model = build_model(y_train, params)

                    t0 = time.time()
                    with redirect_stdout(io.StringIO()):
                        model.fit(X_train_norm, y_train)
                    y_pred = model.predict(X_test_norm)
                    elapsed = time.time() - t0

                    results.append(
                        {
                            "dataset": cfg.DATASET,
                            "normalization": cfg.NORMALIZATION,
                            "kernel": kernel,
                            "C": c_val,
                            "gamma": gamma,
                            "n_iters": n_iters,
                            "optimizer": params["optimizer"],
                            "accuracy": float(accuracy(y_test, y_pred)),
                            "train_eval_seconds": round(elapsed, 4),
                        }
                    )
                    print(
                        f"done kernel={kernel:<10} C={c_val:<3} gamma={str(gamma):<4} "
                        f"iters={n_iters:<3} acc={results[-1]['accuracy']:.4f}",
                        flush=True,
                    )

    df = pd.DataFrame(results).sort_values(
        by=["accuracy", "train_eval_seconds"], ascending=[False, True]
    )
    os.makedirs(cfg.RESULTS_CMP, exist_ok=True)
    out_path = os.path.join(cfg.RESULTS_CMP, "svm_sweep_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved ranked sweep results -> {out_path}")
    print("\nTop 10:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    run_sweep()
