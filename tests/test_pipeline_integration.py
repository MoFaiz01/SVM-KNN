"""Integration tests for end-to-end training pipeline on real data."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.preprocessing import load_iris, stratified_split, normalize
from src.preprocessing.normalization import apply_normalization
from src.knn.knn import KNNClassifier
from src.evaluation.metrics import accuracy
from main import OneVsRestSVM


class TestPipelineIntegration:
    def test_iris_knn_pipeline_accuracy(self):
        X, y, _ = load_iris()
        X_train, X_test, y_train, y_test = stratified_split(
            X, y, test_size=0.2, seed=42
        )
        X_train_norm, stats = normalize(X_train, method="minmax")
        X_test_norm = apply_normalization(X_test, method="minmax", stats=stats)

        clf = KNNClassifier(k=5, metric="euclidean", weighted=False)
        clf.fit(X_train_norm, y_train)
        y_pred = clf.predict(X_test_norm)
        acc = accuracy(y_test, y_pred)

        assert acc >= 0.85, f"KNN integration accuracy too low: {acc:.4f}"

    def test_iris_svm_ovr_pipeline_accuracy(self):
        X, y, _ = load_iris()
        X_train, X_test, y_train, y_test = stratified_split(
            X, y, test_size=0.2, seed=42
        )
        X_train_norm, stats = normalize(X_train, method="minmax")
        X_test_norm = apply_normalization(X_test, method="minmax", stats=stats)

        clf = OneVsRestSVM(kernel="linear", C=1.0, n_iters=400, optimizer="sgd")
        clf.fit(X_train_norm, y_train)
        y_pred = clf.predict(X_test_norm)
        acc = accuracy(y_test, y_pred)

        assert acc >= 0.65, f"SVM OVR integration accuracy too low: {acc:.4f}"
