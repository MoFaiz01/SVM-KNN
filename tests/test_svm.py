"""
tests/test_svm.py
------------------
Unit tests for SVM classifier and optimisers.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.svm.svm import SVMClassifier
from src.svm.optimizer import SGDOptimizer, SMOOptimizer
from src.svm.kernels import linear_kernel, rbf_kernel, polynomial_kernel


@pytest.fixture
def binary_data():
    np.random.seed(42)
    pos = np.random.randn(40, 2) + [3, 3]
    neg = np.random.randn(40, 2) + [-3, -3]
    X = np.vstack([pos, neg])
    y = np.array([1]*40 + [-1]*40)
    perm = np.random.permutation(80)
    X, y = X[perm], y[perm]
    return X[:64], X[64:], y[:64], y[64:]


class TestSGDOptimizer:
    def test_fit_runs(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        opt = SGDOptimizer(C=1.0, lr=0.01, n_iters=200, decay=True)
        opt.fit(X_tr, y_tr)
        assert opt.w is not None
        assert opt.w.shape == (X_tr.shape[1],)

    def test_predict_binary(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        opt = SGDOptimizer(C=1.0, lr=0.01, n_iters=300)
        opt.fit(X_tr, y_tr)
        preds = opt.predict(X_te)
        assert set(preds).issubset({-1, 1})

    def test_high_accuracy_separable(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        opt = SGDOptimizer(C=1.0, lr=0.01, n_iters=500)
        opt.fit(X_tr, y_tr)
        acc = np.mean(opt.predict(X_te) == y_te)
        assert acc >= 0.85

    def test_loss_decreasing(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        opt = SGDOptimizer(C=1.0, lr=0.01, n_iters=300, decay=True)
        opt.fit(X_tr, y_tr)
        # Loss should be lower in second half than first half
        first_half  = np.mean(opt.loss_history[:150])
        second_half = np.mean(opt.loss_history[150:])
        assert second_half < first_half


class TestSVMClassifier:
    def test_fit_and_predict_linear(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        clf = SVMClassifier(kernel="linear", C=1.0, n_iters=300)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert len(preds) == len(X_te)

    def test_score_reasonable(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        clf = SVMClassifier(kernel="linear", C=1.0, n_iters=400)
        clf.fit(X_tr, y_tr)
        assert clf.score(X_te, y_te) >= 0.8

    def test_decision_function_shape(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        clf = SVMClassifier(kernel="linear", n_iters=200)
        clf.fit(X_tr, y_tr)
        df = clf.decision_function(X_te)
        assert df.shape == (len(X_te),)

    def test_label_encoding_any_labels(self):
        np.random.seed(0)
        X = np.random.randn(60, 2)
        X[:30] += 3; X[30:] -= 3
        y = np.array(["cat"]*30 + ["dog"]*30)
        clf = SVMClassifier(kernel="linear", n_iters=200)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({"cat", "dog"})

    def test_three_classes_raises(self):
        X = np.random.randn(30, 2)
        y = np.array([0]*10 + [1]*10 + [2]*10)
        clf = SVMClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])