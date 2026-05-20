"""
tests/test_knn.py
------------------
Unit tests for k-NN classifier, distance metrics, and elbow method.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.knn.knn import KNNClassifier
from src.knn.distance_metrics import (
    euclidean_distance, manhattan_distance, minkowski_distance,
    chebyshev_distance, cosine_distance, pairwise_distances,
)
from src.knn.elbow_method import compute_error_rates, find_optimal_k
from src.knn.weighted_knn import WeightedKNNClassifier


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def simple_binary_data():
    """Simple 2D binary classification dataset."""
    np.random.seed(0)
    X_pos = np.random.randn(30, 2) + [3, 3]
    X_neg = np.random.randn(30, 2) + [-3, -3]
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*30 + [0]*30)
    perm = np.random.permutation(60)
    return X[perm], y[perm]


@pytest.fixture
def simple_train_test(simple_binary_data):
    X, y = simple_binary_data
    return X[:48], X[48:], y[:48], y[48:]


# ─── Distance Metric Tests ───────────────────────────────────────────

class TestDistanceMetrics:

    def test_euclidean_same_point(self):
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == pytest.approx(0.0)

    def test_euclidean_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_manhattan_known_value(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert manhattan_distance(a, b) == pytest.approx(7.0)

    def test_minkowski_p2_equals_euclidean(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert minkowski_distance(a, b, p=2) == pytest.approx(euclidean_distance(a, b), rel=1e-5)

    def test_chebyshev_value(self):
        a = np.array([1.0, 5.0])
        b = np.array([3.0, 2.0])
        assert chebyshev_distance(a, b) == pytest.approx(3.0)

    def test_cosine_identical(self):
        a = np.array([1.0, 1.0, 1.0])
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0)

    def test_pairwise_distances_shape(self):
        X = np.random.randn(20, 4)
        x = np.random.randn(4)
        d = pairwise_distances(X, x, "euclidean")
        assert d.shape == (20,)
        assert np.all(d >= 0)

    def test_pairwise_distances_zero_for_self(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        d = pairwise_distances(X, X[0], "euclidean")
        assert d[0] == pytest.approx(0.0)


# ─── KNN Classifier Tests ────────────────────────────────────────────

class TestKNNClassifier:

    def test_fit_stores_data(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        clf = KNNClassifier(k=3)
        clf.fit(X_tr, y_tr)
        assert clf.X_train is not None
        assert len(clf.X_train) == len(X_tr)

    def test_predict_output_shape(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        clf = KNNClassifier(k=3)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_predict_labels_in_training_set(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        clf = KNNClassifier(k=3)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        unique_preds = set(preds)
        unique_train = set(y_tr)
        assert unique_preds.issubset(unique_train)

    def test_high_accuracy_on_separable_data(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        clf = KNNClassifier(k=3)
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        assert acc >= 0.85, f"Expected high accuracy, got {acc:.2f}"

    def test_k1_memorises_training(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        clf = KNNClassifier(k=1)
        clf.fit(X_tr, y_tr)
        assert clf.score(X_tr, y_tr) == pytest.approx(1.0)

    def test_all_metrics_run(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        for metric in ["euclidean", "manhattan", "minkowski", "cosine"]:
            clf = KNNClassifier(k=3, metric=metric)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            assert len(preds) == len(X_te)

    def test_confusion_matrix_shape(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        clf = KNNClassifier(k=3)
        clf.fit(X_tr, y_tr)
        cm, classes = clf.confusion_matrix(X_te, y_te)
        n = len(np.unique(y_te))
        assert cm.shape == (n, n)
        assert cm.sum() == len(y_te)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            KNNClassifier(k=0)

    def test_predict_before_fit_raises(self):
        clf = KNNClassifier(k=3)
        with pytest.raises(RuntimeError):
            clf.predict(np.array([[1, 2]]))


# ─── Weighted KNN Tests ──────────────────────────────────────────────

class TestWeightedKNN:

    def test_inverse_weight_runs(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        clf = WeightedKNNClassifier(k=5, weight="inverse")
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        assert 0 <= acc <= 1

    def test_gaussian_weight_runs(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        clf = WeightedKNNClassifier(k=5, weight="gaussian", sigma=1.0)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert len(preds) == len(X_te)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            WeightedKNNClassifier(weight="unknown")


# ─── Elbow Method Tests ──────────────────────────────────────────────

class TestElbowMethod:

    def test_compute_error_rates_length(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        k_vals, errs, accs = compute_error_rates(
            X_tr, y_tr, X_te, y_te, k_range=range(1, 11), verbose=False
        )
        assert len(k_vals) == 10
        assert len(errs) == 10
        assert len(accs) == 10

    def test_error_rates_in_range(self, simple_train_test):
        X_tr, X_te, y_tr, y_te = simple_train_test
        _, errs, accs = compute_error_rates(
            X_tr, y_tr, X_te, y_te, k_range=range(1, 6), verbose=False
        )
        assert all(0 <= e <= 1 for e in errs)
        assert all(0 <= a <= 1 for a in accs)

    def test_find_optimal_k(self):
        k_vals = [1, 2, 3, 4, 5]
        errs   = [0.3, 0.2, 0.1, 0.15, 0.18]
        opt_k, min_err = find_optimal_k(k_vals, errs)
        assert opt_k == 3
        assert min_err == pytest.approx(0.1)


# ─── Run directly ───────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])