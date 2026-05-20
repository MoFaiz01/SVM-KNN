"""tests/test_metrics.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.evaluation.metrics import accuracy, precision, recall, f1_score
from src.evaluation.confusion_matrix import confusion_matrix
from src.preprocessing.normalization import min_max_normalize, z_score_normalize
from src.preprocessing.encoding import encode_labels, encode_binary
from src.preprocessing.train_test_split import train_test_split, stratified_split

class TestMetrics:
    def test_perfect_accuracy(self):
        y = np.array([0, 1, 2, 0, 1])
        assert accuracy(y, y) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_precision_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert precision(y_true, y_pred, cls=1) == pytest.approx(1.0)

    def test_recall_zero(self):
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])
        assert recall(y_true, y_pred, cls=1) == pytest.approx(0.0)

    def test_f1_harmonic_mean(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        p = precision(y_true, y_pred, 1)
        r = recall(y_true, y_pred, 1)
        expected_f1 = 2 * p * r / (p + r)
        assert f1_score(y_true, y_pred, 1) == pytest.approx(expected_f1)

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        cm, cls = confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm.sum() == 6

    def test_confusion_matrix_diagonal_correct(self):
        y = np.array([0, 1, 2])
        cm, _ = confusion_matrix(y, y)
        assert np.trace(cm) == 3

class TestPreprocessing:
    def test_minmax_range(self):
        X = np.array([[1,2],[3,4],[5,6]], dtype=float)
        Xs, mn, mx = min_max_normalize(X)
        assert Xs.min() == pytest.approx(0.0)
        assert Xs.max() == pytest.approx(1.0)

    def test_zscore_mean_zero(self):
        X = np.random.randn(100, 4) * 10 + 5
        Xs, _, _ = z_score_normalize(X)
        assert np.abs(Xs.mean(axis=0)).max() < 1e-10

    def test_encode_labels_consistent(self):
        y = ["a", "b", "a", "c"]
        y_enc, lmap, classes = encode_labels(y)
        assert len(y_enc) == 4
        assert set(y_enc) == {0, 1, 2}

    def test_encode_binary_pm1(self):
        y = ["cat", "dog", "dog", "cat"]
        y_bin, pos, neg = encode_binary(y, positive_class="dog")
        assert set(y_bin) == {-1, 1}

    def test_train_test_split_sizes(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        assert len(X_tr) == 40 and len(X_te) == 10

    def test_stratified_split_proportions(self):
        y = np.array([0]*60 + [1]*40)
        X = np.random.randn(100, 2)
        _, _, y_tr, y_te = stratified_split(X, y, test_size=0.2)
        ratio_train = np.sum(y_tr == 0) / len(y_tr)
        ratio_test  = np.sum(y_te == 0) / len(y_te)
        assert abs(ratio_train - ratio_test) < 0.1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])