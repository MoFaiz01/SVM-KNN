"""
src/preprocessing/data_loader.py
----------------------------------
Load datasets (Iris, Breast Cancer, Wine) from CSV files or
generate synthetic fallbacks when files are not present.
No Scikit-learn used.
"""

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Generic CSV loader
# ──────────────────────────────────────────────

def load_csv(path: str, target_col: int = -1, header: bool = True):
    """
    Load any CSV dataset.

    Parameters
    ----------
    path       : str  — path to CSV file
    target_col : int  — column index for labels (-1 = last column)
    header     : bool — whether the CSV has a header row

    Returns
    -------
    X      : np.ndarray, shape (n_samples, n_features)
    y      : np.ndarray, shape (n_samples,)
    columns: list of feature names (or generated names)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, header=0 if header else None)

    if target_col == -1:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values.astype(float)
        columns = list(df.columns[:-1])
    else:
        y = df.iloc[:, target_col].values
        X = df.drop(df.columns[target_col], axis=1).values.astype(float)
        columns = [c for i, c in enumerate(df.columns) if i != target_col]

    return X, y, columns


# ──────────────────────────────────────────────
# Named dataset loaders with fallback synthesis
# ──────────────────────────────────────────────

def load_iris(path: str = "data/raw/iris.csv"):
    """
    Load Iris dataset.
    Expected columns: sepal_length, sepal_width, petal_length, petal_width, species
    Falls back to synthetic Iris-like data if file not found.

    Returns
    -------
    X      : np.ndarray, shape (150, 4)
    y      : np.ndarray of string labels
    names  : list of feature names
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        X  = df.iloc[:, :4].values.astype(float)
        y  = df.iloc[:, 4].values
        return X, y, list(df.columns[:4])
    else:
        print(f"[Loader] '{path}' not found — using synthetic Iris data.")
        np.random.seed(42)
        c0 = np.random.randn(50, 4) * 0.35 + [5.0, 3.4, 1.5, 0.3]
        c1 = np.random.randn(50, 4) * 0.40 + [5.9, 2.8, 4.3, 1.3]
        c2 = np.random.randn(50, 4) * 0.50 + [6.6, 3.0, 5.5, 2.0]
        X  = np.vstack([c0, c1, c2])
        y  = np.array(["setosa"]*50 + ["versicolor"]*50 + ["virginica"]*50)
        return X, y, ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_breast_cancer(path: str = "data/raw/breast_cancer.csv"):
    """
    Load Breast Cancer Wisconsin dataset.
    Expected: 30 feature columns + 'diagnosis' column (M/B or 0/1).
    Falls back to synthetic binary data if file not found.

    Returns
    -------
    X      : np.ndarray, shape (n, 30)
    y      : np.ndarray of labels
    names  : list of feature names
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Drop id column if present
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        # Find target column
        target = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
        y = df[target].values
        X = df.drop(columns=[target]).values.astype(float)
        return X, y, [c for c in df.columns if c != target]
    else:
        print(f"[Loader] '{path}' not found — using synthetic binary data.")
        np.random.seed(7)
        n_feat = 30
        X0 = np.random.randn(212, n_feat) * 0.5 + 1.0   # benign
        X1 = np.random.randn(357, n_feat) * 0.8 + 0.0   # malignant
        X  = np.vstack([X0, X1])
        y  = np.array(["B"]*212 + ["M"]*357)
        features = [f"feature_{i}" for i in range(n_feat)]
        return X, y, features


def load_wine(path: str = "data/raw/wine.csv"):
    """
    Load Wine dataset.
    Expected: 13 feature columns + 'class' column (1/2/3).
    Falls back to synthetic 3-class data if file not found.

    Returns
    -------
    X      : np.ndarray, shape (n, 13)
    y      : np.ndarray of labels
    names  : list of feature names
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        target = "class" if "class" in df.columns else df.columns[0]
        y  = df[target].values
        X  = df.drop(columns=[target]).values.astype(float)
        return X, y, [c for c in df.columns if c != target]
    else:
        print(f"[Loader] '{path}' not found — using synthetic wine-like data.")
        np.random.seed(13)
        c1 = np.random.randn(59,  13) + np.random.randn(13) * 2
        c2 = np.random.randn(71,  13) + np.random.randn(13) * 2
        c3 = np.random.randn(48,  13) + np.random.randn(13) * 2
        X  = np.vstack([c1, c2, c3])
        y  = np.array([1]*59 + [2]*71 + [3]*48)
        features = [f"feature_{i}" for i in range(13)]
        return X, y, features


# ──────────────────────────────────────────────
# Dataset info
# ──────────────────────────────────────────────

def dataset_info(X: np.ndarray, y: np.ndarray,
                 feature_names: list = None, name: str = "Dataset"):
    """Print a quick summary of a loaded dataset."""
    classes, counts = np.unique(y, return_counts=True)
    print(f"\n{'─'*45}")
    print(f"  Dataset      : {name}")
    print(f"  Samples      : {len(X)}")
    print(f"  Features     : {X.shape[1]}")
    print(f"  Classes      : {len(classes)}")
    for cls, cnt in zip(classes, counts):
        print(f"    {cls}: {cnt} samples")
    print(f"  Missing vals : {np.isnan(X).sum()}")
    if feature_names:
        print(f"  Features     : {feature_names}")
    print(f"{'─'*45}\n")


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    X, y, feat = load_iris()
    dataset_info(X, y, feat, name="Iris")

    X2, y2, feat2 = load_breast_cancer()
    dataset_info(X2, y2, feat2, name="Breast Cancer")

    X3, y3, feat3 = load_wine()
    dataset_info(X3, y3, feat3, name="Wine")