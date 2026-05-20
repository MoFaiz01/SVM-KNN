"""
src/preprocessing/encoding.py
-------------------------------
Label encoding utilities built from scratch.
Converts string/categorical labels to integers and back.
No Scikit-learn used.
"""

import numpy as np


def encode_labels(y):
    """
    Convert string/categorical labels to integer class indices.

    Parameters
    ----------
    y : array-like — original labels (strings, ints, floats)

    Returns
    -------
    y_encoded  : np.ndarray of int — 0, 1, 2, ...
    label_map  : dict  {original_label → int}
    classes    : list  [sorted unique class labels]
    """
    y = np.array(y)
    classes   = sorted(set(y), key=lambda x: str(x))
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    y_encoded = np.array([label_map[yi] for yi in y], dtype=int)
    return y_encoded, label_map, classes


def decode_labels(y_encoded, label_map: dict):
    """
    Convert integer class indices back to original labels.

    Parameters
    ----------
    y_encoded : np.ndarray of int
    label_map : dict  {original_label → int}  (same as returned by encode_labels)

    Returns
    -------
    y_original : np.ndarray
    """
    inv_map = {v: k for k, v in label_map.items()}
    return np.array([inv_map[yi] for yi in y_encoded])


def encode_binary(y, positive_class=None):
    """
    Encode a binary classification label array as {-1, +1}.
    Required format for SVM training.

    Parameters
    ----------
    y              : array-like — two-class labels
    positive_class : value to assign +1 (default: second class alphabetically)

    Returns
    -------
    y_binary     : np.ndarray of {-1, +1}
    pos_class    : the class mapped to +1
    neg_class    : the class mapped to -1
    """
    y = np.array(y)
    classes = sorted(set(y), key=lambda x: str(x))
    if len(classes) != 2:
        raise ValueError(f"encode_binary requires exactly 2 classes, got {len(classes)}")

    if positive_class is None:
        positive_class = classes[1]
    negative_class = [c for c in classes if c != positive_class][0]

    y_binary = np.where(y == positive_class, 1, -1).astype(int)
    return y_binary, positive_class, negative_class


def one_hot_encode(y, n_classes: int = None):
    """
    One-hot encode an integer label array.

    Parameters
    ----------
    y         : np.ndarray of int — class indices 0..C-1
    n_classes : int | None — number of classes (inferred if None)

    Returns
    -------
    ohe : np.ndarray, shape (n_samples, n_classes)
          Each row has exactly one 1 and rest 0s.
    """
    y = np.array(y, dtype=int)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    ohe = np.zeros((len(y), n_classes), dtype=int)
    ohe[np.arange(len(y)), y] = 1
    return ohe


def ordinal_encode(y, order: list = None):
    """
    Encode ordinal labels by their position in a specified order.

    Parameters
    ----------
    y     : array-like — labels to encode
    order : list       — desired ordering, e.g. ['low','medium','high']
                         If None, uses alphabetical order.

    Returns
    -------
    y_ordinal : np.ndarray of int
    order     : list — the ordering used
    """
    y = np.array(y)
    if order is None:
        order = sorted(set(y), key=lambda x: str(x))
    order_map = {cls: i for i, cls in enumerate(order)}
    y_ordinal = np.array([order_map[yi] for yi in y], dtype=int)
    return y_ordinal, order


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    labels = ["setosa", "versicolor", "virginica",
              "setosa", "virginica", "versicolor"]

    y_enc, lmap, classes = encode_labels(labels)
    print("Original    :", labels)
    print("Encoded     :", y_enc)
    print("Label map   :", lmap)

    y_dec = decode_labels(y_enc, lmap)
    print("Decoded     :", list(y_dec))

    print("\nOne-Hot Encoded:")
    print(one_hot_encode(y_enc))

    bin_labels = ["cat", "dog", "cat", "dog", "dog"]
    y_bin, pos, neg = encode_binary(bin_labels, positive_class="dog")
    print(f"\nBinary encoded (pos={pos}, neg={neg}):", y_bin)