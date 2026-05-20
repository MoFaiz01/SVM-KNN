"""
src/utils/helpers.py
---------------------
Utility helpers: timing, seeding, progress display, result saving.
"""

import time
import numpy as np
import os


def timer(func):
    """Decorator that prints the execution time of any function."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[Timer] {func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def set_seed(seed: int = 42):
    """Set NumPy random seed for reproducibility."""
    np.random.seed(seed)
    print(f"[Seed] Random seed set to {seed}")


def progress_bar(current: int, total: int, width: int = 40, prefix: str = ""):
    """Print a simple ASCII progress bar."""
    filled = int(width * current / total)
    bar    = "█" * filled + "─" * (width - filled)
    pct    = 100 * current / total
    print(f"\r{prefix} [{bar}] {pct:5.1f}%  ({current}/{total})", end="", flush=True)
    if current == total:
        print()


def save_results_txt(results: dict, filepath: str, header: str = ""):
    """Save a results dictionary to a plain text file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        if header:
            f.write(header + "\n")
            f.write("=" * 50 + "\n")
        for key, val in results.items():
            f.write(f"{key}: {val}\n")
    print(f"[Results] Saved → {filepath}")


def flatten(nested_list: list) -> list:
    """Flatten a nested list one level."""
    return [item for sublist in nested_list for item in sublist]


def safe_divide(numerator, denominator, default=0.0):
    """Division that returns default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def check_shapes(X: np.ndarray, y: np.ndarray):
    """Assert compatible shapes for features and labels."""
    assert len(X) == len(y), (
        f"Shape mismatch: X has {len(X)} rows but y has {len(y)} elements."
    )
    return True