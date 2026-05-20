# src/utils/constants.py
"""Project-wide constants."""

# Dataset file paths
DATASETS = {
    "iris":          "data/raw/iris.csv",
    "breast_cancer": "data/raw/breast_cancer.csv",
    "wine":          "data/raw/wine.csv",
}

# Available distance metrics
METRICS = ["euclidean", "manhattan", "minkowski", "chebyshev", "cosine", "hamming"]

# Available SVM kernels
KERNELS = ["linear", "polynomial", "rbf", "sigmoid"]

# k-NN defaults
DEFAULT_K       = 5
K_RANGE         = range(1, 31)

# SVM defaults
DEFAULT_C       = 1.0
DEFAULT_GAMMA   = None      # auto: 1/n_features
DEFAULT_DEGREE  = 3
DEFAULT_COEF0   = 1.0
DEFAULT_LR      = 0.001
DEFAULT_ITERS   = 1000

# Result directories
RESULTS_KNN     = "results/knn/"
RESULTS_SVM     = "results/svm/"
RESULTS_COMPARE = "results/comparison/"

# Normalization default
DEFAULT_NORM    = "minmax"

# Cross-validation
CV_FOLDS        = 5
RANDOM_SEED     = 42

# Plot settings
FIG_DPI         = 150
FIG_SIZE_SMALL  = (8, 5)
FIG_SIZE_MEDIUM = (10, 6)
FIG_SIZE_LARGE  = (14, 6)