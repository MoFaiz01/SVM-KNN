"""
config.py
----------
Central configuration file for the project.
Override defaults here before running main.py or notebooks.
"""

# ──────────────────────────────────────────────
# Dataset Configuration
# ──────────────────────────────────────────────
DATASET = "iris"           # "iris" | "breast_cancer" | "wine"
NORMALIZATION = "minmax"   # "minmax" | "zscore" | "l2" | "robust"
TEST_SIZE  = 0.2
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# k-NN Configuration
# ──────────────────────────────────────────────
KNN_K         = 5                       # default k (overridden by elbow method)
KNN_METRIC    = "euclidean"             # distance metric
KNN_WEIGHTED  = False                   # use weighted voting
KNN_WEIGHT_SCHEME = "inverse"           # "inverse" | "gaussian"
KNN_SIGMA     = 1.0                     # Gaussian bandwidth (if weighted)
K_RANGE       = range(1, 31)            # range for elbow method

# ──────────────────────────────────────────────
# SVM Configuration
# ──────────────────────────────────────────────
SVM_KERNEL    = "rbf"      # "linear" | "polynomial" | "rbf"
SVM_C         = 1.0        # regularisation
SVM_GAMMA     = None       # auto = 1/n_features
SVM_DEGREE    = 3          # polynomial degree
SVM_COEF0     = 1.0        # polynomial independent term
SVM_LR        = 0.001      # learning rate (SGD)
SVM_ITERS     = 1000       # epochs / max SMO passes
SVM_OPTIMIZER = None       # None = auto-select

# ──────────────────────────────────────────────
# Cross-Validation
# ──────────────────────────────────────────────
CV_FOLDS = 5

# ──────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────
RESULTS_DIR   = "results/"
RESULTS_KNN   = "results/knn/"
RESULTS_SVM   = "results/svm/"
RESULTS_CMP   = "results/comparison/"

# ──────────────────────────────────────────────
# Feature flags
# ──────────────────────────────────────────────
RUN_KNN   = True
RUN_SVM   = True
RUN_CV    = True
SAVE_PLOTS = True
VERBOSE    = True