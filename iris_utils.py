"""
Provided starter code -- Homework: From Two Binary Classifiers to Softmax
Regression on Iris (introductory version).

You do NOT need to modify or hand in this file. It gives you working,
from-scratch (numpy + csv only) implementations of the data-handling and
scoring utilities so you can focus on the models themselves in Parts 1-3.

Put iris_utils.py and iris.csv in the same folder as your solution and:

    from iris_utils import *
    X, y = load_iris_csv("iris.csv")
    Xtr_raw, Xte_raw, ytr, yte = stratified_split(X, y, test_fraction=0.3, seed=0)
    Xtr, Xte, mu, sd = standardize(Xtr_raw, Xte_raw)
"""
import csv
import numpy as np

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal length", "sepal width", "petal length", "petal width"]


def load_iris_csv(path="iris.csv"):
    """Load iris.csv and encode species as 0=setosa, 1=versicolor, 2=virginica."""
    name_to_id = {"setosa": 0, "versicolor": 1, "virginica": 2}
    X_rows, y_rows = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_rows.append([float(row["sepal_length"]), float(row["sepal_width"]),
                           float(row["petal_length"]), float(row["petal_width"])])
            y_rows.append(name_to_id[row["species"]])
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)


def stratified_split(X, y, test_fraction=0.3, seed=0):
    """Seeded, class-proportional train/test split."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for k in np.unique(y):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        n_test_k = int(round(len(idx_k) * test_fraction))
        test_idx.extend(idx_k[:n_test_k])
        train_idx.extend(idx_k[n_test_k:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(Xtr, Xte):
    """Mean/std from the TRAINING set only, applied to both sets."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def add_bias(X):
    """Prepend a column of ones (the bias feature)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def one_hot(y, K):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, K=3):
    """Rows = true class, columns = predicted class."""
    C = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C


def multiclass_log_loss(P, y, eps=1e-12):
    """Mean negative log-probability assigned to the true class."""
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0))))
