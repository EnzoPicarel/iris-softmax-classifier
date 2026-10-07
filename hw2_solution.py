"""Homework 2: logistic regression and softmax on the Iris data set."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from iris_utils import (
    accuracy,
    add_bias,
    confusion_matrix,
    load_iris_csv,
    one_hot,
    stratified_split,
    standardize,
    CLASS_NAMES,
    FEATURE_NAMES,
)


def sigmoid(z):
    """Numerically stable logistic sigmoid."""
    z = np.clip(np.asarray(z, dtype=float), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_loss(theta, Xb, y, lam):
    """Mean binary cross-entropy with L2 penalty (excluding the bias)."""
    probabilities = sigmoid(Xb @ theta)
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    data_loss = -np.mean(
        y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities)
    )
    regularization = 0.5 * lam * np.sum(theta[1:] ** 2)
    return float(data_loss + regularization)


def binary_grad(theta, Xb, y, lam):
    """Gradient of binary_loss, with no regularization on the bias."""
    probabilities = sigmoid(Xb @ theta)
    grad = Xb.T @ (probabilities - y) / len(y)
    grad[1:] += lam * theta[1:]
    return grad


class BinaryLogisticRegression:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, eta=0.5, lam=0.01, n_iters=2000):
        self.eta = eta
        self.lam = lam
        self.n_iters = n_iters
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        Xb = add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        self.theta = np.zeros(Xb.shape[1], dtype=float)
        self.loss_history = []

        for _ in range(self.n_iters):
            self.theta -= self.eta * binary_grad(self.theta, Xb, y, self.lam)
            self.loss_history.append(binary_loss(self.theta, Xb, y, self.lam))
        return self

    def predict_proba(self, X):
        if self.theta is None:
            raise RuntimeError("Call fit before predict_proba.")
        Xb = add_bias(np.asarray(X, dtype=float))
        return sigmoid(Xb @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class TwoStageClassifier:
    """Three-class classifier formed from two conditional binary models."""

    def __init__(self, eta=0.5, lam=0.01, n_iters=2000):
        self.eta = eta
        self.lam = lam
        self.n_iters = n_iters
        self.setosa_model = None
        self.non_setosa_model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # Stage 1: class 0 is setosa; classes 1 and 2 are the rest.
        self.setosa_model = BinaryLogisticRegression(
            eta=self.eta, lam=self.lam, n_iters=self.n_iters
        )
        self.setosa_model.fit(X, (y == 0).astype(int))

        # Stage 2: among non-setosa samples, class 0 is versicolor,
        # and class 1 is virginica.
        non_setosa = y != 0
        self.non_setosa_model = BinaryLogisticRegression(
            eta=self.eta, lam=self.lam, n_iters=self.n_iters
        )
        self.non_setosa_model.fit(X[non_setosa], (y[non_setosa] == 2).astype(int))
        return self

    def predict_proba(self, X):
        if self.setosa_model is None or self.non_setosa_model is None:
            raise RuntimeError("Call fit before predict_proba.")

        q1 = self.setosa_model.predict_proba(X)
        q2 = self.non_setosa_model.predict_proba(X)
        return np.column_stack((q1, (1.0 - q1) * (1.0 - q2), (1.0 - q1) * q2))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def softmax(S):
    """Compute row-wise softmax with max subtraction for numerical stability."""
    S = np.asarray(S, dtype=float)
    shifted = S - np.max(S, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def softmax_loss(W, Xb, Y, lam):
    """Mean multiclass cross-entropy with L2 penalty (excluding bias row)."""
    probabilities = softmax(Xb @ W)
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    data_loss = -np.sum(Y * np.log(probabilities)) / Xb.shape[0]
    regularization = 0.5 * lam * np.sum(W[1:, :] ** 2)
    return float(data_loss + regularization)


def softmax_grad(W, Xb, Y, lam):
    """Gradient of softmax_loss, with no regularization on the bias row."""
    probabilities = softmax(Xb @ W)
    grad = Xb.T @ (probabilities - Y) / Xb.shape[0]
    grad[1:, :] += lam * W[1:, :]
    return grad


class SoftmaxRegression:
    """Multinomial logistic regression trained with batch gradient descent."""

    def __init__(self, eta=0.5, lam=0.01, n_iters=3000, n_classes=3):
        self.eta = eta
        self.lam = lam
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.W = None
        self.loss_history = []

    def fit(self, X, y):
        Xb = add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=int)
        Y = one_hot(y, self.n_classes)
        self.W = np.zeros((Xb.shape[1], self.n_classes), dtype=float)
        self.loss_history = []

        for _ in range(self.n_iters):
            self.W -= self.eta * softmax_grad(self.W, Xb, Y, self.lam)
            self.loss_history.append(softmax_loss(self.W, Xb, Y, self.lam))
        return self

    def predict_proba(self, X):
        if self.W is None:
            raise RuntimeError("Call fit before predict_proba.")
        Xb = add_bias(np.asarray(X, dtype=float))
        return softmax(Xb @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def run_model_comparison(Xtr, Xte, ytr, yte):
    """Train both required multiclass models and report their test results."""
    models = {
        "Two-stage": TwoStageClassifier(eta=0.5, lam=0.01, n_iters=2000),
        "Softmax": SoftmaxRegression(eta=0.5, lam=0.01, n_iters=3000),
    }

    for name, model in models.items():
        model.fit(Xtr, ytr)
        train_accuracy = accuracy(ytr, model.predict(Xtr))
        test_predictions = model.predict(Xte)
        test_accuracy = accuracy(yte, test_predictions)
        test_cm = confusion_matrix(yte, test_predictions, K=3)
        print(f"\n{name} classifier:")
        print(f"  Train accuracy: {train_accuracy:.3f}")
        print(f"  Test accuracy:  {test_accuracy:.3f}")
        print("  Test confusion matrix (rows=true, columns=predicted):")
        print(test_cm)

        probabilities = model.predict_proba(Xte)
        print(
            "  Maximum probability-sum error:",
            f"{np.max(np.abs(probabilities.sum(axis=1) - 1.0)):.2e}",
        )

    return models


def plot_petal_scatter(Xtr, ytr):
    """Plot training samples by petal length and width, colored by species."""
    colors = ("tab:blue", "tab:orange", "tab:green")
    for class_id, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = ytr == class_id
        plt.scatter(
            Xtr[mask, 2],
            Xtr[mask, 3],
            color=color,
            label=class_name,
            alpha=0.8,
        )
    plt.xlabel(f"Standardized {FEATURE_NAMES[2]}")
    plt.ylabel(f"Standardized {FEATURE_NAMES[3]}")
    plt.title("Training set: petal measurements by species")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.show()


def load_data():
    """Load Iris, make the required seed-0 split, and standardize from train data."""
    data_path = Path(__file__).resolve().parent / "iris.csv"
    X, y = load_iris_csv(data_path)
    Xtr_raw, Xte_raw, ytr, yte = stratified_split(
        X, y, test_fraction=0.3, seed=0
    )
    Xtr, Xte, mu, sd = standardize(Xtr_raw, Xte_raw)
    return Xtr, Xte, ytr, yte, mu, sd


def run_setosa_experiment(Xtr, Xte, ytr, yte, show_plot=True):
    """Train and report the required setosa-versus-rest binary experiment."""
    ytr_binary = (ytr == 0).astype(int)
    yte_binary = (yte == 0).astype(int)

    model = BinaryLogisticRegression(eta=0.5, lam=0.1, n_iters=2000)
    model.fit(Xtr, ytr_binary)

    train_accuracy = accuracy(ytr_binary, model.predict(Xtr))
    test_accuracy = accuracy(yte_binary, model.predict(Xte))
    print(f"Setosa vs rest train accuracy: {train_accuracy:.3f}")
    print(f"Setosa vs rest test accuracy:  {test_accuracy:.3f}")

    if show_plot:
        iterations = np.arange(1, len(model.loss_history) + 1)
        plt.plot(iterations, model.loss_history, label="Training loss")
        plt.xlabel("Iteration")
        plt.ylabel("Binary cross-entropy loss")
        plt.title("Setosa vs rest: training loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model, train_accuracy, test_accuracy


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, mu, sd = load_data()
    print(f"Train: X={Xtr.shape}, y={ytr.shape}")
    print(f"Test:  X={Xte.shape}, y={yte.shape}")
    print("Training counts by class:", np.bincount(ytr, minlength=3))
    print("Test counts by class:    ", np.bincount(yte, minlength=3))
    plot_petal_scatter(Xtr, ytr)
    run_setosa_experiment(Xtr, Xte, ytr, yte)
    run_model_comparison(Xtr, Xte, ytr, yte)
