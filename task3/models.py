"""手动实现的 Logistic 回归、Gaussian Naive Bayes 和 Softmax 回归。"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


class LogisticRegression:
    """一对多（One-vs-Rest）Logistic 回归。

    对每个类别训练一个二分类器：
        p(y=c|x) = sigmoid(x^T w_c + b_c)
    目标函数为二分类交叉熵 + L2 惩罚，使用全批量梯度下降求解。
    多分类预测时取各类别得分最大的一个。
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        l2: float = 0.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.n_iter_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        all_coef = []
        all_intercept = []
        all_iterations = []

        for cls in self.classes_:
            y_binary = (y == cls).astype(float)
            weights = np.zeros(n_features)
            intercept = 0.0
            iterations = 0

            for iteration in range(self.max_iter):
                probabilities = _sigmoid(X @ weights + intercept)
                errors = probabilities - y_binary
                grad_w = X.T @ errors / n_samples + self.l2 * weights
                grad_b = float(errors.mean())

                weights -= self.learning_rate * grad_w
                intercept -= self.learning_rate * grad_b
                iterations = iteration + 1

                if np.max(np.abs(grad_w)) < self.tol:
                    break

            all_coef.append(weights)
            all_intercept.append(intercept)
            all_iterations.append(iterations)

        self.coef_ = np.vstack(all_coef)
        self.intercept_ = np.asarray(all_intercept)
        self.n_iter_ = np.asarray(all_iterations)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练")
        return np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """每个类别单独的 sigmoid 得分，并归一化为行和为 1 的概率。

        One-vs-Rest 原始得分不要求和为 1；归一化不改变 argmax，
        但方便把多类得分当作概率用于 ROC/AUC。
        """
        scores = _sigmoid(self.decision_function(X))
        row_sum = scores.sum(axis=1, keepdims=True)
        return scores / np.where(row_sum == 0.0, 1.0, row_sum)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("模型尚未训练")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class GaussianNaiveBayes:
    """高斯朴素贝叶斯。

    假设每个类别内各特征服从正态分布且特征之间条件独立：
        p(x|y=c) = Π_j N(x_j | mean_cj, var_cj)
    预测时比较 log p(y=c) + Σ_j log p(x_j|y=c)。
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = float(var_smoothing)
        self.classes_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.epsilon_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.unique(y)

        # 与 sklearn 类似：平滑量相对于特征方差的最大值。
        max_var = float(np.var(X, axis=0).max())
        self.epsilon_ = max(self.var_smoothing * max_var, 1e-12)

        priors = []
        means = []
        variances = []
        for cls in self.classes_:
            X_class = X[y == cls]
            priors.append(len(X_class) / len(X))
            means.append(X_class.mean(axis=0))
            variances.append(X_class.var(axis=0) + self.epsilon_)

        self.priors_ = np.asarray(priors)
        self.means_ = np.vstack(means)
        self.variances_ = np.vstack(variances)
        return self

    def _log_joint(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.variances_ is None or self.priors_ is None:
            raise RuntimeError("模型尚未训练")

        log_prior = np.log(self.priors_)
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi * self.variances_)[None, :, :]
            + (X[:, None, :] - self.means_[None, :, :]) ** 2
            / self.variances_[None, :, :],
            axis=2,
        )
        return log_likelihood + log_prior[None, :]

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        log_joint = self._log_joint(np.asarray(X, dtype=float))
        log_joint -= log_joint.max(axis=1, keepdims=True)
        return log_joint - np.log(np.exp(log_joint).sum(axis=1, keepdims=True))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("模型尚未训练")
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class SoftmaxRegression:
    """多分类 Softmax 回归（多项 Logistic 回归）。

    目标函数为多分类交叉熵 + L2 惩罚，使用全批量梯度下降求解。
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        l2: float = 0.0,
        max_iter: int = 3000,
        tol: float = 1e-6,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        one_hot = np.zeros((n_samples, n_classes))
        for index, cls in enumerate(self.classes_):
            one_hot[y == cls, index] = 1.0

        weights = np.zeros((n_features, n_classes))
        intercept = np.zeros(n_classes)

        for iteration in range(self.max_iter):
            probabilities = _softmax(X @ weights + intercept)
            errors = probabilities - one_hot
            grad_w = X.T @ errors / n_samples + self.l2 * weights
            grad_b = errors.mean(axis=0)

            weights -= self.learning_rate * grad_w
            intercept -= self.learning_rate * grad_b
            self.n_iter_ = iteration + 1

            if np.max(np.abs(grad_w)) < self.tol:
                break

        self.coef_ = weights.T
        self.intercept_ = intercept
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练")
        logits = np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("模型尚未训练")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
