"""手动实现的岭回归与套索回归（仅使用 NumPy 线性代数）。"""

from __future__ import annotations

import numpy as np


def soft_threshold(value: float, threshold: float) -> float:
    """软阈值算子 sign(value) * max(|value| - threshold, 0)。"""
    return float(np.sign(value) * max(abs(value) - threshold, 0.0))


class RidgeRegression:
    """岭回归。

    目标函数：1/2 * ||y - Xw - b||² + alpha/2 * ||w||²。
    对中心化后的数据求导可得：
        (Xc^T Xc + alpha I) w = Xc^T yc
    截距 b 不参与正则化。
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha 不能为负数")
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0
            X_centered = X
            y_centered = y

        gram = X_centered.T @ X_centered
        rhs = X_centered.T @ y_centered
        self.coef_ = np.linalg.solve(
            gram + self.alpha * np.eye(X.shape[1]), rhs
        )
        self.intercept_ = float(y_mean - X_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class LassoRegression:
    """套索回归，使用坐标下降求解。

    目标函数：1/2 * ||y - Xw - b||² + alpha * ||w||_1。
    对每个坐标 j，固定其他系数，闭式解为：
        w_j = soft_threshold(x_j^T (y - Xw + x_j w_j), alpha)
              / ||x_j||²
    截距 b 不参与正则化。特征在调用本类之前应已标准化。
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 2000,
        tol: float = 1e-6,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha 不能为负数")
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0
            X_centered = X
            y_centered = y

        n_features = X.shape[1]
        col_sq = np.sum(X_centered**2, axis=0)
        col_sq[col_sq == 0.0] = 1.0

        coefficients = np.zeros(n_features)
        residual = y_centered - X_centered @ coefficients

        for iteration in range(self.max_iter):
            max_delta = 0.0
            for j in range(n_features):
                column = X_centered[:, j]
                # residual = yc - Xc w，先加回第 j 个特征当前的贡献
                rho = column @ residual + coefficients[j] * col_sq[j]
                new_value = soft_threshold(rho, self.alpha) / col_sq[j]
                delta = new_value - coefficients[j]
                if delta != 0.0:
                    residual -= column * delta
                    coefficients[j] = new_value
                    max_delta = max(max_delta, abs(delta))

            if max_delta < self.tol:
                break

        self.coef_ = coefficients
        self.intercept_ = float(y_mean - X_mean @ coefficients)
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
