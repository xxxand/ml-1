"""手动实现的线性回归与岭回归（仅使用 NumPy 线性代数）。"""

from __future__ import annotations

import numpy as np


class LinearRegression:
    """普通最小二乘，通过正规方程求解。"""

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(X)), X])
        else:
            X_design = X

        # 正规方程 (X^T X) w = X^T y
        gram = X_design.T @ X_design
        rhs = X_design.T @ y
        weights = np.linalg.solve(gram, rhs)

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class RidgeRegression:
    """岭回归闭式解；截距不参与正则化。"""

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

        # (X_c^T X_c + alpha I) w = X_c^T y_c
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
