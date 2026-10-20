"""K 折交叉验证与正则参数选择。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from task2.metrics import mean_absolute_error, mean_squared_error
from task2.models import LassoRegression, RidgeRegression


def kfold_indices(
    n_samples: int, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """返回 K 折的 (训练索引, 验证索引)。"""
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError("n_splits 取值不合法")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, n_splits)

    splits = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate(
            [folds[j] for j in range(n_splits) if j != i]
        )
        splits.append((train_idx, val_idx))
    return splits


def _cross_validate(
    X: np.ndarray,
    y_scaled: np.ndarray,
    alphas: np.ndarray,
    model_factory,
    target_std: float,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """通用 K 折交叉验证；返回的误差已换算回美元。"""
    splits = kfold_indices(len(X), n_splits=n_splits, seed=seed)
    rows = []

    for alpha in alphas:
        fold_mse = []
        fold_mae = []
        for train_idx, val_idx in splits:
            model = model_factory(float(alpha)).fit(X[train_idx], y_scaled[train_idx])
            prediction = model.predict(X[val_idx])
            fold_mse.append(mean_squared_error(y_scaled[val_idx], prediction))
            fold_mae.append(mean_absolute_error(y_scaled[val_idx], prediction))

        fold_mse = np.asarray(fold_mse) * target_std**2
        fold_mae = np.asarray(fold_mae) * target_std
        fold_rmse = np.sqrt(fold_mse)

        rows.append(
            {
                "alpha": float(alpha),
                "cv_mse_mean": float(fold_mse.mean()),
                "cv_mse_std": float(fold_mse.std(ddof=0)),
                "cv_mae_mean": float(fold_mae.mean()),
                "cv_rmse_mean": float(fold_rmse.mean()),
                "cv_rmse_std": float(fold_rmse.std(ddof=0)),
            }
        )

    return pd.DataFrame(rows)


def cross_validate_ridge(
    X: np.ndarray,
    y_scaled: np.ndarray,
    alphas: np.ndarray,
    target_std: float,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """岭回归的 5 折交叉验证。"""
    return _cross_validate(
        X,
        y_scaled,
        alphas,
        lambda alpha: RidgeRegression(alpha=alpha),
        target_std,
        n_splits=n_splits,
        seed=seed,
    )


def cross_validate_lasso(
    X: np.ndarray,
    y_scaled: np.ndarray,
    alphas: np.ndarray,
    target_std: float,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """套索回归的 5 折交叉验证。"""
    return _cross_validate(
        X,
        y_scaled,
        alphas,
        lambda alpha: LassoRegression(alpha=alpha),
        target_std,
        n_splits=n_splits,
        seed=seed,
    )
