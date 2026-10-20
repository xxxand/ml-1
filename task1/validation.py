"""K 折划分与岭回归正则参数选择。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from task1.metrics import mean_squared_error
from task1.models import RidgeRegression


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
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def cross_validate_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """在训练集内部做 K 折交叉验证，返回每个 alpha 的验证 MSE。"""
    splits = kfold_indices(len(X), n_splits=n_splits, seed=seed)
    rows = []

    for alpha in alphas:
        fold_mse = []
        for train_idx, val_idx in splits:
            model = RidgeRegression(alpha=alpha).fit(X[train_idx], y[train_idx])
            fold_mse.append(mean_squared_error(y[val_idx], model.predict(X[val_idx])))
        rows.append(
            {
                "alpha": float(alpha),
                "cv_mse_mean": float(np.mean(fold_mse)),
                "cv_mse_std": float(np.std(fold_mse, ddof=0)),
            }
        )

    return pd.DataFrame(rows)
