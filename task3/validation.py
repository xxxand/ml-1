"""分层 K 折交叉验证与分类模型参数选择。"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from task3.metrics import accuracy_score, classification_metrics


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按类别分层的 K 折索引，保证每折各类别比例接近。"""
    y = np.asarray(y, dtype=int)
    if n_splits < 2 or n_splits > len(y):
        raise ValueError("n_splits 取值不合法")

    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        class_indices = np.where(y == cls)[0]
        class_indices = rng.permutation(class_indices)
        for fold_index, part in enumerate(np.array_split(class_indices, n_splits)):
            folds[fold_index].extend(part.tolist())

    splits = []
    all_indices = np.arange(len(y))
    for fold_index in range(n_splits):
        val_idx = np.array(sorted(folds[fold_index]), dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx)
        splits.append((train_idx, val_idx))
    return splits


def _param_grid(**parameters) -> list[dict[str, object]]:
    """把 {参数名: 候选值列表} 展开成参数组合列表。"""
    names = list(parameters)
    values = [parameters[name] for name in names]
    return [dict(zip(names, combination)) for combination in product(*values)]


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: list[dict[str, object]],
    model_factory,
    class_names: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """对每个参数组合做分层 K 折交叉验证。"""
    splits = stratified_kfold_indices(y, n_splits=n_splits, seed=seed)
    rows = []

    for params in param_grid:
        accuracies = []
        f1_scores = []
        for train_idx, val_idx in splits:
            model = model_factory(**params).fit(X[train_idx], y[train_idx])
            prediction = model.predict(X[val_idx])
            accuracies.append(accuracy_score(y[val_idx], prediction))
            f1_scores.append(
                classification_metrics(
                    y[val_idx], prediction, class_names
                )["f1_macro"]
            )

        rows.append(
            {
                **params,
                "cv_accuracy_mean": float(np.mean(accuracies)),
                "cv_accuracy_std": float(np.std(accuracies, ddof=0)),
                "cv_f1_macro_mean": float(np.mean(f1_scores)),
            }
        )

    return pd.DataFrame(rows)


def make_param_grid(parameters: dict[str, list[object]]) -> list[dict[str, object]]:
    """供 main.py 调用的参数网格构造器。"""
    return _param_grid(**parameters)


def select_best_row(cv_result: pd.DataFrame) -> pd.Series:
    """按 CV 准确率选择最优参数；准确率相同时比较 macro F1。"""
    ordered = cv_result.sort_values(
        ["cv_accuracy_mean", "cv_f1_macro_mean"],
        ascending=False,
        kind="stable",
    )
    return ordered.iloc[0]
