"""分类任务：Iris 数据读取、分层划分与标准化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "Species"
FEATURES = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]


@dataclass
class Dataset:
    """预处理后的训练集与测试集。

    X_train / X_test 是标准化后的特征；X_train_raw / X_test_raw 保留原始
    数值，用于画图（尤其是花瓣长度的二维决策区域）。
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    missing_before: pd.Series


def load_raw(path: str) -> pd.DataFrame:
    """读取 Iris 数据，并去掉 CSV 中多余的行号列。"""
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    if TARGET not in df.columns:
        raise ValueError(f"目标列 {TARGET!r} 不存在")
    if df[TARGET].isna().any():
        raise ValueError("目标列存在缺失值")
    return df


def split_holdout(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按类别分层留出训练集/测试集。"""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 必须位于 (0, 1) 之间")

    rng = np.random.default_rng(seed)
    test_parts = []
    for _, group in df.groupby(TARGET, sort=True):
        indices = rng.permutation(group.index.to_numpy())
        n_take = int(round(len(indices) * test_size))
        n_take = min(max(n_take, 1), len(indices) - 1)
        test_parts.append(indices[:n_take])

    test_idx = np.concatenate(test_parts)
    train_idx = np.setdiff1d(df.index.to_numpy(), test_idx)
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dataset:
    """缺失值填补、类别编码和特征标准化。"""
    missing_before = train_df[FEATURES].isna().sum()
    impute_values = train_df[FEATURES].median().fillna(0.0)

    X_train_raw_df = train_df[FEATURES].fillna(impute_values)
    X_test_raw_df = test_df[FEATURES].fillna(impute_values)
    X_train_raw = X_train_raw_df.to_numpy(dtype=float)
    X_test_raw = X_test_raw_df.to_numpy(dtype=float)

    feature_mean = X_train_raw.mean(axis=0)
    feature_std = X_train_raw.std(axis=0, ddof=0)
    feature_std[feature_std == 0] = 1.0

    X_train = (X_train_raw - feature_mean) / feature_std
    X_test = (X_test_raw - feature_mean) / feature_std

    class_names = sorted(train_df[TARGET].unique())
    if not set(test_df[TARGET].unique()).issubset(set(class_names)):
        raise ValueError("测试集出现了训练集没有的类别")
    y_train = np.array([class_names.index(v) for v in train_df[TARGET]], dtype=int)
    y_test = np.array([class_names.index(v) for v in test_df[TARGET]], dtype=int)

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        feature_names=list(FEATURES),
        class_names=class_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        missing_before=missing_before,
    )
