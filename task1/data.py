"""数据读取、缺失值处理、留出划分与标准化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "MEDV"


@dataclass
class Dataset:
    """预处理后的训练集与测试集。"""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    missing_before: pd.Series
    impute_values: pd.Series


def load_raw(path: str) -> pd.DataFrame:
    """读取 Boston 原始数据。"""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"目标列 {TARGET!r} 不存在")
    if df[TARGET].isna().any():
        raise ValueError("目标列 MEDV 存在缺失值，无法直接建模")
    return df


def split_holdout(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """留出法划分训练集/测试集。

    默认按 MEDV 分位数分层抽样，避免随机划分造成目标分布偏斜；
    固定随机种子，结果可复现。
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 必须位于 (0, 1) 之间")

    rng = np.random.default_rng(seed)

    if stratify:
        bins = pd.qcut(df[TARGET], q=n_bins, labels=False, duplicates="drop")
        bin_frame = pd.DataFrame({"position": np.arange(len(df)), "bin": bins})
        test_parts = []
        for _, group in bin_frame.groupby("bin", sort=True):
            indices = rng.permutation(group["position"].to_numpy())
            n_take = int(round(len(indices) * test_size))
            n_take = min(max(n_take, 1), len(indices) - 1)
            test_parts.append(indices[:n_take])
        test_idx = np.concatenate(test_parts)
    else:
        indices = rng.permutation(len(df))
        n_test = int(round(len(df) * test_size))
        test_idx = indices[:n_test]

    train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def preprocess(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dataset:
    """用训练集统计量做中位数填补与标准化，避免测试集信息泄漏。"""
    feature_names = [c for c in train_df.columns if c != TARGET]
    missing_before = train_df[feature_names].isna().sum()

    # 中位数只由训练集计算；若某列全缺失则退化为 0。
    impute_values = train_df[feature_names].median().fillna(0.0)
    X_train_df = train_df[feature_names].fillna(impute_values)
    X_test_df = test_df[feature_names].fillna(impute_values)

    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=train_df[TARGET].to_numpy(dtype=float),
        y_test=test_df[TARGET].to_numpy(dtype=float),
        feature_names=feature_names,
        missing_before=missing_before,
        impute_values=impute_values,
    )
