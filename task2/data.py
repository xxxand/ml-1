"""California housing 数据读取、特征构造、划分与标准化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "median_house_value"
CATEGORICAL = ["ocean_proximity"]


@dataclass
class Dataset:
    """预处理后的训练集与测试集。

    注意：y_train / y_test 是原始美元价格，y_train_scaled / y_test_scaled
    是用于训练的目标值（按训练集均值、标准差标准化）。这样岭回归和套索
    回归的正则参数处在相近的数值尺度上，预测后再还原成美元。
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_scaled: np.ndarray
    y_test_scaled: np.ndarray
    feature_names: list[str]
    target_mean: float
    target_std: float
    missing_before: pd.Series
    impute_values: pd.Series


def load_raw(path: str) -> pd.DataFrame:
    """读取 California housing 原始数据。"""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"目标列 {TARGET!r} 不存在")
    if df[TARGET].isna().any():
        raise ValueError(f"目标列 {TARGET} 存在缺失值，无法直接建模")
    return df


def split_holdout(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """留出法划分训练集/测试集。

    默认按目标值分位数分层，保证训练集和测试集的房价分布接近。
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


def _safe_divide(numerator, denominator) -> np.ndarray:
    """逐元素安全相除，分母为 0 时结果为 0。"""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _add_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    """增加三个常见的房屋结构比例特征。"""
    frame = frame.copy()
    frame["rooms_per_household"] = _safe_divide(
        frame["total_rooms"], frame["households"]
    )
    frame["bedrooms_per_room"] = _safe_divide(
        frame["total_bedrooms"], frame["total_rooms"]
    )
    frame["population_per_household"] = _safe_divide(
        frame["population"], frame["households"]
    )
    return frame


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dataset:
    """用训练集统计量完成缺失值填补、编码、特征构造和标准化。"""
    numeric_cols = [
        c for c in train_df.columns if c not in [TARGET, *CATEGORICAL]
    ]
    missing_before = train_df[numeric_cols].isna().sum()

    # 缺失值只用训练集中位数填补；本数据只有 total_bedrooms 缺失。
    impute_values = train_df[numeric_cols].median().fillna(0.0)
    train_num = _add_ratio_features(
        train_df[numeric_cols].fillna(impute_values)
    )
    test_num = _add_ratio_features(
        test_df[numeric_cols].fillna(impute_values)
    )

    # 类别特征做 one-hot；drop_first 避免与截距完全共线。
    train_cat = train_df[CATEGORICAL].fillna("UNKNOWN")
    test_cat = test_df[CATEGORICAL].fillna("UNKNOWN")
    train_dummies = pd.get_dummies(
        train_cat, prefix="ocean", drop_first=True, dtype=float
    )
    test_dummies = pd.get_dummies(
        test_cat, prefix="ocean", drop_first=True, dtype=float
    ).reindex(columns=train_dummies.columns, fill_value=0.0)

    X_train_df = pd.concat([train_num, train_dummies], axis=1)
    X_test_df = pd.concat([test_num, test_dummies], axis=1)
    feature_names = list(X_train_df.columns)

    mean = X_train_df.mean(axis=0)
    std = X_train_df.std(axis=0, ddof=0).replace(0.0, 1.0)
    X_train = ((X_train_df - mean) / std).to_numpy(dtype=float)
    X_test = ((X_test_df - mean) / std).to_numpy(dtype=float)

    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)
    target_mean = float(y_train.mean())
    target_std = float(y_train.std(ddof=0))
    if target_std == 0.0:
        target_std = 1.0

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_scaled=(y_train - target_mean) / target_std,
        y_test_scaled=(y_test - target_mean) / target_std,
        feature_names=feature_names,
        target_mean=target_mean,
        target_std=target_std,
        missing_before=missing_before,
        impute_values=impute_values,
    )
