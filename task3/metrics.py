"""分类评价指标：准确率、混淆矩阵、Precision/Recall/F1、ROC/AUC。"""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> list[dict[str, float | str]]:
    """返回每个类别的 precision / recall / F1 / support。"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rows = []
    for index, name in enumerate(class_names):
        tp = int(np.sum((y_true == index) & (y_pred == index)))
        fp = int(np.sum((y_true != index) & (y_pred == index)))
        fn = int(np.sum((y_true == index) & (y_pred != index)))
        support = int(np.sum(y_true == index))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        rows.append(
            {
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    """返回 accuracy 与 macro 平均的 precision / recall / F1。"""
    rows = per_class_metrics(y_true, y_pred, class_names)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": float(np.mean([row["precision"] for row in rows])),
        "recall_macro": float(np.mean([row["recall"] for row in rows])),
        "f1_macro": float(np.mean([row["f1"] for row in rows])),
    }


def roc_curve(y_true_binary: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """二分类 ROC 曲线，返回 FPR 和 TPR（包含 (0,0) 与 (1,1)）。"""
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    y_sorted = y_true_binary[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tps / tps[-1]])
    fpr = np.concatenate([[0.0], fps / fps[-1]])
    return fpr, tpr


def auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """用梯形法计算 ROC 曲线下面积。"""
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def macro_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, float], float]:
    """One-vs-Rest 多分类 AUC，返回每类 AUC 和宏平均。"""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    per_class = {}
    for index, name in enumerate(class_names):
        binary = (y_true == index).astype(int)
        fpr, tpr = roc_curve(binary, probabilities[:, index])
        per_class[name] = auc_score(fpr, tpr)
    return per_class, float(np.mean(list(per_class.values())))
