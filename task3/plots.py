"""任务三图表绘制。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from task3.metrics import accuracy_score, confusion_matrix

MODEL_COLORS = {
    "Logistic": "#35618f",
    "Gaussian NB": "#2e8b57",
    "Softmax": "#c26a1b",
}
CLASS_COLORS = ["#4c78a8", "#f58518", "#54a24b"]


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_decision_boundaries(models, data, path: Path) -> None:
    """在“花瓣长度-花瓣宽度”平面上画决策区域和测试样本。

    其余两个特征固定在训练集均值，因此这是四维决策边界的一个二维切片。
    """
    _apply_style()
    feature_x, feature_y = 2, 3
    x_min = min(data.X_train_raw[:, feature_x].min(), data.X_test_raw[:, feature_x].min()) - 0.2
    x_max = max(data.X_train_raw[:, feature_x].max(), data.X_test_raw[:, feature_x].max()) + 0.2
    y_min = min(data.X_train_raw[:, feature_y].min(), data.X_test_raw[:, feature_y].min()) - 0.2
    y_max = max(data.X_train_raw[:, feature_y].max(), data.X_test_raw[:, feature_y].max()) + 0.2

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240)
    )
    n_classes = len(data.class_names)
    cmap = ListedColormap(CLASS_COLORS[:n_classes])

    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (name, model) in zip(axes, models.items()):
        grid_raw = np.tile(data.feature_mean, (grid_x.size, 1))
        grid_raw[:, feature_x] = grid_x.ravel()
        grid_raw[:, feature_y] = grid_y.ravel()
        grid_scaled = (grid_raw - data.feature_mean) / data.feature_std
        predicted_grid = model.predict(grid_scaled).reshape(grid_x.shape)

        ax.contourf(
            grid_x,
            grid_y,
            predicted_grid,
            levels=np.arange(-0.5, n_classes, 1.0),
            colors=CLASS_COLORS[:n_classes],
            alpha=0.18,
        )
        ax.scatter(
            data.X_test_raw[:, feature_x],
            data.X_test_raw[:, feature_y],
            c=data.y_test,
            cmap=cmap,
            vmin=-0.5,
            vmax=n_classes - 0.5,
            edgecolors="#222222",
            linewidths=0.6,
            s=34,
            zorder=3,
        )
        accuracy = accuracy_score(data.y_test, model.predict(data.X_test))
        ax.set_title(f"{name} (test acc={accuracy:.3f})")
        ax.set_xlabel(data.feature_names[feature_x])
        ax.set_ylabel(data.feature_names[feature_y])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    handles = [
        Patch(facecolor=CLASS_COLORS[i], alpha=0.35, label=name)
        for i, name in enumerate(data.class_names)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=n_classes,
        bbox_to_anchor=(0.5, 0.065),
        title="True class",
        title_fontsize=8,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.01,
        "Decision regions: sepal features fixed at training mean.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    class_names: list[str],
    path: Path,
) -> None:
    """三个模型的测试集混淆矩阵。"""
    _apply_style()
    fig, axes = plt.subplots(
        1, len(predictions), figsize=(4.7 * len(predictions), 4.2)
    )
    axes = np.atleast_1d(axes)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_true, y_pred, len(class_names))
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for row in range(len(class_names)):
            for col in range(len(class_names)):
                value = matrix[row, col]
                color = "white" if value > matrix.max() * 0.55 else "#222222"
                ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=25, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        accuracy = accuracy_score(y_true, y_pred)
        ax.set_title(f"{name} (acc={accuracy:.3f})")
        ax.grid(False)

    _save(fig, path)


def plot_roc_curves(roc_data: dict[str, dict], path: Path) -> None:
    """三个模型的 One-vs-Rest ROC 曲线。"""
    _apply_style()
    fig, axes = plt.subplots(
        1, len(roc_data), figsize=(5.2 * len(roc_data), 4.6), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, (name, result) in zip(axes, roc_data.items()):
        for class_index, (class_name, (fpr, tpr, auc_value)) in enumerate(
            result["per_class"].items()
        ):
            ax.plot(
                fpr,
                tpr,
                color=CLASS_COLORS[class_index],
                linewidth=1.8,
                label=f"{class_name} (AUC={auc_value:.3f})",
            )
        ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{name} (macro AUC={result['macro_auc']:.3f})")
        ax.legend(loc="lower right", fontsize=8)

    _save(fig, path)


def plot_parameter_selection(cv_results: dict[str, pd.DataFrame], best_params: dict[str, dict], path: Path) -> None:
    """交叉验证准确率随参数的变化。"""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    for ax, name in zip(axes[:2], ["Logistic", "Softmax"]):
        cv = cv_results[name]
        l2_values = sorted(cv["l2"].unique())
        positions = np.arange(len(l2_values))
        for learning_rate in sorted(cv["learning_rate"].unique()):
            subset = cv[cv["learning_rate"] == learning_rate].copy()
            subset["l2_order"] = subset["l2"].map({v: i for i, v in enumerate(l2_values)})
            subset = subset.sort_values("l2_order")
            ax.errorbar(
                positions[subset["l2_order"].to_numpy(dtype=int)],
                subset["cv_accuracy_mean"],
                yerr=subset["cv_accuracy_std"],
                fmt="o-",
                markersize=4,
                linewidth=1.2,
                capsize=2.5,
                color=plt.get_cmap("viridis")(learning_rate / max(cv["learning_rate"])),
                label=f"lr={learning_rate:g}",
            )

        best = best_params[name]
        best_x = l2_values.index(best["l2"])
        best_row = cv[
            (cv["learning_rate"] == best["learning_rate"])
            & (cv["l2"] == best["l2"])
        ].iloc[0]
        ax.scatter(
            best_x,
            best_row["cv_accuracy_mean"],
            marker="*",
            s=180,
            color="#a93226",
            zorder=5,
            label="selected",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{value:g}" for value in l2_values])
        ax.set_xlabel("L2 regularization strength")
        ax.set_ylabel("CV accuracy")
        ax.set_title(f"{name}: parameter selection")
        ax.set_ylim(0.7, 1.01)
        ax.legend(fontsize=8, loc="lower left")

    # Gaussian NB 只有 var_smoothing 一个参数
    cv = cv_results["Gaussian NB"]
    ax = axes[2]
    ax.errorbar(
        cv["var_smoothing"],
        cv["cv_accuracy_mean"],
        yerr=cv["cv_accuracy_std"],
        fmt="o-",
        markersize=4,
        linewidth=1.2,
        capsize=2.5,
        color=MODEL_COLORS["Gaussian NB"],
    )
    best = best_params["Gaussian NB"]
    best_row = cv[cv["var_smoothing"] == best["var_smoothing"]].iloc[0]
    ax.scatter(
        best["var_smoothing"],
        best_row["cv_accuracy_mean"],
        marker="*",
        s=180,
        color="#a93226",
        zorder=5,
        label="selected",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Variance smoothing")
    ax.set_ylabel("CV accuracy")
    ax.set_title("Gaussian NB: parameter selection")
    ax.set_ylim(0.7, 1.01)
    ax.legend(fontsize=8, loc="lower left")

    _save(fig, path)


def plot_metrics_comparison(metrics: pd.DataFrame, path: Path) -> None:
    """三个模型的 Accuracy / Macro Precision / Recall / F1 对比。"""
    _apply_style()
    columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    models = metrics["model"].tolist()
    x = np.arange(len(models))
    width = 0.19

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for index, (column, label) in enumerate(zip(columns, labels)):
        values = metrics[column].to_numpy()
        bars = ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=label,
            color=plt.get_cmap("Set2")(index),
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Test-set classification metrics")
    ax.legend(ncol=4, loc="upper center", fontsize=8)

    _save(fig, path)
