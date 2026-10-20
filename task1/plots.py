"""实验图表绘制。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from task1.metrics import evaluate

COLORS = {"OLS": "#35618f", "Ridge": "#c26a1b"}


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


def plot_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    """测试集真实值-预测值散点图。"""
    _apply_style()
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    values = [y_true, *predictions.values()]
    lo = min(float(v.min()) for v in values) - 1.5
    hi = max(float(v.max()) for v in values) + 1.5

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        metrics = evaluate(y_true, y_pred)
        ax.scatter(
            y_true,
            y_pred,
            s=22,
            alpha=0.65,
            color=COLORS.get(name, "#555555"),
            edgecolors="white",
            linewidths=0.4,
        )
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual MEDV ($1000s)")
        ax.set_ylabel("Predicted MEDV ($1000s)")
        ax.set_title(f"{name} on test set")
        ax.text(
            0.04,
            0.96,
            f"MSE  = {metrics['MSE']:.2f}\n"
            f"MAE  = {metrics['MAE']:.2f}\n"
            f"RMSE = {metrics['RMSE']:.2f}\n"
            f"$R^2$   = {metrics['R2']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
        )

    _save(fig, path)


def plot_residuals(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    """测试集残差图。"""
    _apply_style()
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        residual = y_true - y_pred
        rmse = float(np.sqrt(np.mean(residual**2)))
        ax.scatter(
            y_pred,
            residual,
            s=20,
            alpha=0.65,
            color=COLORS.get(name, "#555555"),
            edgecolors="white",
            linewidths=0.4,
        )
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        ax.axhline(rmse, color="#999999", linestyle=":", linewidth=1.0)
        ax.axhline(-rmse, color="#999999", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Predicted MEDV ($1000s)")
        ax.set_ylabel("Residual ($1000s)")
        ax.set_title(f"{name} residuals")

    _save(fig, path)


def plot_alpha_selection(cv_result, best_alpha: float, path: Path) -> None:
    """交叉验证误差随正则参数的变化。"""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.errorbar(
        cv_result["alpha"],
        cv_result["cv_mse_mean"],
        yerr=cv_result["cv_mse_std"],
        fmt="o-",
        markersize=4,
        linewidth=1.4,
        capsize=3,
        color=COLORS["Ridge"],
        ecolor="#d9b08c",
        label="5-fold CV MSE (mean $\\pm$ std)",
    )
    ax.axvline(
        best_alpha,
        color="#a93226",
        linestyle="--",
        linewidth=1.2,
        label=f"selected $\\alpha$ = {best_alpha:.3g}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength $\\alpha$")
    ax.set_ylabel("CV MSE")
    ax.set_title("Ridge: parameter selection on the training set")
    ax.legend(loc="upper left")

    _save(fig, path)


def plot_alpha_generalization(
    alphas: np.ndarray,
    train_mse: np.ndarray,
    test_mse: np.ndarray,
    best_alpha: float,
    path: Path,
) -> None:
    """训练/测试误差随正则参数的变化（测试曲线仅用于分析）。"""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.plot(
        alphas,
        train_mse,
        "o-",
        markersize=3.5,
        linewidth=1.4,
        color=COLORS["OLS"],
        label="Training MSE",
    )
    ax.plot(
        alphas,
        test_mse,
        "s-",
        markersize=3.5,
        linewidth=1.4,
        color=COLORS["Ridge"],
        label="Test MSE",
    )
    ax.axvline(
        best_alpha,
        color="#a93226",
        linestyle="--",
        linewidth=1.2,
        label=f"selected $\\alpha$ = {best_alpha:.3g}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength $\\alpha$")
    ax.set_ylabel("MSE")
    ax.set_title("Ridge: training vs test error")
    ax.legend(loc="upper left")

    _save(fig, path)


def plot_coefficient_paths(
    alphas: np.ndarray,
    coef_path: np.ndarray,
    feature_names: list[str],
    best_alpha: float,
    path: Path,
) -> None:
    """岭回归系数路径，展示正则化的收缩作用。"""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    cmap = plt.get_cmap("tab20")

    for j, name in enumerate(feature_names):
        ax.plot(
            alphas,
            coef_path[:, j],
            color=cmap(j % 20),
            linewidth=1.4,
            label=name,
        )

    ax.axvline(
        best_alpha,
        color="#a93226",
        linestyle="--",
        linewidth=1.2,
        label=f"selected $\\alpha$ = {best_alpha:.3g}",
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength $\\alpha$")
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("Ridge coefficient paths")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title="Feature",
        title_fontsize=8,
    )

    _save(fig, path)


def plot_coefficient_comparison(
    ols_coef: np.ndarray,
    ridge_coef: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    """OLS 与岭回归（选定 alpha）的系数对比。"""
    _apply_style()
    order = np.argsort(np.abs(ols_coef))
    names = [feature_names[i] for i in order]
    ols = ols_coef[order]
    ridge = ridge_coef[order]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    y = np.arange(len(names))
    height = 0.38

    ax.barh(y - height / 2, ols, height, color=COLORS["OLS"], label="OLS")
    ax.barh(y + height / 2, ridge, height, color=COLORS["Ridge"], label="Ridge")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("Coefficient comparison on the training set")
    ax.legend(loc="lower right")

    _save(fig, path)
