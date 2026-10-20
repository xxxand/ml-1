"""任务二图表绘制。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task2.metrics import evaluate

COLORS = {"Ridge": "#35618f", "Lasso": "#c26a1b"}


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
    """测试集真实值与预测值散点图。"""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=True, sharey=True)

    values = [y_true, *predictions.values()]
    hi = max(float(v.max()) for v in values) * 1.03
    lo = 0.0

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        metrics = evaluate(y_true, y_pred)
        ax.scatter(
            y_true,
            y_pred,
            s=8,
            alpha=0.3,
            color=COLORS.get(name, "#555555"),
            edgecolors="none",
        )
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual median house value (USD)")
        ax.set_ylabel("Predicted median house value (USD)")
        ax.set_title(f"{name} on test set")
        ax.text(
            0.04,
            0.96,
            f"RMSE = {metrics['RMSE'] / 1000:,.1f}k\n"
            f"MAE  = {metrics['MAE'] / 1000:,.1f}k\n"
            f"$R^2$   = {metrics['R2']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
        )

    _save(fig, path)


def plot_residuals(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    """测试集残差图。"""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        residual = y_true - y_pred
        rmse = float(np.sqrt(np.mean(residual**2)))
        ax.scatter(
            y_pred,
            residual,
            s=8,
            alpha=0.3,
            color=COLORS.get(name, "#555555"),
            edgecolors="none",
        )
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        ax.axhline(rmse, color="#999999", linestyle=":", linewidth=1.0)
        ax.axhline(-rmse, color="#999999", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Predicted median house value (USD)")
        ax.set_ylabel("Residual (USD)")
        ax.set_title(f"{name} residuals")

    _save(fig, path)


def plot_alpha_cv(
    cv_results: dict[str, pd.DataFrame],
    best_alphas: dict[str, float],
    path: Path,
) -> None:
    """交叉验证 RMSE 随正则参数的变化。"""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    for ax, (name, cv) in zip(axes, cv_results.items()):
        ax.errorbar(
            cv["alpha"],
            cv["cv_rmse_mean"] / 1000.0,
            yerr=cv["cv_rmse_std"] / 1000.0,
            fmt="o-",
            markersize=3.5,
            linewidth=1.3,
            capsize=2.5,
            color=COLORS[name],
            ecolor="#c9b8a8",
            label="5-fold CV RMSE",
        )
        ax.axvline(
            best_alphas[name],
            color="#a93226",
            linestyle="--",
            linewidth=1.2,
            label=f"selected $\\alpha$ = {best_alphas[name]:.3g}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Regularization strength $\\alpha$")
        ax.set_ylabel("CV RMSE (thousand USD)")
        ax.set_title(f"{name}: parameter selection")
        ax.legend(loc="upper left", fontsize=8)

    _save(fig, path)


def plot_alpha_generalization(
    alpha_metrics,
    best_alphas: dict[str, float],
    path: Path,
) -> None:
    """训练/测试 RMSE 随正则参数的变化（测试曲线仅用于事后分析）。"""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)

    for ax, name in zip(axes, ["Ridge", "Lasso"]):
        subset = alpha_metrics[alpha_metrics["model"] == name]
        ax.plot(
            subset["alpha"],
            subset["train_rmse"] / 1000.0,
            "o-",
            markersize=3.2,
            linewidth=1.2,
            color=COLORS[name],
            label="Training RMSE",
        )
        ax.plot(
            subset["alpha"],
            subset["test_rmse"] / 1000.0,
            "s--",
            markersize=3.2,
            linewidth=1.2,
            color="#7f8c8d",
            label="Test RMSE",
        )
        ax.axvline(best_alphas[name], color="#a93226", linestyle=":", linewidth=1.1)
        ax.set_xscale("log")
        ax.set_xlabel("Regularization strength $\\alpha$")
        ax.set_ylabel("RMSE (thousand USD)")
        ax.set_title(f"{name}: training vs test")
        ax.legend(loc="upper left", fontsize=8)

    _save(fig, path)


def plot_coefficient_paths(
    coef_paths: dict[str, np.ndarray],
    alphas: dict[str, np.ndarray],
    feature_names: list[str],
    best_alphas: dict[str, float],
    path: Path,
) -> None:
    """岭回归和套索回归的系数路径。"""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    cmap = plt.get_cmap("tab20")
    handles = []

    for ax, name in zip(axes, ["Ridge", "Lasso"]):
        for j, feature in enumerate(feature_names):
            line, = ax.plot(
                alphas[name],
                coef_paths[name][:, j],
                color=cmap(j % 20),
                linewidth=1.2,
            )
            if name == "Ridge":
                handles.append(line)
        ax.axvline(
            best_alphas[name],
            color="#a93226",
            linestyle="--",
            linewidth=1.2,
        )
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Regularization strength $\\alpha$")
        ax.set_ylabel("Coefficient (standardized target)")
        ax.set_title(f"{name} coefficient paths")

    fig.legend(
        handles,
        feature_names,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        fontsize=8,
        title="Feature",
        title_fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_coefficient_comparison(
    ridge_coef: np.ndarray,
    lasso_coef: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    """选定参数下岭回归与套索回归的系数对比。"""
    _apply_style()
    order = np.argsort(np.maximum(np.abs(ridge_coef), np.abs(lasso_coef)))
    names = [feature_names[i] for i in order]
    ridge = ridge_coef[order]
    lasso = lasso_coef[order]

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    y = np.arange(len(names))
    height = 0.38

    ax.barh(y - height / 2, ridge, height, color=COLORS["Ridge"], label="Ridge")
    ax.barh(y + height / 2, lasso, height, color=COLORS["Lasso"], label="Lasso")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Coefficient (standardized target per std. feature)")
    ax.set_title("Selected coefficients: Ridge vs Lasso")
    ax.legend(loc="lower right")

    _save(fig, path)
