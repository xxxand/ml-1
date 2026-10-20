"""回归任务二主流程：数据预处理、岭回归/套索回归、调参与可视化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from task2.data import Dataset, TARGET, load_raw, preprocess, split_holdout
from task2.metrics import evaluate
from task2.models import LassoRegression, RidgeRegression
from task2.plots import (
    plot_alpha_cv,
    plot_alpha_generalization,
    plot_coefficient_comparison,
    plot_coefficient_paths,
    plot_predictions,
    plot_residuals,
)
from task2.validation import cross_validate_lasso, cross_validate_ridge

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "housing.csv"
OUTPUT_DIR = ROOT / "results" / "task2"

TEST_SIZE = 0.2
SEED = 42
CV_FOLDS = 5
RIDGE_ALPHAS = np.logspace(-2, 4, 31)
LASSO_ALPHAS = np.logspace(-3, 3, 31)


def _to_dollars(
    scaled_prediction: np.ndarray, target_mean: float, target_std: float
) -> np.ndarray:
    """把标准化目标尺度上的预测还原成美元。"""
    return scaled_prediction * target_std + target_mean


def _fit_alpha_path(
    model_class,
    alphas: np.ndarray,
    data: Dataset,
) -> tuple[np.ndarray, pd.DataFrame]:
    """对每个 alpha 训练一次，记录系数、训练/测试误差和非零系数个数。"""
    coef_path = []
    rows = []

    for alpha in alphas:
        model = model_class(alpha=float(alpha)).fit(
            data.X_train, data.y_train_scaled
        )
        train_pred = _to_dollars(
            model.predict(data.X_train), data.target_mean, data.target_std
        )
        test_pred = _to_dollars(
            model.predict(data.X_test), data.target_mean, data.target_std
        )
        train_metrics = evaluate(data.y_train, train_pred)
        test_metrics = evaluate(data.y_test, test_pred)

        coef_path.append(model.coef_)
        rows.append(
            {
                "alpha": float(alpha),
                "train_mse": train_metrics["MSE"],
                "train_mae": train_metrics["MAE"],
                "train_rmse": train_metrics["RMSE"],
                "test_mse": test_metrics["MSE"],
                "test_mae": test_metrics["MAE"],
                "test_rmse": test_metrics["RMSE"],
                "n_nonzero": int(np.sum(np.abs(model.coef_) > 1e-8)),
            }
        )

    return np.vstack(coef_path), pd.DataFrame(rows)


def _markdown_metrics(metrics: pd.DataFrame) -> str:
    header = "| Model | Split | MSE | MAE | RMSE | R² |\n|---|---|---:|---:|---:|---:|"
    rows = [
        f"| {r.model} | {r.split} | {r.MSE:,.0f} | {r.MAE:,.0f} | "
        f"{r.RMSE:,.0f} | {r.R2:.3f} |"
        for r in metrics.itertuples()
    ]
    return "\n".join([header, *rows])


def _write_summary(
    path: Path,
    raw: pd.DataFrame,
    data: Dataset,
    metrics: pd.DataFrame,
    cv_results: dict[str, pd.DataFrame],
    best_alphas: dict[str, float],
    best_rows: dict[str, pd.Series],
    alpha_metrics: pd.DataFrame,
    coef_norms: dict[str, float],
    n_nonzero: dict[str, int],
) -> None:
    ridge_test = metrics.query("model == 'Ridge' and split == 'test'").iloc[0]
    lasso_test = metrics.query("model == 'Lasso' and split == 'test'").iloc[0]
    rmse_diff = (lasso_test.RMSE - ridge_test.RMSE) / ridge_test.RMSE * 100.0

    selected_rows = []
    for name in ("Ridge", "Lasso"):
        row = best_rows[name]
        selected_rows.append(
            f"| {name} | {best_alphas[name]:.4g} | "
            f"{row.cv_rmse_mean:,.0f} | {n_nonzero[name]} |"
        )
    selected_table = "\n".join(
        [
            "| Model | Selected alpha | CV RMSE | Nonzero coefficients |",
            "|---|---:|---:|---:|",
            *selected_rows,
        ]
    )

    strong_rows = []
    for name in ("Ridge", "Lasso"):
        subset = alpha_metrics[alpha_metrics["model"] == name]
        last = subset.iloc[-1]
        strong_rows.append(
            f"- {name} 在最大 alpha={last.alpha:.4g} 时，测试 RMSE="
            f"{last.test_rmse:,.0f}，非零系数 {int(last.n_nonzero)} 个。"
        )

    text = f"""# 回归任务二结果摘要

- 数据：California housing，{len(raw):,} 个样本，{len(data.feature_names)} 个特征。
- 目标变量：`median_house_value`，单位美元；原始数据中有 {int((raw[TARGET] == 500001).sum()):,} 个样本被截断在 500001。
- 数据划分：留出法，训练集 {len(data.y_train):,}、测试集 {len(data.y_test):,}，按目标值五分位分层，随机种子 {SEED}。
- 缺失值：训练集 `total_bedrooms` 缺失 {int(data.missing_before.sum())} 个，用训练集中位数填补；测试集使用同一中位数。
- 类别特征：`ocean_proximity` 做 one-hot 编码，并去掉一个基准类别。
- 特征构造：增加 `rooms_per_household`、`bedrooms_per_room`、`population_per_household`。
- 特征标准化：均值和标准差只来自训练集。
- 目标标准化：用训练集目标均值和标准差缩放，便于岭回归与套索回归使用同一数量级的正则参数；评价时已还原成美元。
- 参数选择：训练集内 {CV_FOLDS} 折交叉验证，按 CV MSE 最小选择。

## 参数选择

{selected_table}

## 测试集与训练集指标

{_markdown_metrics(metrics)}

## 正则化影响

- 岭回归系数 L2 范数：{coef_norms['Ridge']:.3f}
- 套索回归系数 L1 范数：{coef_norms['Lasso']:.3f}
- 所选套索模型的非零系数：{n_nonzero['Lasso']} / {len(data.feature_names)}
- 测试集上套索相对岭回归的 RMSE 变化：{rmse_diff:+.2f}%

强正则化的效果：

{chr(10).join(strong_rows)}

## 图表

- `predictions.png`：测试集真实值与预测值
- `residuals.png`：测试集残差
- `alpha_cv.png`：5 折交叉验证 RMSE 随 alpha 的变化
- `alpha_generalization.png`：训练/测试 RMSE 随 alpha 的变化（测试曲线仅用于分析，不参与选参）
- `coefficient_paths.png`：岭回归与套索回归的系数路径
- `coefficient_comparison.png`：选定 alpha 下的系数对比
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(DATA_PATH)
    train_df, test_df = split_holdout(raw, test_size=TEST_SIZE, seed=SEED)
    data = preprocess(train_df, test_df)

    print(f"样本总数: {len(raw):,}，特征数: {len(data.feature_names)}")
    print(f"训练集 / 测试集: {len(data.y_train):,} / {len(data.y_test):,}")
    print(
        "训练集缺失值: "
        + ", ".join(
            f"{name}={int(count)}"
            for name, count in data.missing_before.items()
            if count > 0
        )
    )

    ridge_cv = cross_validate_ridge(
        data.X_train,
        data.y_train_scaled,
        RIDGE_ALPHAS,
        target_std=data.target_std,
        n_splits=CV_FOLDS,
        seed=SEED,
    )
    lasso_cv = cross_validate_lasso(
        data.X_train,
        data.y_train_scaled,
        LASSO_ALPHAS,
        target_std=data.target_std,
        n_splits=CV_FOLDS,
        seed=SEED,
    )
    cv_results = {"Ridge": ridge_cv, "Lasso": lasso_cv}

    best_alphas: dict[str, float] = {}
    best_rows: dict[str, pd.Series] = {}
    for name, cv in cv_results.items():
        best_index = cv["cv_mse_mean"].idxmin()
        best_alphas[name] = float(cv.loc[best_index, "alpha"])
        best_rows[name] = cv.loc[best_index]

    ridge = RidgeRegression(alpha=best_alphas["Ridge"]).fit(
        data.X_train, data.y_train_scaled
    )
    lasso = LassoRegression(alpha=best_alphas["Lasso"]).fit(
        data.X_train, data.y_train_scaled
    )
    models = {"Ridge": ridge, "Lasso": lasso}

    predictions = {
        name: _to_dollars(
            model.predict(data.X_test), data.target_mean, data.target_std
        )
        for name, model in models.items()
    }

    rows = []
    for name, model in models.items():
        for split, X, y in (
            ("train", data.X_train, data.y_train),
            ("test", data.X_test, data.y_test),
        ):
            prediction = _to_dollars(
                model.predict(X), data.target_mean, data.target_std
            )
            rows.append({"model": name, "split": split, **evaluate(y, prediction)})
    metrics = pd.DataFrame(rows)

    ridge_path, ridge_alpha_metrics = _fit_alpha_path(
        RidgeRegression, RIDGE_ALPHAS, data
    )
    lasso_path, lasso_alpha_metrics = _fit_alpha_path(
        LassoRegression, LASSO_ALPHAS, data
    )
    ridge_alpha_metrics.insert(0, "model", "Ridge")
    lasso_alpha_metrics.insert(0, "model", "Lasso")
    alpha_metrics = pd.concat(
        [ridge_alpha_metrics, lasso_alpha_metrics], ignore_index=True
    )

    coef_paths = {"Ridge": ridge_path, "Lasso": lasso_path}
    alphas = {"Ridge": RIDGE_ALPHAS, "Lasso": LASSO_ALPHAS}
    n_nonzero = {
        name: int(np.sum(np.abs(model.coef_) > 1e-8))
        for name, model in models.items()
    }
    coef_norms = {
        "Ridge": float(np.linalg.norm(ridge.coef_)),
        "Lasso": float(np.linalg.norm(lasso.coef_, ord=1)),
    }

    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    alpha_metrics.to_csv(OUTPUT_DIR / "alpha_metrics.csv", index=False)
    pd.concat(
        [ridge_cv.assign(model="Ridge"), lasso_cv.assign(model="Lasso")],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "alpha_cv.csv", index=False)
    pd.DataFrame(
        {
            "feature": data.feature_names,
            "ridge_std": ridge.coef_,
            "lasso_std": lasso.coef_,
            "ridge_usd_per_std": ridge.coef_ * data.target_std,
            "lasso_usd_per_std": lasso.coef_ * data.target_std,
        }
    ).to_csv(OUTPUT_DIR / "coefficients.csv", index=False)

    plot_predictions(data.y_test, predictions, OUTPUT_DIR / "predictions.png")
    plot_residuals(data.y_test, predictions, OUTPUT_DIR / "residuals.png")
    plot_alpha_cv(cv_results, best_alphas, OUTPUT_DIR / "alpha_cv.png")
    plot_alpha_generalization(
        alpha_metrics, best_alphas, OUTPUT_DIR / "alpha_generalization.png"
    )
    plot_coefficient_paths(
        coef_paths,
        alphas,
        data.feature_names,
        best_alphas,
        OUTPUT_DIR / "coefficient_paths.png",
    )
    plot_coefficient_comparison(
        ridge.coef_,
        lasso.coef_,
        data.feature_names,
        OUTPUT_DIR / "coefficient_comparison.png",
    )

    print("\n交叉验证选择：")
    for name in ("Ridge", "Lasso"):
        row = best_rows[name]
        print(
            f"  {name}: alpha={best_alphas[name]:.4g}, "
            f"CV RMSE={row.cv_rmse_mean:,.0f}, 非零系数={n_nonzero[name]}"
        )
    print("\n指标：")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(
        f"\n系数范数: Ridge L2={coef_norms['Ridge']:.3f}, "
        f"Lasso L1={coef_norms['Lasso']:.3f}"
    )
    print(f"图表与结果已保存到: {OUTPUT_DIR}")

    _write_summary(
        OUTPUT_DIR / "summary.md",
        raw,
        data,
        metrics,
        cv_results,
        best_alphas,
        best_rows,
        alpha_metrics,
        coef_norms,
        n_nonzero,
    )


if __name__ == "__main__":
    main()
