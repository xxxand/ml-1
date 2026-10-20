"""回归任务一主流程：数据划分、建模、参数选择、评估与可视化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from task1.data import TARGET, load_raw, preprocess, split_holdout
from task1.metrics import evaluate, mean_squared_error
from task1.models import LinearRegression, RidgeRegression
from task1.plots import (
    plot_alpha_generalization,
    plot_alpha_selection,
    plot_coefficient_comparison,
    plot_coefficient_paths,
    plot_predictions,
    plot_residuals,
)
from task1.validation import cross_validate_ridge

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "boston.csv"
OUTPUT_DIR = ROOT / "results" / "task1"

TEST_SIZE = 0.2
SEED = 42
CV_FOLDS = 5
ALPHAS = np.logspace(-2, 3, 26)


def _markdown_table(metrics: pd.DataFrame) -> str:
    header = "| Model | Split | MSE | MAE | RMSE | R² |\n|---|---|---:|---:|---:|---:|"
    rows = [
        f"| {r.model} | {r.split} | {r.MSE:.2f} | {r.MAE:.2f} | "
        f"{r.RMSE:.2f} | {r.R2:.3f} |"
        for r in metrics.itertuples()
    ]
    return "\n".join([header, *rows])


def _write_summary(
    path: Path,
    metrics: pd.DataFrame,
    best_alpha: float,
    coef_norm_ols: float,
    coef_norm_ridge: float,
    test_improvement: float,
) -> None:
    text = f"""# 回归任务一结果摘要

- 数据划分：留出法，训练集 80%，测试集 20%，按 MEDV 五分位分层，随机种子 {SEED}
- 缺失值处理：仅用训练集特征中位数填补
- 特征处理：按训练集均值/标准差标准化
- 岭回归参数选择：训练集内 {CV_FOLDS} 折交叉验证
- 选定正则参数：alpha = {best_alpha:.4g}

## 测试集与训练集指标

{_markdown_table(metrics)}

## 正则化影响

- OLS 系数 L2 范数：{coef_norm_ols:.3f}
- 岭回归系数 L2 范数：{coef_norm_ridge:.3f}
- 测试集 MSE 相对变化：{test_improvement:+.1f}%

## 图表

- `predictions.png`：测试集真实值与预测值
- `residuals.png`：测试集残差
- `alpha_cv.png`：交叉验证误差随 alpha 的变化
- `alpha_generalization.png`：训练/测试误差随 alpha 的变化
- `coefficient_paths.png`：系数路径
- `coefficient_comparison.png`：OLS 与岭回归系数对比
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(DATA_PATH)
    train_df, test_df = split_holdout(raw, test_size=TEST_SIZE, seed=SEED)
    data = preprocess(train_df, test_df)

    print(f"样本总数: {len(raw)}，特征数: {len(data.feature_names)}")
    print(f"训练集 / 测试集: {len(data.y_train)} / {len(data.y_test)}")
    print(
        "训练集缺失值: "
        + ", ".join(
            f"{name}={int(count)}"
            for name, count in data.missing_before.items()
            if count > 0
        )
    )

    ols = LinearRegression().fit(data.X_train, data.y_train)

    cv_result = cross_validate_ridge(
        data.X_train,
        data.y_train,
        ALPHAS,
        n_splits=CV_FOLDS,
        seed=SEED,
    )
    best_row = cv_result.loc[cv_result["cv_mse_mean"].idxmin()]
    best_alpha = float(best_row["alpha"])

    ridge = RidgeRegression(alpha=best_alpha).fit(data.X_train, data.y_train)

    coef_path = []
    train_curve = []
    test_curve = []
    for alpha in ALPHAS:
        model = RidgeRegression(alpha=alpha).fit(data.X_train, data.y_train)
        coef_path.append(model.coef_)
        train_curve.append(mean_squared_error(data.y_train, model.predict(data.X_train)))
        test_curve.append(mean_squared_error(data.y_test, model.predict(data.X_test)))
    coef_path = np.vstack(coef_path)
    train_curve = np.asarray(train_curve)
    test_curve = np.asarray(test_curve)

    predictions = {
        "OLS": ols.predict(data.X_test),
        "Ridge": ridge.predict(data.X_test),
    }

    rows = []
    for name, model in (("OLS", ols), ("Ridge", ridge)):
        for split, X, y in (
            ("train", data.X_train, data.y_train),
            ("test", data.X_test, data.y_test),
        ):
            rows.append({"model": name, "split": split, **evaluate(y, model.predict(X))})
    metrics = pd.DataFrame(rows)

    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    cv_result.to_csv(OUTPUT_DIR / "alpha_cv.csv", index=False)
    pd.DataFrame(
        {
            "alpha": ALPHAS,
            "train_mse": train_curve,
            "test_mse": test_curve,
        }
    ).to_csv(OUTPUT_DIR / "alpha_generalization.csv", index=False)
    pd.DataFrame(
        {
            "feature": data.feature_names,
            "OLS": ols.coef_,
            "Ridge": ridge.coef_,
        }
    ).to_csv(OUTPUT_DIR / "coefficients.csv", index=False)

    plot_predictions(data.y_test, predictions, OUTPUT_DIR / "predictions.png")
    plot_residuals(data.y_test, predictions, OUTPUT_DIR / "residuals.png")
    plot_alpha_selection(cv_result, best_alpha, OUTPUT_DIR / "alpha_cv.png")
    plot_alpha_generalization(
        ALPHAS,
        train_curve,
        test_curve,
        best_alpha,
        OUTPUT_DIR / "alpha_generalization.png",
    )
    plot_coefficient_paths(
        ALPHAS,
        coef_path,
        data.feature_names,
        best_alpha,
        OUTPUT_DIR / "coefficient_paths.png",
    )
    plot_coefficient_comparison(
        ols.coef_,
        ridge.coef_,
        data.feature_names,
        OUTPUT_DIR / "coefficient_comparison.png",
    )

    ols_test = metrics.query("model == 'OLS' and split == 'test'").iloc[0]
    ridge_test = metrics.query("model == 'Ridge' and split == 'test'").iloc[0]
    norm_ols = float(np.linalg.norm(ols.coef_))
    norm_ridge = float(np.linalg.norm(ridge.coef_))
    improvement = (ridge_test.MSE - ols_test.MSE) / ols_test.MSE * 100.0

    print(f"\n交叉验证选择 alpha = {best_alpha:.4g}（CV MSE = {best_row.cv_mse_mean:.3f}）")
    print("\n指标：")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n系数 L2 范数: OLS={norm_ols:.3f}, Ridge={norm_ridge:.3f}")
    print(
        f"测试集 MSE 变化: {improvement:+.1f}% "
        f"(OLS {ols_test.MSE:.2f} -> Ridge {ridge_test.MSE:.2f})"
    )
    print(f"\n图表与结果已保存到: {OUTPUT_DIR}")

    _write_summary(
        OUTPUT_DIR / "summary.md",
        metrics,
        best_alpha,
        norm_ols,
        norm_ridge,
        improvement,
    )


if __name__ == "__main__":
    main()
