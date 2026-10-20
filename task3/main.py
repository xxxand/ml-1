"""分类任务主流程：Iris 数据、Logistic / Gaussian NB / Softmax、评估与可视化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from task3.data import Dataset, load_raw, preprocess, split_holdout
from task3.metrics import (
    classification_metrics,
    confusion_matrix,
    macro_auc,
    per_class_metrics,
    roc_curve,
)
from task3.models import GaussianNaiveBayes, LogisticRegression, SoftmaxRegression
from task3.plots import (
    plot_confusion_matrices,
    plot_decision_boundaries,
    plot_metrics_comparison,
    plot_parameter_selection,
    plot_roc_curves,
)
from task3.validation import (
    cross_validate,
    make_param_grid,
    select_best_row,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "iris.csv"
OUTPUT_DIR = ROOT / "results" / "task3"

TEST_SIZE = 0.2
SEED = 42
CV_FOLDS = 5

LOGISTIC_GRID = make_param_grid(
    {
        "learning_rate": [0.05, 0.1, 0.2, 0.5],
        "l2": [0.0, 0.01, 0.1, 1.0],
    }
)
SOFTMAX_GRID = make_param_grid(
    {
        "learning_rate": [0.05, 0.1, 0.2, 0.5],
        "l2": [0.0, 0.01, 0.1, 1.0],
    }
)
NB_GRID = make_param_grid(
    {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]}
)


def _format_params(params: dict) -> str:
    return ", ".join(f"{key}={value:g}" for key, value in params.items())


def _metrics_table(metrics: pd.DataFrame) -> str:
    header = "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |\n|---|---:|---:|---:|---:|"
    rows = [
        f"| {row.model} | {row.accuracy:.3f} | {row.precision_macro:.3f} | "
        f"{row.recall_macro:.3f} | {row.f1_macro:.3f} |"
        for row in metrics.itertuples()
    ]
    return "\n".join([header, *rows])


def _per_class_table(per_class: pd.DataFrame) -> str:
    header = "| Model | Class | Precision | Recall | F1 | Support |\n|---|---|---:|---:|---:|---:|"
    rows = [
        f"| {row.model} | {row.class_name} | {row.precision:.3f} | {row.recall:.3f} | "
        f"{row.f1:.3f} | {row.support} |"
        for row in per_class.itertuples()
    ]
    return "\n".join([header, *rows])


def _auc_table(roc_auc: pd.DataFrame) -> str:
    header = "| Model | Class | AUC |\n|---|---|---:|"
    rows = [
        f"| {row.model} | {row.class_name} | {row.auc:.3f} |"
        for row in roc_auc.itertuples()
    ]
    return "\n".join([header, *rows])


def _write_summary(
    path: Path,
    data: Dataset,
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    roc_auc: pd.DataFrame,
    cv_results: dict[str, pd.DataFrame],
    best_params: dict[str, dict],
) -> None:
    best_accuracy_value = float(metrics["accuracy"].max())
    best_models = metrics.loc[
        np.isclose(metrics["accuracy"], best_accuracy_value), "model"
    ].tolist()
    best_model_text = "、".join(best_models)

    selected_rows = []
    for name in ("Logistic", "Gaussian NB", "Softmax"):
        cv = cv_results[name]
        mask = pd.Series(True, index=cv.index)
        for key, value in best_params[name].items():
            mask &= np.isclose(cv[key], value)
        row = cv.loc[mask].iloc[0]
        selected_rows.append(
            f"| {name} | {_format_params(best_params[name])} | "
            f"{row.cv_accuracy_mean:.3f} | {row.cv_f1_macro_mean:.3f} |"
        )
    selected_table = "\n".join(
        [
            "| Model | Best parameters | CV accuracy | CV macro F1 |",
            "|---|---|---:|---:|",
            *selected_rows,
        ]
    )

    macro_auc_rows = (
        roc_auc.groupby("model", sort=False)["auc"]
        .mean()
        .reset_index()
        .rename(columns={"auc": "macro_auc"})
    )
    auc_text = "、".join(
        f"{row.model} {row.macro_auc:.3f}" for row in macro_auc_rows.itertuples()
    )

    text = f"""# 分类任务结果摘要

- 数据：Iris，{len(data.y_train) + len(data.y_test)} 个样本，4 个数值特征，3 个类别。
- 数据划分：按类别分层留出，训练集 {len(data.y_train)}、测试集 {len(data.y_test)}，随机种子 {SEED}。
- 缺失值：{'无' if int(data.missing_before.sum()) == 0 else int(data.missing_before.sum())}。
- 特征处理：按训练集均值/标准差标准化。
- 参数选择：训练集内 {CV_FOLDS} 折分层交叉验证；先比较准确率，再以 macro F1 打破平局。
- 分类模型：Logistic（One-vs-Rest）、Gaussian Naive Bayes、Softmax（选做）。

## 参数选择

{selected_table}

## 测试集指标

{_metrics_table(metrics)}

## 每类指标

{_per_class_table(per_class)}

## 多分类 ROC / AUC

{_auc_table(roc_auc)}

宏平均 AUC：{auc_text}。

## 分析

- 测试集准确率最高的是 **{best_model_text}**（{best_accuracy_value:.3f}）。
- 测试集只有 {len(data.y_test)} 个样本，错 1 个样本就会让准确率变化约 {100.0 / len(data.y_test):.1f} 个百分点，因此不要过度解读模型之间的小差异。
- Logistic 使用 One-vs-Rest 策略，概率已归一化；Gaussian NB 假设特征在类别内条件独立；Softmax 直接建模三类概率。
- 详细结果见 `metrics.csv`、`per_class_metrics.csv`、`roc_auc.csv`、`cv_results.csv`。

## 图表

- `decision_boundaries.png`：花瓣长度/宽度平面上的决策区域（其余特征固定在训练集均值）
- `confusion_matrices.png`：三个模型的测试集混淆矩阵
- `roc_curves.png`：One-vs-Rest ROC 曲线
- `parameter_selection.png`：交叉验证准确率随参数变化
- `metrics_comparison.png`：Accuracy / Precision / Recall / F1 对比
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(DATA_PATH)
    train_df, test_df = split_holdout(raw, test_size=TEST_SIZE, seed=SEED)
    data = preprocess(train_df, test_df)

    print(f"样本总数: {len(raw)}，特征数: {len(data.feature_names)}")
    print(
        f"训练集 / 测试集: {len(data.y_train)} / {len(data.y_test)}，"
        f"类别: {', '.join(data.class_names)}"
    )

    logistic_cv = cross_validate(
        data.X_train,
        data.y_train,
        LOGISTIC_GRID,
        lambda **params: LogisticRegression(**params),
        data.class_names,
        n_splits=CV_FOLDS,
        seed=SEED,
    )
    nb_cv = cross_validate(
        data.X_train,
        data.y_train,
        NB_GRID,
        lambda **params: GaussianNaiveBayes(**params),
        data.class_names,
        n_splits=CV_FOLDS,
        seed=SEED,
    )
    softmax_cv = cross_validate(
        data.X_train,
        data.y_train,
        SOFTMAX_GRID,
        lambda **params: SoftmaxRegression(**params),
        data.class_names,
        n_splits=CV_FOLDS,
        seed=SEED,
    )
    cv_results = {
        "Logistic": logistic_cv,
        "Gaussian NB": nb_cv,
        "Softmax": softmax_cv,
    }

    best_rows = {name: select_best_row(cv) for name, cv in cv_results.items()}
    best_params = {
        "Logistic": {
            "learning_rate": float(best_rows["Logistic"]["learning_rate"]),
            "l2": float(best_rows["Logistic"]["l2"]),
        },
        "Gaussian NB": {
            "var_smoothing": float(best_rows["Gaussian NB"]["var_smoothing"])
        },
        "Softmax": {
            "learning_rate": float(best_rows["Softmax"]["learning_rate"]),
            "l2": float(best_rows["Softmax"]["l2"]),
        },
    }

    models = {
        "Logistic": LogisticRegression(**best_params["Logistic"]).fit(
            data.X_train, data.y_train
        ),
        "Gaussian NB": GaussianNaiveBayes(**best_params["Gaussian NB"]).fit(
            data.X_train, data.y_train
        ),
        "Softmax": SoftmaxRegression(**best_params["Softmax"]).fit(
            data.X_train, data.y_train
        ),
    }
    predictions = {
        name: model.predict(data.X_test) for name, model in models.items()
    }

    metric_rows = []
    per_class_rows = []
    for name, y_pred in predictions.items():
        metric_rows.append(
            {
                "model": name,
                **classification_metrics(data.y_test, y_pred, data.class_names),
            }
        )
        for row in per_class_metrics(data.y_test, y_pred, data.class_names):
            per_class_rows.append({"model": name, **row})
    metrics = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(per_class_rows)

    roc_data = {}
    roc_rows = []
    for name, model in models.items():
        probabilities = model.predict_proba(data.X_test)
        per_class_auc, macro_auc_value = macro_auc(
            data.y_test, probabilities, data.class_names
        )
        curves = {}
        for class_index, class_name in enumerate(data.class_names):
            fpr, tpr = roc_curve(
                (data.y_test == class_index).astype(int),
                probabilities[:, class_index],
            )
            curves[class_name] = (fpr, tpr, per_class_auc[class_name])
            roc_rows.append(
                {
                    "model": name,
                    "class_name": class_name,
                    "auc": per_class_auc[class_name],
                }
            )
        roc_data[name] = {"per_class": curves, "macro_auc": macro_auc_value}

    roc_auc = pd.DataFrame(roc_rows)

    # 保存交叉验证结果（含参数列，便于直接查看）
    for name, cv in cv_results.items():
        cv.to_csv(
            OUTPUT_DIR / f"cv_{name.lower().replace(' ', '_')}.csv", index=False
        )
    cv_combined_rows = []
    for name, cv in cv_results.items():
        for row in cv.itertuples(index=False):
            parameter_names = [
                column
                for column in cv.columns
                if column not in {"cv_accuracy_mean", "cv_accuracy_std", "cv_f1_macro_mean"}
            ]
            params = {column: getattr(row, column) for column in parameter_names}
            cv_combined_rows.append(
                {
                    "model": name,
                    "params": _format_params(params),
                    "cv_accuracy_mean": row.cv_accuracy_mean,
                    "cv_accuracy_std": row.cv_accuracy_std,
                    "cv_f1_macro_mean": row.cv_f1_macro_mean,
                }
            )
    pd.DataFrame(cv_combined_rows).to_csv(
        OUTPUT_DIR / "cv_results.csv", index=False
    )

    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    per_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
    roc_auc.to_csv(OUTPUT_DIR / "roc_auc.csv", index=False)

    confusion_rows = []
    for name, y_pred in predictions.items():
        matrix = confusion_matrix(data.y_test, y_pred, len(data.class_names))
        for true_index, true_name in enumerate(data.class_names):
            for pred_index, pred_name in enumerate(data.class_names):
                confusion_rows.append(
                    {
                        "model": name,
                        "true": true_name,
                        "predicted": pred_name,
                        "count": int(matrix[true_index, pred_index]),
                    }
                )
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT_DIR / "confusion_matrices.csv", index=False
    )

    plot_decision_boundaries(models, data, OUTPUT_DIR / "decision_boundaries.png")
    plot_confusion_matrices(
        data.y_test, predictions, data.class_names, OUTPUT_DIR / "confusion_matrices.png"
    )
    plot_roc_curves(roc_data, OUTPUT_DIR / "roc_curves.png")
    plot_parameter_selection(
        cv_results, best_params, OUTPUT_DIR / "parameter_selection.png"
    )
    plot_metrics_comparison(metrics, OUTPUT_DIR / "metrics_comparison.png")

    print("\n交叉验证选择：")
    for name in ("Logistic", "Gaussian NB", "Softmax"):
        row = best_rows[name]
        print(
            f"  {name}: {_format_params(best_params[name])}, "
            f"CV accuracy={row.cv_accuracy_mean:.3f}, "
            f"CV macro F1={row.cv_f1_macro_mean:.3f}"
        )

    print("\n测试集指标：")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nAUC：")
    print(roc_auc.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\n图表与结果已保存到: {OUTPUT_DIR}")

    _write_summary(
        OUTPUT_DIR / "summary.md",
        data,
        metrics,
        per_class,
        roc_auc,
        cv_results,
        best_params,
    )


if __name__ == "__main__":
    main()
