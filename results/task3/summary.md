# 分类任务结果摘要

- 数据：Iris，150 个样本，4 个数值特征，3 个类别。
- 数据划分：按类别分层留出，训练集 120、测试集 30，随机种子 42。
- 缺失值：无。
- 特征处理：按训练集均值/标准差标准化。
- 参数选择：训练集内 5 折分层交叉验证；先比较准确率，再以 macro F1 打破平局。
- 分类模型：Logistic（One-vs-Rest）、Gaussian Naive Bayes、Softmax（选做）。

## 参数选择

| Model | Best parameters | CV accuracy | CV macro F1 |
|---|---|---:|---:|
| Logistic | learning_rate=0.5, l2=0 | 0.942 | 0.942 |
| Gaussian NB | var_smoothing=1e-09 | 0.950 | 0.950 |
| Softmax | learning_rate=0.5, l2=0 | 0.950 | 0.950 |

## 测试集指标

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| Logistic | 1.000 | 1.000 | 1.000 | 1.000 |
| Gaussian NB | 0.967 | 0.970 | 0.967 | 0.967 |
| Softmax | 1.000 | 1.000 | 1.000 | 1.000 |

## 每类指标

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Logistic | setosa | 1.000 | 1.000 | 1.000 | 10 |
| Logistic | versicolor | 1.000 | 1.000 | 1.000 | 10 |
| Logistic | virginica | 1.000 | 1.000 | 1.000 | 10 |
| Gaussian NB | setosa | 1.000 | 1.000 | 1.000 | 10 |
| Gaussian NB | versicolor | 0.909 | 1.000 | 0.952 | 10 |
| Gaussian NB | virginica | 1.000 | 0.900 | 0.947 | 10 |
| Softmax | setosa | 1.000 | 1.000 | 1.000 | 10 |
| Softmax | versicolor | 1.000 | 1.000 | 1.000 | 10 |
| Softmax | virginica | 1.000 | 1.000 | 1.000 | 10 |

## 多分类 ROC / AUC

| Model | Class | AUC |
|---|---|---:|
| Logistic | setosa | 1.000 |
| Logistic | versicolor | 1.000 |
| Logistic | virginica | 1.000 |
| Gaussian NB | setosa | 1.000 |
| Gaussian NB | versicolor | 0.995 |
| Gaussian NB | virginica | 0.995 |
| Softmax | setosa | 1.000 |
| Softmax | versicolor | 1.000 |
| Softmax | virginica | 1.000 |

宏平均 AUC：Logistic 1.000、Gaussian NB 0.997、Softmax 1.000。

## 分析

- 测试集准确率最高的是 **Logistic、Softmax**（1.000）。
- 测试集只有 30 个样本，错 1 个样本就会让准确率变化约 3.3 个百分点，因此不要过度解读模型之间的小差异。
- Logistic 使用 One-vs-Rest 策略，概率已归一化；Gaussian NB 假设特征在类别内条件独立；Softmax 直接建模三类概率。
- 详细结果见 `metrics.csv`、`per_class_metrics.csv`、`roc_auc.csv`、`cv_results.csv`。

## 图表

- `decision_boundaries.png`：花瓣长度/宽度平面上的决策区域（其余特征固定在训练集均值）
- `confusion_matrices.png`：三个模型的测试集混淆矩阵
- `roc_curves.png`：One-vs-Rest ROC 曲线
- `parameter_selection.png`：交叉验证准确率随参数变化
- `metrics_comparison.png`：Accuracy / Precision / Recall / F1 对比
