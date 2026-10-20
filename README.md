# 机器学习实验

本科生《机器学习》第一次实验作业代码与结果。

## 回归任务一：Boston 房价预测

使用 `datasets/boston.csv`，以 `MEDV` 为目标变量，手动实现普通最小二乘线性回归与岭回归。

零基础概念与代码讲解见 [`guide_1.md`](guide_1.md)。

- 缺失值：仅用训练集各特征的中位数填补（CRIM、ZN、INDUS、CHAS、AGE、LSTAT）。
- 数据划分：留出法，训练集 80%、测试集 20%，按 `MEDV` 五分位分层，随机种子 42。
- 特征标准化：均值/标准差均来自训练集。
- 线性回归：正规方程求解，截距参与估计。
- 岭回归：闭式解，截距不参与正则化；在训练集内部用 5 折交叉验证选择 `alpha`。
- 评价指标：MSE、MAE、RMSE、R²。

### 运行

```bash
pip install -r requirements.txt
python -m task1.main
```

### 代码结构

```
task1/
├── data.py        # 读取、缺失值填补、分层留出划分、标准化
├── models.py      # 手动实现的 LinearRegression / RidgeRegression
├── metrics.py     # MSE / MAE / RMSE / R²
├── validation.py  # K 折划分与 alpha 交叉验证
├── plots.py       # 预测值、残差、alpha 选择、系数路径等图
└── main.py        # 实验主流程
```

运行后结果写入 `results/task1/`：

- `metrics.csv`：训练集/测试集指标
- `alpha_cv.csv`、`alpha_generalization.csv`：参数选择与误差曲线数据
- `coefficients.csv`：OLS 与岭回归系数
- `predictions.png`、`residuals.png`：回归结果与残差
- `alpha_cv.png`、`alpha_generalization.png`、`coefficient_paths.png`、`coefficient_comparison.png`：正则化分析
- `summary.md`：本次运行的结果摘要

## 回归任务二：California 房价预测

使用 `datasets/housing.csv`，以 `median_house_value` 为目标变量，手动实现岭回归与套索回归。

零基础概念与代码讲解见 [`guide_2.md`](guide_2.md)。

- 缺失值：`total_bedrooms` 用训练集中位数填补。
- 特征处理：`ocean_proximity` 做 one-hot 编码，并构造 `rooms_per_household`、`bedrooms_per_room`、`population_per_household` 三个比例特征。
- 数据划分：留出法，训练集 80%、测试集 20%，按目标值五分位分层，随机种子 42。
- 标准化：特征和目标均按训练集均值/标准差缩放；评价时预测值还原为美元。
- 岭回归：闭式解，截距不参与正则化。
- 套索回归：坐标下降 + 软阈值，截距不参与正则化。
- 参数选择：训练集内 5 折交叉验证，分别选择两种模型的正则参数。
- 评价指标：MSE、MAE、RMSE、R²。

### 运行

```bash
python -m task2.main
```

### 代码结构

```
task2/
├── data.py        # 读取、缺失值、one-hot、比例特征、分层划分、标准化
├── models.py      # 手动实现的 RidgeRegression / LassoRegression
├── metrics.py     # MSE / MAE / RMSE / R²
├── validation.py  # K 折划分与两种模型的 alpha 交叉验证
├── plots.py       # 预测值、残差、alpha 曲线、系数路径等图
└── main.py        # 实验主流程
```

运行后结果写入 `results/task2/`：

- `metrics.csv`：训练集/测试集指标
- `alpha_cv.csv`、`alpha_metrics.csv`：交叉验证与不同 alpha 的误差数据
- `coefficients.csv`：两种模型在选定 alpha 下的系数
- `predictions.png`、`residuals.png`：回归结果与残差
- `alpha_cv.png`、`alpha_generalization.png`、`coefficient_paths.png`、`coefficient_comparison.png`：参数与正则化分析
- `summary.md`：本次运行的结果摘要

## 分类任务：Iris

使用 `datasets/iris.csv`，根据花萼、花瓣的四个数值特征预测三个鸢尾花类别。

- 数据划分：按类别分层留出 80% 训练、20% 测试，随机种子 42。
- 特征处理：按训练集均值/标准差标准化。
- Logistic 回归：手动实现 One-vs-Rest 二分类器，全批量梯度下降。
- 高斯朴素贝叶斯：手动实现，使用方差平滑参数。
- Softmax 回归（选做）：手动实现多分类交叉熵和梯度下降。
- 参数选择：训练集内 5 折分层交叉验证，比较学习率、L2 正则强度和方差平滑。
- 评价指标：Accuracy、Precision、Recall、F1，以及 One-vs-Rest ROC/AUC。

### 运行

```bash
python -m task3.main
```

### 代码结构

```
task3/
├── data.py        # 读取、分层划分、类别编码、标准化
├── models.py      # 手动实现的 Logistic / GaussianNB / Softmax
├── metrics.py     # 准确率、混淆矩阵、P/R/F1、ROC/AUC
├── validation.py  # 分层 K 折交叉验证与参数选择
├── plots.py       # 决策区域、混淆矩阵、ROC、参数曲线等图
└── main.py        # 实验主流程
```

运行后结果写入 `results/task3/`：

- `metrics.csv`、`per_class_metrics.csv`：整体与每类指标
- `cv_results.csv`、`cv_*.csv`：交叉验证参数结果
- `roc_auc.csv`、`confusion_matrices.csv`：AUC 与混淆矩阵数据
- `decision_boundaries.png`、`confusion_matrices.png`、`roc_curves.png`、`parameter_selection.png`、`metrics_comparison.png`：可视化结果
- `summary.md`：本次运行的结果摘要
