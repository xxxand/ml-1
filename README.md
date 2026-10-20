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
