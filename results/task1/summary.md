# 回归任务一结果摘要

- 数据划分：留出法，训练集 80%，测试集 20%，按 MEDV 五分位分层，随机种子 42
- 缺失值处理：仅用训练集特征中位数填补
- 特征处理：按训练集均值/标准差标准化
- 岭回归参数选择：训练集内 5 折交叉验证
- 选定正则参数：alpha = 3.981

## 测试集与训练集指标

| Model | Split | MSE | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| OLS | train | 22.90 | 3.32 | 4.79 | 0.735 |
| OLS | test | 22.08 | 3.22 | 4.70 | 0.711 |
| Ridge | train | 22.92 | 3.32 | 4.79 | 0.735 |
| Ridge | test | 22.11 | 3.19 | 4.70 | 0.711 |

## 正则化影响

- OLS 系数 L2 范数：6.885
- 岭回归系数 L2 范数：6.651
- 测试集 MSE 相对变化：+0.2%

## 图表

- `predictions.png`：测试集真实值与预测值
- `residuals.png`：测试集残差
- `alpha_cv.png`：交叉验证误差随 alpha 的变化
- `alpha_generalization.png`：训练/测试误差随 alpha 的变化
- `coefficient_paths.png`：系数路径
- `coefficient_comparison.png`：OLS 与岭回归系数对比
