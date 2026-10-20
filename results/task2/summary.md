# 回归任务二结果摘要

- 数据：California housing，20,640 个样本，15 个特征。
- 目标变量：`median_house_value`，单位美元；原始数据中有 965 个样本被截断在 500001。
- 数据划分：留出法，训练集 16,512、测试集 4,128，按目标值五分位分层，随机种子 42。
- 缺失值：训练集 `total_bedrooms` 缺失 161 个，用训练集中位数填补；测试集使用同一中位数。
- 类别特征：`ocean_proximity` 做 one-hot 编码，并去掉一个基准类别。
- 特征构造：增加 `rooms_per_household`、`bedrooms_per_room`、`population_per_household`。
- 特征标准化：均值和标准差只来自训练集。
- 目标标准化：用训练集目标均值和标准差缩放，便于岭回归与套索回归使用同一数量级的正则参数；评价时已还原成美元。
- 参数选择：训练集内 5 折交叉验证，按 CV MSE 最小选择。

## 参数选择

| Model | Selected alpha | CV RMSE | Nonzero coefficients |
|---|---:|---:|---:|
| Ridge | 25.12 | 68,101 | 15 |
| Lasso | 15.85 | 68,093 | 15 |

## 测试集与训练集指标

| Model | Split | MSE | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Ridge | train | 4,593,348,057 | 49,089 | 67,774 | 0.652 |
| Ridge | test | 4,963,013,916 | 50,109 | 70,449 | 0.640 |
| Lasso | train | 4,594,319,411 | 49,089 | 67,781 | 0.652 |
| Lasso | test | 4,960,485,044 | 50,105 | 70,431 | 0.640 |

## 正则化影响

- 岭回归系数 L2 范数：1.099
- 套索回归系数 L1 范数：2.865
- 所选套索模型的非零系数：15 / 15
- 测试集上套索相对岭回归的 RMSE 变化：-0.03%

强正则化的效果：

- Ridge 在最大 alpha=1e+04 时，测试 RMSE=80,032，非零系数 15 个。
- Lasso 在最大 alpha=1000 时，测试 RMSE=76,558，非零系数 4 个。

## 图表

- `predictions.png`：测试集真实值与预测值
- `residuals.png`：测试集残差
- `alpha_cv.png`：5 折交叉验证 RMSE 随 alpha 的变化
- `alpha_generalization.png`：训练/测试 RMSE 随 alpha 的变化（测试曲线仅用于分析，不参与选参）
- `coefficient_paths.png`：岭回归与套索回归的系数路径
- `coefficient_comparison.png`：选定 alpha 下的系数对比
