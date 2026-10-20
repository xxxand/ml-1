# 分类任务（Iris）零基础学习指南

> 这份指南假设你几乎没有机器学习基础。如果你已经读过 `guide_1.md` 或 `guide_2.md`，可以跳过第 1 节的部分复习，直接从第 2 节开始。
>
> 读完之后，你应该能回答：
>
> 1. 分类问题和回归问题有什么区别？
> 2. Logistic 回归、Gaussian 朴素贝叶斯、Softmax 回归分别在做什么？
> 3. 交叉验证怎样选择学习率、L2 正则强度和方差平滑参数？
> 4. Accuracy、Precision、Recall、F1、混淆矩阵、ROC/AUC 怎么读？
> 5. 为什么 Logistic 和 Softmax 测试集全对，而交叉验证不是 100%？

---

## 怎么读这份指南

- 只有 30 分钟：读第 0 节 → 第 1 节 → 第 2 节 → 3.1～3.4 → 4.1～4.3 → 6.2 → 7.1～7.2 → 8.1。
- 想彻底看懂代码：按第 2～6 节顺序读，再配合第 9 节的代码地图。
- 只想写报告：直接看第 7、8 节；8.4 节有一段可以直接改写的文字。
- 遇到陌生词：先查第 12 节术语表，再做第 13 节自测题。

---

## 0. 先用五句话看懂这个任务

1. 数据里有 150 朵鸢尾花，每朵花有 4 个测量值（花萼长度、花萼宽度、花瓣长度、花瓣宽度），还有一个答案：品种。
2. **Logistic 回归**给每个品种训练一个“是不是这个品种”的二分类器，然后取得分最高的品种。
3. **Gaussian 朴素贝叶斯**用每个品种的特征均值和方差，结合贝叶斯公式计算“这朵花属于各品种的概率”。
4. **Softmax 回归**直接用一个模型输出三个品种的概率，是 Logistic 回归的多分类版本（本实验的选做部分）。
5. 训练集内部用 5 折分层交叉验证选参数；测试集只有 30 朵花，Logistic 和 Softmax 全对，Gaussian NB 错了 1 朵，但样本太小，不能因此断言谁永远更好。

---

## 1. 分类问题的最小世界观

### 1.1 数据长什么样

Iris 数据可以想成一张 Excel 表：

| 花 | 花萼长度 | 花萼宽度 | 花瓣长度 | 花瓣宽度 | 品种 |
|---|---:|---:|---:|---:|---|
| 第 1 朵 | 5.1 | 3.5 | 1.4 | 0.2 | setosa |
| 第 2 朵 | 4.9 | 3.0 | 1.4 | 0.2 | setosa |
| 第 51 朵 | 7.0 | 3.2 | 4.7 | 1.4 | versicolor |
| 第 101 朵 | 6.3 | 3.3 | 6.0 | 2.5 | virginica |
| …… | …… | …… | …… | …… | …… |

- 每一行是一个**样本**（sample），这里是一朵花。
- 4 个测量列是**特征**（feature），记作 `X`。
- 品种列是**目标**或**标签**（target / label），记作 `y`。
- 因为品种是离散类别，不是连续数值，所以这是**分类**问题。

如果目标是房价、温度这种连续数值，就是**回归**；如果目标是“猫/狗/鸟”或“A/B/C 类”，就是**分类**。

### 1.2 训练、测试、泛化

- **训练集**：用来拟合模型参数的数据，相当于平时练习题。
- **测试集**：模型没见过的数据，相当于期末考试。
- **泛化**：模型在没见过的数据上的表现。
- **过拟合**：练习题做得很好，期末考试却很差。
- **欠拟合**：模型太简单，练习题和期末考试都不好。

本实验：

```text
150 朵花
  ├─ 训练集：120 朵（每个品种 40 朵）
  └─ 测试集： 30 朵（每个品种 10 朵）
```

测试集只在最后评价时使用。

### 1.3 分类模型在做什么

分类模型通常做两件事：

1. 给每个类别算一个**得分**（score）；
2. 把得分变成概率，或者直接取得分最高的类别。

例如三个品种的得分是：

```text
setosa     0.02
versicolor 0.15
virginica  0.83
```

模型就会预测 `virginica`。

不同模型的区别在于：

- 得分怎么算？
- 怎么从得分得到概率？
- 参数怎么从训练数据中学习？

### 1.4 标准化

4 个特征的量纲虽然接近，但数值范围不同：

- 花萼长度大约 4.3～7.9；
- 花瓣宽度大约 0.1～2.5。

代码用训练集的均值和标准差做标准化：

```text
z = (x - 训练集均值) / 训练集标准差
```

标准化后每个特征均值约为 0、标准差约为 1。

对 Logistic 和 Softmax 的梯度下降来说，标准化能让不同特征的更新步调更一致，收敛更稳定；对 Gaussian NB 来说，标准化不是必需的（它理论上对特征缩放不敏感），但代码为了统一处理也做了标准化。

### 1.5 分类和回归的评价指标不同

回归用 MSE、MAE、RMSE、R²；分类不能用这些，而用：

- 准确率（Accuracy）；
- 精确率（Precision）；
- 召回率（Recall）；
- F1 值；
- 混淆矩阵；
- ROC 曲线和 AUC。

后面会逐个解释。

---

## 2. 数据：Iris

### 2.1 数据规模

`datasets/iris.csv`：

- 150 行；
- 6 列，其中第一列 `Unnamed: 0` 是行号，不是特征，代码会删掉；
- 4 个数值特征 + 1 个类别目标 `Species`。

4 个特征：

| 列名 | 含义 |
|---|---|
| `Sepal.Length` | 花萼长度（cm） |
| `Sepal.Width` | 花萼宽度（cm） |
| `Petal.Length` | 花瓣长度（cm） |
| `Petal.Width` | 花瓣宽度（cm） |

三个类别各 50 朵：

```text
setosa      50
versicolor  50
virginica   50
```

这是一个**类别平衡**的数据集，所以准确率不会因为某一类特别多而失真。不过后面仍然会看每一类的 Precision/Recall。

### 2.2 缺失值

Iris 数据没有缺失值。代码仍然保留了缺失值检查和中位数填补逻辑，是为了让流程更通用；本数据不会真正用到填补。

### 2.3 分层划分

代码按类别分层留出：

```python
train_df, test_df = split_holdout(raw, test_size=0.2, seed=SEED)
```

结果：

```text
训练集 120 朵：setosa 40、versicolor 40、virginica 40
测试集  30 朵：setosa 10、versicolor 10、virginica 10
```

为什么分层？如果完全随机抽 30 朵，可能某一类只有 7 朵、另一类有 13 朵，测试集类别比例和训练集差很多，评价会不稳定。分层保证每类比例接近。

`SEED = 42` 让划分可复现。

### 2.4 数据泄漏

所有统计量只能用训练集计算：

- 特征均值、标准差；
- 缺失值中位数；
- 类别列表。

测试集只用训练集的统计量做变换，不能用自己的均值/标准差，否则测试信息会泄漏进训练。

---

## 3. Logistic 回归

名字里有“回归”，但它其实是**分类**模型。名字来自它建模的是“对数几率”（log-odds），不是输出连续房价。

### 3.1 二分类：从线性得分到概率

先考虑只有两个类别的情况，比如“是 setosa”与“不是 setosa”。

线性得分：

```text
z = w1x1 + w2x2 + w3x3 + w4x4 + b
```

`z` 可以是任意实数。用 **sigmoid** 函数把它压到 0～1：

```text
p = sigmoid(z) = 1 / (1 + exp(-z))
```

性质：

- `z = 0` → `p = 0.5`；
- `z` 很大 → `p` 接近 1；
- `z` 很小 → `p` 接近 0。

把 `p` 解释为“属于正类的概率”。

### 3.2 决策边界

二分类时，通常规定：

```text
p >= 0.5  →  预测为正类
p < 0.5   →  预测为负类
```

`p = 0.5` 对应 `z = 0`，也就是：

```text
w1x1 + w2x2 + w3x3 + w4x4 + b = 0
```

这是一个**线性决策边界**。所以 Logistic 回归是线性分类器：它不能自动画出弯曲的边界。

### 3.3 损失函数：交叉熵

训练时不能直接用 MSE 吗？可以，但交叉熵更适合概率输出，优化性质更好。

二分类交叉熵：

```text
L = -[ y·log(p) + (1-y)·log(1-p) ]
```

- 真实 `y = 1` 时，损失是 `-log(p)`：预测概率越接近 1，损失越小；
- 真实 `y = 0` 时，损失是 `-log(1-p)`：预测概率越接近 0，损失越小。

对所有样本取平均，再加上 L2 惩罚：

```text
平均交叉熵 + (l2/2)·||w||²
```

`l2` 越大，系数被压得越小，模型更保守。

### 3.4 梯度下降

Logistic 回归没有像线性回归那样的简单闭式解，代码用**梯度下降**：

1. 初始化 `w = 0, b = 0`；
2. 计算当前预测概率 `p`；
3. 计算梯度；
4. 沿梯度反方向走一小步；
5. 重复，直到梯度很小或达到最大迭代次数。

二分类的梯度是：

```text
grad_w = X^T (p - y) / n + l2·w
grad_b = mean(p - y)
```

更新：

```text
w = w - learning_rate · grad_w
b = b - learning_rate · grad_b
```

- `learning_rate` 是步长；
- 太小：收敛慢；
- 太大：可能震荡甚至发散；
- `max_iter` 是最大迭代次数；
- `tol` 是停止条件：梯度足够小就停。

对应代码在 `task3/models.py` 的 `LogisticRegression.fit()`。

### 3.5 多分类：One-vs-Rest

Iris 有 3 个类别。Logistic 回归本身是二分类器，所以代码用 **One-vs-Rest（OvR）**：

```text
分类器 1：setosa     vs  (versicolor + virginica)
分类器 2：versicolor vs  (setosa + virginica)
分类器 3：virginica  vs  (setosa + versicolor)
```

每个分类器输出一个“属于该类的得分”。预测时取三个得分中最大的：

```text
prediction = argmax(score_setosa, score_versicolor, score_virginica)
```

代码里 `coef_` 的形状是 `3 × 4`：3 个类别、4 个特征；`intercept_` 是长度 3 的向量。

### 3.6 为什么代码要归一化概率

每个二分类器都输出一个 0～1 的 sigmoid 得分，但三个得分加起来不一定是 1。代码用：

```python
scores = sigmoid(decision_function(X))
probabilities = scores / scores.sum(axis=1, keepdims=True)
```

归一化后每行概率和为 1，方便画 ROC/AUC。**归一化不改变 argmax**，所以不改变分类结果。

### 3.7 参数怎么影响结果

代码搜索：

```python
learning_rate = [0.05, 0.1, 0.2, 0.5]
l2 = [0.0, 0.01, 0.1, 1.0]
```

交叉验证结果：

- `learning_rate=0.5, l2=0` 最好，CV accuracy 约 0.942；
- `l2=0.1` 时，各学习率都降到约 0.875；
- `l2=1.0` 时降到约 0.850；
- 学习率太小（0.05）且 `l2=0` 时只有约 0.900。

说明：

- 在这个简单数据上，适当大的学习率收敛更好；
- 过强的 L2 惩罚会把系数压得太小，导致欠拟合；
- 交叉验证准确率的波动约 0.04～0.07，模型间的小差异要谨慎解读。

---

## 4. Gaussian 朴素贝叶斯

### 4.1 贝叶斯公式

我们想算：

```text
P(品种 | 花的特征)
```

贝叶斯公式：

```text
P(类别 | 特征) = P(特征 | 类别) · P(类别) / P(特征)
```

- `P(类别)` 是**先验**：在没看特征之前，各类别出现的概率；Iris 三类各 1/3。
- `P(特征 | 类别)` 是**似然**：如果已知是某个品种，这组测量值出现的可能性。
- `P(类别 | 特征)` 是**后验**：看到特征后，属于各类别的概率。

分母 `P(特征)` 对所有类别都一样，比较时可以不看，只比较：

```text
P(特征 | 类别) · P(类别)
```

### 4.2 高斯假设

特征都是连续数值，Gaussian NB 假设：

> 在每个类别内部，每个特征服从正态分布。

对类别 `c`、特征 `j`：

```text
P(xj | c) = 1 / sqrt(2π·var_cj) · exp( -(xj - mean_cj)² / (2·var_cj) )
```

其中 `mean_cj` 和 `var_cj` 是该类别该特征的均值和方差，从训练数据中估计。

### 4.3 朴素：特征条件独立

“朴素”指的是一个强假设：

> 在已知类别的条件下，各特征相互独立。

于是：

```text
P(x1, x2, x3, x4 | c) = P(x1|c)·P(x2|c)·P(x3|c)·P(x4|c)
```

Iris 的花瓣长度和花瓣宽度其实高度相关，所以这个假设并不完全成立；但 Gaussian NB 仍然常常表现不错，因为分类只需要比较各类别得分的大小。

### 4.4 用对数避免下溢

很多概率相乘会变得非常小，计算机可能下溢成 0。代码在**对数空间**计算：

```text
log P(c | x) ∝ log P(c) + Σ_j log P(xj | c)
```

高斯对数似然：

```text
-0.5·log(2π·var_cj) - (xj - mean_cj)² / (2·var_cj)
```

最后用 softmax 把对数联合概率变成概率。

对应代码在 `task3/models.py` 的 `GaussianNaiveBayes`：

- `priors_`：每个类别的先验；
- `means_`：每个类别每个特征的均值；
- `variances_`：每个类别每个特征的方差；
- `_log_joint()`：对数先验 + 对数似然。

### 4.5 方差平滑

如果某个特征在某个类别内方差为 0（所有值都一样），高斯公式会除以 0。代码加一个很小的平滑量：

```text
epsilon = var_smoothing × 训练集特征方差的最大值
variances += epsilon
```

`var_smoothing` 越大，方差越“被抹平”，模型越不敏感，但太大就会把类别之间的差异抹掉。

本实验的交叉验证结果：

| var_smoothing | CV accuracy |
|---:|---:|
| 1e-9 | 0.950 |
| 1e-7 | 0.950 |
| 1e-5 | 0.950 |
| 1e-3 | 0.950 |
| 0.1 | 0.925 |
| 1.0 | 0.875 |

前四个几乎没有区别；平滑到 0.1、1.0 后准确率下降，说明过大的平滑会让模型欠拟合。代码选中 1e-9。

### 4.6 Gaussian NB 的优缺点

优点：

- 训练非常快，只需算均值和方差；
- 小样本也能工作；
- 输出概率，适合和 Logistic/Softmax 比较。

缺点：

- 特征条件独立假设太强；
- 对特征分布形状有假设；
- 当特征高度相关时，概率估计可能不够准。

本实验测试集上 Gaussian NB 只错了 1 朵，说明它在这个简单数据上仍然够用。

---

## 5. Softmax 回归（选做）

### 5.1 从二分类到多分类

Softmax 回归直接给每个类别算一个线性得分：

```text
z_k = w_k·x + b_k       k = setosa, versicolor, virginica
```

然后用 **softmax** 变成概率：

```text
p_k = exp(z_k) / Σ_j exp(z_j)
```

性质：

- 每个 `p_k` 在 0～1 之间；
- 三个概率之和为 1；
- 得分最高的类别概率最大。

和 Logistic OvR 的区别：

| 对比项 | Logistic OvR | Softmax |
|---|---|---|
| 分类器数量 | 每个类别一个二分类器 | 一个多分类模型 |
| 概率 | 三个独立 sigmoid 后归一化 | 直接 softmax，和为 1 |
| 训练目标 | 三个二分类交叉熵 | 一个多分类交叉熵 |
| 决策边界 | 线性 | 线性 |

### 5.2 多分类交叉熵

真实类别用 one-hot 表示：

```text
setosa     = [1, 0, 0]
versicolor = [0, 1, 0]
virginica  = [0, 0, 1]
```

损失：

```text
L = -Σ_k y_k · log(p_k)
```

如果真实类别是 versicolor，就只有 `-log(p_versicolor)` 这一项起作用：模型给真实类别的概率越高，损失越小。

加 L2 惩罚后，梯度是：

```text
grad_W = X^T (P - Y) / n + l2·W
grad_b = mean(P - Y)
```

更新方式和 Logistic 一样，用梯度下降。

对应代码在 `task3/models.py` 的 `SoftmaxRegression`。

### 5.3 参数影响

代码搜索：

```python
learning_rate = [0.05, 0.1, 0.2, 0.5]
l2 = [0.0, 0.01, 0.1, 1.0]
```

交叉验证结果：

- `l2=0` 或 `0.01` 时最好，约 0.942～0.950；
- `l2=0.1` 时约 0.908；
- `l2=1.0` 时约 0.858。

最终选中 `learning_rate=0.5, l2=0`，CV accuracy 0.950。`learning_rate=0.05, l2=0.01` 也达到 0.950，二者差距在交叉验证波动范围内。

### 5.4 为什么还保留 Logistic

Softmax 看起来更“直接”，但 Logistic OvR 也有价值：

- 每个类别可以单独解释；
- 某些类别样本很少时，OvR 更灵活；
- 它是理解多分类的基础。

本实验把两者都实现，方便比较。

---

## 6. 交叉验证与参数选择

### 6.1 为什么不能直接用测试集

如果拿测试集试很多组参数，挑测试准确率最高的，测试集就变成了“调参数据”，最后成绩会偏乐观。正确做法：

```text
训练集：拟合模型参数
训练集内部再分：选超参数
测试集：只在最后评价一次
```

学习率、L2 强度、方差平滑都是**超参数**，不是模型从数据里直接学出来的。

### 6.2 分层 5 折交叉验证

Iris 三类各 50 朵，训练集每类 40 朵。代码按类别分层切成 5 折：

```text
每折验证集：每类 8 朵，共 24 朵
每折训练集：每类 32 朵，共 96 朵
```

每轮用 4 折训练、1 折验证，重复 5 轮，取平均准确率。

为什么分层？保证每折都有三个类别，不会出现某一折只有两个类别。

代码在 `task3/validation.py`：

- `stratified_kfold_indices()`：分层切分；
- `cross_validate()`：对每组参数做 5 折；
- `select_best_row()`：先按 CV 准确率排序，准确率相同再比较 macro F1。

### 6.3 参数网格

| 模型 | 参数 | 候选值 |
|---|---|---|
| Logistic | learning_rate | 0.05, 0.1, 0.2, 0.5 |
| Logistic | l2 | 0, 0.01, 0.1, 1.0 |
| Gaussian NB | var_smoothing | 1e-9, 1e-7, 1e-5, 1e-3, 0.1, 1.0 |
| Softmax | learning_rate | 0.05, 0.1, 0.2, 0.5 |
| Softmax | l2 | 0, 0.01, 0.1, 1.0 |

Logistic 和 Softmax 各 16 组，Gaussian NB 6 组；每组做 5 折。

### 6.4 本实验的选择结果

| 模型 | 选中参数 | CV accuracy | CV macro F1 |
|---|---|---:|---:|
| Logistic | lr=0.5, l2=0 | 0.942 | 0.942 |
| Gaussian NB | var_smoothing=1e-9 | 0.950 | 0.950 |
| Softmax | lr=0.5, l2=0 | 0.950 | 0.950 |

### 6.5 怎么读 `parameter_selection.png`

- 左图 Logistic，中图 Softmax，右图 Gaussian NB；
- 纵轴是 5 折交叉验证准确率；
- 误差棒是 5 折之间的标准差；
- 红星是选中的参数；
- 对 Logistic/Softmax，横轴是 L2 强度，不同颜色是学习率；
- 对 Gaussian NB，横轴是 `var_smoothing`（对数刻度）。

读法：

- L2 越大，准确率总体下降，说明强正则化导致欠拟合；
- 学习率 0.5 在这一数据上表现较好；
- Gaussian NB 的平滑参数在 1e-9～1e-3 之间没有区别，0.1 和 1.0 明显变差；
- 误差棒很大（约 0.03～0.07），说明 120 个训练样本下交叉验证本身有波动，不要过度追求小数点后的差异。

### 6.6 一个严谨性提醒

本实验为了代码简洁，先对完整训练集做标准化，再在训练集内部切 5 折。严格来说，每一折内部应该重新用该折的训练部分计算均值和标准差。Iris 没有缺失值、样本量小、特征分布稳定，这个简化对结论影响很小；如果做更严格的比较，可以改成“每折内部预处理”。

---

## 7. 评价指标：从准确率到 ROC/AUC

### 7.1 准确率

```text
Accuracy = 预测正确的样本数 / 总样本数
```

本实验测试集 30 朵：

- 全对：1.000；
- 错 1 朵：29/30 = 0.967；
- 错 2 朵：28/30 = 0.933。

所以**错 1 朵就会让准确率变化 3.3 个百分点**。30 个样本的测试集很小，模型之间 0.033 的差距可能只是随机波动。

### 7.2 混淆矩阵

混淆矩阵的行是**真实类别**，列是**预测类别**。

Logistic 和 Softmax 的测试集混淆矩阵都是：

```text
             预测 setosa  预测 versicolor  预测 virginica
真实 setosa        10            0               0
真实 versicolor     0           10               0
真实 virginica      0            0              10
```

Gaussian NB：

```text
             预测 setosa  预测 versicolor  预测 virginica
真实 setosa        10            0               0
真实 versicolor     0           10               0
真实 virginica      0            1               9
```

最后一行表示：1 朵真实 virginica 被预测成 versicolor。

### 7.3 Precision、Recall、F1

先以某个类别为“正类”，其余为“负类”：

- **TP**（True Positive）：真实是正类，预测也是正类；
- **FP**（False Positive）：真实是负类，预测成正类；
- **FN**（False Negative）：真实是正类，预测成负类；
- **TN**（True Negative）：真实是负类，预测也是负类。

Precision：

```text
Precision = TP / (TP + FP)
```

“预测为正类的样本里，有多少是真的正类。” 也叫查准率。

Recall：

```text
Recall = TP / (TP + FN)
```

“真实为正类的样本里，有多少被找出来了。” 也叫查全率。

F1：

```text
F1 = 2 · Precision · Recall / (Precision + Recall)
```

Precision 和 Recall 的调和平均，两者都高时 F1 才高。

用 Gaussian NB 的例子：

- 1 朵 virginica 被预测成 versicolor；
- 对 versicolor 来说，这是一个 **FP**：预测了 versicolor，但真实不是 → Precision = 9/10 = 0.909；
- 对 virginica 来说，这是一个 **FN**：真实是 virginica，但没预测出来 → Recall = 9/10 = 0.900；
- F1 分别为 0.952 和 0.947。

### 7.4 Macro 平均

多分类时，每个类别都算一套 Precision/Recall/F1，然后取平均：

```text
Macro Precision = (P_setosa + P_versicolor + P_virginica) / 3
```

Macro 平均不按样本数加权，每个类别权重一样。Iris 三类平衡，所以 macro 和 weighted 很接近；代码报告的是 macro。

### 7.5 ROC 和 AUC

ROC 曲线描述：当我们改变“判定为正类的阈值”时，TPR 和 FPR 怎么变化。

- TPR（True Positive Rate）= Recall = TP/(TP+FN)；
- FPR（False Positive Rate）= FP/(FP+TN)；
- 横轴 FPR，纵轴 TPR；
- 阈值从高到低移动，得到一条曲线；
- 曲线越靠近左上角越好。

AUC 是 ROC 曲线下的面积：

- AUC = 1.0：完美排序；
- AUC = 0.5：和随机猜差不多；
- AUC < 0.5：比随机还差（通常说明标签或得分方向反了）。

多分类时，代码用 **One-vs-Rest**：每次把一个类别当正类，其余两类当负类，画三条 ROC，再取平均得到 macro AUC。

本实验：

| 模型 | setosa AUC | versicolor AUC | virginica AUC | macro AUC |
|---|---:|---:|---:|---:|
| Logistic | 1.000 | 1.000 | 1.000 | 1.000 |
| Gaussian NB | 1.000 | 0.995 | 0.995 | 0.997 |
| Softmax | 1.000 | 1.000 | 1.000 | 1.000 |

注意：AUC 高不代表测试集准确率一定 100%，它衡量的是“得分排序”的能力；不过在这个简单数据上，两者都很高。

---

## 8. 结果解读

### 8.1 测试集结果

| 模型 | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| Logistic | 1.000 | 1.000 | 1.000 | 1.000 |
| Gaussian NB | 0.967 | 0.970 | 0.967 | 0.967 |
| Softmax | 1.000 | 1.000 | 1.000 | 1.000 |

- Logistic 和 Softmax 在 30 个测试样本上全对；
- Gaussian NB 错了 1 个 virginica；
- 三者差距很小，且测试集小，不能据此断言 Logistic/Softmax 永远更好。

### 8.2 为什么测试集全对，交叉验证却不是 100%

- 测试集只有 30 个样本，而且 Iris 本身比较容易分；
- 交叉验证每折只用 96 个样本训练，而且要在 5 折上平均；
- 交叉验证的 0.942、0.950 更能代表“在不同数据划分下的平均表现”；
- 测试集的 1.000 是这一次划分的结果，带有运气成分。

所以写报告时不要只写“准确率 100%”，要同时写交叉验证结果和小样本限制。

### 8.3 三个模型怎么比较

**Logistic 回归**

- 优点：简单、可解释、输出概率；
- 缺点：线性边界；One-vs-Rest 的概率需要额外归一化；
- 本实验：CV 0.942，测试 1.000。

**Gaussian 朴素贝叶斯**

- 优点：训练极快、小样本友好、天然输出概率；
- 缺点：特征条件独立假设太强；Iris 花瓣特征高度相关；
- 本实验：CV 0.950，测试 0.967，错 1 个 virginica。

**Softmax 回归**

- 优点：直接建模多分类概率，概率和为 1；
- 缺点：仍是线性模型；需要调学习率和 L2；
- 本实验：CV 0.950，测试 1.000。

结论：在这个数据上，三者都够用；Logistic/Softmax 的测试表现略好，但差距不足以忽略样本波动。

### 8.4 可以直接写进报告的一段话

> 本实验在 Iris 数据集上比较了 Logistic 回归、Gaussian 朴素贝叶斯和 Softmax 回归。数据按类别分层留出 80% 训练、20% 测试，特征按训练集均值与标准差标准化。三种模型均在训练集内用 5 折分层交叉验证选择参数：Logistic 选择学习率 0.5、L2=0，CV 准确率 0.942；Gaussian NB 选择方差平滑 1e-9，CV 准确率 0.950；Softmax 选择学习率 0.5、L2=0，CV 准确率 0.950。测试集上 Logistic 和 Softmax 的 Accuracy、Precision、Recall、F1 均为 1.000，Gaussian NB 为 0.967/0.970/0.967/0.967，后者将 1 个 virginica 错分为 versicolor。One-vs-Rest 宏平均 AUC 分别为 1.000、0.997、1.000。由于测试集只有 30 个样本，错 1 个就会带来约 3.3 个百分点的准确率变化，因此模型差异应结合交叉验证结果谨慎解读。

### 8.5 五张图分别看什么

#### `decision_boundaries.png`：决策区域

- 横轴花瓣长度，纵轴花瓣宽度；
- 背景颜色是模型预测的类别；
- 圆点是测试集真实类别；
- 其余两个特征固定在训练集均值，所以这是四维决策边界的一个二维切片；
- Logistic/Softmax 的边界是直线，Gaussian NB 的边界可以是曲线；
- 图中可以看到一个绿色点落在橙色区域附近，对应被错分的 virginica；但背景切片固定了花萼特征，测试点的实际分类还取决于未显示的两个花萼特征，切片位置只用于直观比较。

#### `confusion_matrices.png`：混淆矩阵

- 行是真实类别，列是预测类别；
- 对角线是分对的样本；
- Logistic/Softmax 都是对角 10；
- Gaussian NB 的 virginica 行、versicolor 列有 1。

#### `roc_curves.png`：ROC 曲线

- 每个面板一个模型，三条曲线对应三个 One-vs-Rest 类别；
- 虚线是随机猜测；
- 图例里是每个类别的 AUC；
- 标题里是宏平均 AUC；
- 越靠近左上角越好。

#### `parameter_selection.png`：参数选择

- 左 Logistic，中 Softmax，右 Gaussian NB；
- 纵轴 CV 准确率，误差棒是折间标准差；
- 红星是选中参数；
- 强 L2 或过大方差平滑会让准确率下降。

#### `metrics_comparison.png`：指标对比

- 每个模型四根柱子：Accuracy、Precision、Recall、F1；
- 本实验数值都很高，柱子高度接近；
- 不要只看柱子高度，要结合混淆矩阵和测试集大小。

---

## 9. 代码地图

### 9.1 文件结构

```text
task3/
├── __init__.py     # 包声明
├── data.py         # 读取、分层划分、类别编码、标准化
├── models.py       # 手动实现 Logistic / GaussianNB / Softmax
├── metrics.py      # 准确率、混淆矩阵、P/R/F1、ROC/AUC
├── validation.py   # 分层 K 折与参数选择
├── plots.py        # 五张图
└── main.py         # 实验主流程
```

模型部分没有调用 sklearn 的分类器；pandas 只用于表格处理。

### 9.2 运行

在仓库根目录运行：

```bash
pip install -r requirements.txt
python -m task3.main
```

运行时间约 15～20 秒，主要花在 Logistic/Softmax 的多组参数交叉验证上。

为什么用 `-m`？因为代码使用 `from task3.data import ...` 这种包内导入，`-m` 会把仓库根目录加入 Python 搜索路径；直接运行 `python task3/main.py` 可能报 `ModuleNotFoundError`。

### 9.3 `main.py` 的执行顺序

```text
1. 读取 iris.csv，删除行号列                 load_raw()
2. 按类别分层留出 120/30                     split_holdout()
3. 类别编码、特征标准化                       preprocess()
4. 5 折分层交叉验证选择 Logistic 参数          cross_validate()
5. 5 折分层交叉验证选择 Gaussian NB 参数       cross_validate()
6. 5 折分层交叉验证选择 Softmax 参数           cross_validate()
7. 用最优参数训练三个最终模型                  fit()
8. 测试集预测、指标、混淆矩阵                   predict / metrics
9. One-vs-Rest ROC/AUC                        predict_proba / roc_curve
10. 画图、保存 CSV 和 summary.md               plots / pandas
```

### 9.4 关键变量的形状

| 变量 | 形状 | 含义 |
|---|---|---|
| `raw` | 150 × 5 | `load_raw()` 后的表格（已删除行号列；CSV 文件本身 150 × 6） |
| `train_df` | 120 × 5 | 训练表格（去掉行号列） |
| `test_df` | 30 × 5 | 测试表格 |
| `X_train` | 120 × 4 | 标准化后的训练特征 |
| `X_test` | 30 × 4 | 标准化后的测试特征 |
| `y_train` | 120 | 训练标签（0/1/2） |
| `y_test` | 30 | 测试标签 |
| Logistic `coef_` | 3 × 4 | 三个类别各四个系数 |
| NB `means_` / `variances_` | 3 × 4 | 三个类别各四个特征的均值和方差 |
| Softmax `coef_` | 3 × 4 | 三个类别各四个系数 |
| `predictions["Logistic"]` | 30 | 测试集预测类别 |

### 9.5 输出文件

`results/task3/`：

- `metrics.csv`：三个模型的整体指标；
- `per_class_metrics.csv`：每类 Precision/Recall/F1；
- `roc_auc.csv`：每类 AUC；
- `confusion_matrices.csv`：混淆矩阵的长表；
- `cv_results.csv`、`cv_logistic.csv`、`cv_softmax.csv`、`cv_gaussian_nb.csv`：交叉验证结果；
- `decision_boundaries.png`、`confusion_matrices.png`、`roc_curves.png`、`parameter_selection.png`、`metrics_comparison.png`：五张图；
- `summary.md`：本次运行的结果摘要。

### 9.6 建议阅读顺序

1. `data.py`：数据怎么来、怎么划分；
2. `models.py`：三个模型怎么算；
3. `metrics.py`：指标怎么定义；
4. `validation.py`：参数怎么选；
5. `main.py`：整体流程；
6. `plots.py`：图怎么画。

---

## 10. 常见误区

1. **用测试集选参数。**  
   选学习率、L2、方差平滑只能用训练集内部的交叉验证。

2. **混淆矩阵行列看反。**  
   本代码的行是真实类别，列是预测类别。Gaussian NB 的 `真实 virginica / 预测 versicolor = 1` 表示一朵 virginica 被错分成 versicolor。

3. **只看 Accuracy。**  
   Accuracy 在不平衡数据上会骗人；还要看每类 Precision/Recall/F1。Iris 虽然平衡，但这是好习惯。

4. **把 Precision 和 Recall 搞反。**  
   Precision 的分母是“预测为正类的数量”；Recall 的分母是“真实为正类的数量”。

5. **认为 Macro 和 Weighted 一样。**  
   类别不平衡时两者可能差很多；本实验三类平衡，所以接近。

6. **用预测标签画 ROC。**  
   ROC 需要连续的得分/概率，不是 0/1 标签。代码用 `predict_proba()` 的得分。

7. **看到测试集 100% 就认为模型完美。**  
   测试集只有 30 个样本，错 1 个就是 96.7%；交叉验证才是更稳定的参考。

8. **认为 One-vs-Rest 的三个概率天然和为 1。**  
   三个独立 sigmoid 不一定和为 1；代码做了归一化，方便 ROC/AUC，且不改变 argmax。

9. **认为朴素贝叶斯“假设独立”就代表特征真的独立。**  
   这只是模型假设；Iris 的花瓣长度和宽度明显相关，但 NB 仍然能工作。

10. **把 `var_smoothing` 设得很大。**  
    过大的平滑会抹掉类别差异，导致欠拟合；本实验 0.1、1.0 的 CV 准确率明显下降。

11. **不标准化就做梯度下降。**  
    特征尺度差异会让不同方向的更新步长失衡；标准化后更稳定。

12. **认为 Logistic 回归只能做二分类。**  
    通过 One-vs-Rest 或 Softmax，它可以处理多分类。

13. **把 Logistic 回归当成回归。**  
    它名字里有 regression，但输出是类别概率，属于分类模型。

14. **忽略数据泄漏。**  
    标准化、缺失值填补、类别编码的统计量都只能来自训练集。

---

## 11. 动手练习

改完代码后重新运行 `python -m task3.main`，观察变化。

1. **换随机种子**  
   把 `SEED = 42` 改成 0 或 1。  
   观察：测试集准确率会变吗？Gaussian NB 还会错那一朵吗？

2. **改变测试集比例**  
   把 `TEST_SIZE = 0.2` 改成 0.3。  
   观察：测试集变成 45 朵后，指标是否更稳定？

3. **关掉分层**  
   把 `split_holdout(raw, test_size=TEST_SIZE, seed=SEED)` 改成普通随机划分（可以临时给函数加 `stratify=False`）。  
   观察：测试集类别比例是否还接近 1:1:1？

4. **改变学习率**  
   把 Logistic 的 `learning_rate` 固定为 0.01、0.1、0.5、1.0，观察收敛和准确率。  
   思考：学习率太大或太小分别会怎样？

5. **改变 L2**  
   把 Logistic 或 Softmax 的 `l2` 固定为 0、0.01、0.1、1.0。  
   观察：强 L2 是否让训练集和测试集准确率都下降？

6. **改变方差平滑**  
   把 Gaussian NB 的 `var_smoothing` 设为 1e-9、1e-3、0.1、1.0。  
   观察：过大的平滑为什么会让准确率下降？

7. **去掉标准化**  
   暂时注释 `preprocess()` 中的标准化，重新训练 Logistic/Softmax。  
   观察：收敛速度和准确率有什么变化？正式结果要保留标准化。

8. **只用两个特征**  
   只用 `Petal.Length` 和 `Petal.Width` 训练三个模型。  
   观察：测试准确率下降多少？这和 `decision_boundaries.png` 的二维切片有什么关系？

9. **增加交叉验证折数**  
   把 `CV_FOLDS` 从 5 改成 10。  
   观察：CV 准确率和标准差有什么变化？每折训练样本变多还是变少？

10. **换评价指标选参数**  
    目前按 CV 准确率选参数；改成按 macro F1 选，结果会变吗？  
    思考：类别不平衡时，为什么 F1 可能更合适？

---

## 12. 术语表

| 术语 | 含义 |
|---|---|
| 分类 | 输出是离散类别的监督学习 |
| 类别 / 标签 | 要预测的离散值 |
| 特征 | 输入变量，如花萼长度 |
| 训练集 / 测试集 | 拟合参数的数据 / 最终评价的数据 |
| 泛化 | 模型在未见数据上的表现 |
| 过拟合 / 欠拟合 | 训练集好但新数据差 / 模型太简单两头都差 |
| 标准化 | 把特征变成均值 0、标准差 1 |
| 分层抽样 | 按类别比例划分数据 |
| Logistic 回归 | 用 sigmoid 输出二分类概率的线性分类器 |
| Sigmoid | `1/(1+exp(-z))`，把实数压到 0～1 |
| 决策边界 | 模型分界的位置；Logistic 是线性的 |
| One-vs-Rest | 每个类别训练一个“该类 vs 其余”的二分类器 |
| Softmax | 把多个得分变成和为 1 的概率 |
| 交叉熵 | 分类任务常用的损失函数 |
| 梯度下降 | 沿负梯度方向迭代更新参数 |
| 学习率 | 梯度下降每次更新的步长 |
| L2 正则化 | 惩罚系数平方和，抑制过大系数 |
| 先验 / 似然 / 后验 | 贝叶斯公式中的三个概率 |
| 条件独立 | 已知类别时假设特征互不影响 |
| 方差平滑 | 给方差加小量，避免除以 0 和过拟合 |
| 混淆矩阵 | 行真实、列预测的计数表 |
| TP / FP / FN / TN | 真正例 / 假正例 / 假负例 / 真负例 |
| Precision | 预测为正的样本中真正为正的比例 |
| Recall | 真实为正的样本中被找出的比例 |
| F1 | Precision 和 Recall 的调和平均 |
| Macro 平均 | 每个类别等权平均 |
| ROC | TPR 随 FPR 变化的曲线 |
| AUC | ROC 曲线下面积 |
| 交叉验证 | 把训练数据轮流切分，估计泛化误差 |

---

## 13. 自测：你真的看懂了吗

<details>
<summary>1. 为什么 Logistic 回归叫“回归”，却是分类模型？</summary>

它建模的是对数几率 `log(p/(1-p))` 与特征的线性关系，名字来自“logistic regression”；但输出是类别概率，最终预测离散类别，所以属于分类模型。

</details>

<details>
<summary>2. Logistic 回归的 sigmoid 在做什么？</summary>

把线性得分 `z` 压到 0～1：`sigmoid(z)=1/(1+exp(-z))`。`z=0` 时概率 0.5，`z` 越大越接近 1，越小越接近 0。

</details>

<details>
<summary>3. One-vs-Rest 和 Softmax 有什么区别？</summary>

One-vs-Rest 为每个类别训练一个二分类器，得到多个独立得分，再取最大或归一化；Softmax 用一个多分类模型直接输出和为 1 的概率。两者在本实验里都是线性分类器。

</details>

<details>
<summary>4. Gaussian NB 的“朴素”指什么？</summary>

指在已知类别的条件下假设各特征相互独立。这样联合似然可以拆成每个特征似然的乘积，计算简单，但 Iris 的花瓣特征其实相关，所以这是近似假设。

</details>

<details>
<summary>5. 为什么 `var_smoothing` 太大会变差？</summary>

它给方差加一个较大的量，抹平了类别内部特征的分布差异，模型更难区分类别，导致欠拟合。本实验 0.1、1.0 的 CV 准确率明显低于 1e-9～1e-3。

</details>

<details>
<summary>6. 为什么测试集全对，交叉验证却不是 100%？</summary>

测试集只有 30 个样本且 Iris 容易分；交叉验证每折用 96 个样本训练并在 5 折上平均，反映不同划分下的平均表现。测试集的 100% 带有这次划分的运气成分。

</details>

<details>
<summary>7. Precision 和 Recall 的区别？</summary>

Precision = TP/(TP+FP)，看“预测为正的里面有多少是真的”；Recall = TP/(TP+FN)，看“真实为正的里面有多少被找出来”。Gaussian NB 的 1 个 virginica→versicolor 错误会同时降低 versicolor 的 Precision 和 virginica 的 Recall。

</details>

<details>
<summary>8. AUC 是什么？为什么多分类要 One-vs-Rest？</summary>

AUC 是 ROC 曲线下面积，衡量得分把正类排在负类前面的能力。多分类没有单一正负类，所以每次把一个类别当正类、其余当负类，分别算 AUC，再取平均（macro AUC）。

</details>

---

## 14. 一句话总结

> Logistic 回归用 sigmoid 把线性得分变成二分类概率，再用 One-vs-Rest 处理多分类；Gaussian NB 用贝叶斯公式和高斯假设算概率；Softmax 直接输出多分类概率。本实验三个模型都很强，Logistic/Softmax 在 30 个测试样本上全对，Gaussian NB 只错 1 个；但测试集太小，真正可靠的比较要结合 5 折交叉验证和混淆矩阵。
