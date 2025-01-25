# 逻辑回归详解

逻辑回归（Logistic Regression）是最基础也最重要的分类算法之一。虽然名字中有"回归"，但它实际上是一个分类模型。本章我们将深入探讨逻辑回归的原理、实现和应用。

## 1. 逻辑回归原理

### 1.1 从线性回归到逻辑回归

想象你是一个垃圾邮件分类系统的设计者：
- 如果用线性回归，输出可能是任何数字
- 但我们需要的是一个0到1之间的概率
- 这就是为什么我们需要逻辑回归！

### 1.2 Sigmoid函数

逻辑回归使用Sigmoid函数将线性输出转换为概率：

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

![Sigmoid函数](./images/classification/sigmoid.png)
*Sigmoid函数将任意实数映射到(0,1)区间*

### 1.3 数学原理

1. **模型表达式**：
   ```
   P(y=1|x) = sigmoid(w^T x + b)
   ```

2. **损失函数**（对数似然损失）：
   ```
   L(w) = -[y log(h(x)) + (1-y)log(1-h(x))]
   ```

3. **梯度下降更新**：
   ```
   w := w - α∇L(w)
   ```

## 2. 逻辑回归实现

### 2.1 从零实现逻辑回归

让我们实现一个简单的逻辑回归分类器：

```python
class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # 添加偏置项
        X = np.insert(X, 0, 1, axis=1)
        
        # 初始化参数
        self.theta = np.zeros(X.shape[1])
        
        # 梯度下降
        for _ in range(self.n_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            
            # 计算梯度
            gradient = np.dot(X.T, (h - y)) / y.size
            
            # 更新参数
            self.theta -= self.learning_rate * gradient
    
    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold
```

### 2.2 使用scikit-learn实现

在实际应用中，我们通常使用scikit-learn的实现：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 准备数据
X = [[2, 1],   # 短文本，少链接 -> 正常邮件
     [3, 1],   # 短文本，少链接 -> 正常邮件
     [8, 4],   # 长文本，多链接 -> 垃圾邮件
     [9, 5]]   # 长文本，多链接 -> 垃圾邮件
y = [0, 0, 1, 1]  # 0: 正常邮件, 1: 垃圾邮件

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 创建并训练模型
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 评估模型
print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
```

## 3. 逻辑回归的优化

### 3.1 正则化

防止过拟合的三种正则化方法：

1. **L1正则化**（Lasso）
```python
model = LogisticRegression(penalty='l1', solver='liblinear')
```

2. **L2正则化**（Ridge）
```python
model = LogisticRegression(penalty='l2')  # 默认选项
```

3. **弹性网络**（Elastic Net）
```python
model = LogisticRegression(penalty='elasticnet', 
                          l1_ratio=0.5, 
                          solver='saga')
```

### 3.2 特征工程

1. **特征缩放**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 或者归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

2. **特征交叉**
```python
from sklearn.preprocessing import PolynomialFeatures

# 创建交叉特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### 3.3 类别不平衡处理

1. **调整类别权重**
```python
# 自动平衡权重
model = LogisticRegression(class_weight='balanced')

# 手动设置权重
model = LogisticRegression(class_weight={0:1, 1:3})
```

2. **重采样**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 过采样
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 欠采样
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X, y)
```

## 4. 高级应用

### 4.1 多类别分类

逻辑回归可以通过两种方式处理多类别问题：

1. **一对多（One-vs-Rest）**
```python
from sklearn.linear_model import LogisticRegression

# 自动使用OvR策略
model = LogisticRegression(multi_class='ovr')
```

2. **多项式（Multinomial）**
```python
# 使用多项式逻辑回归
model = LogisticRegression(multi_class='multinomial')
```

### 4.2 概率校准

有时我们需要更准确的概率输出：

```python
from sklearn.calibration import CalibratedClassifierCV

# 创建基础模型
base_model = LogisticRegression()

# 创建校准模型
calibrated_model = CalibratedClassifierCV(
    base_model, 
    cv=5, 
    method='sigmoid'
)

# 训练校准模型
calibrated_model.fit(X_train, y_train)

# 获取校准后的概率
calibrated_probs = calibrated_model.predict_proba(X_test)
```

### 4.3 在线学习

处理大规模数据时的增量学习：

```python
from sklearn.linear_model import SGDClassifier

# 创建随机梯度下降分类器
model = SGDClassifier(
    loss='log',           # 使用逻辑回归损失
    learning_rate='optimal',
    max_iter=1000
)

# 批量学习
for i in range(n_batches):
    X_batch = X[i*batch_size:(i+1)*batch_size]
    y_batch = y[i*batch_size:(i+1)*batch_size]
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
```

## 5. 实战案例：信用卡欺诈检测

让我们用逻辑回归实现一个信用卡欺诈检测系统：

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 加载数据
data = pd.read_csv('credit_card_data.csv')
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 处理类别不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, 
    test_size=0.2, 
    random_state=42
)

# 创建并训练模型
model = LogisticRegression(
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 评估模型
print("分类报告:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model.coef_[0])
})
print("\n最重要的特征:")
print(feature_importance.sort_values('importance', 
                                   ascending=False).head())
```

## 6. 常见问题与解决方案

### 6.1 过拟合

问题：模型在训练集表现很好，但在测试集表现差。

解决方案：
1. 增加正则化强度
2. 减少特征数量
3. 收集更多数据
4. 使用交叉验证

### 6.2 特征共线性

问题：特征之间高度相关导致模型不稳定。

解决方案：
1. 使用L1正则化
2. 使用主成分分析（PCA）
3. 删除高度相关的特征
4. 合并相关特征

### 6.3 类别不平衡

问题：某些类别的样本数量远多于其他类别。

解决方案：
1. 使用类别权重
2. 过采样（SMOTE）
3. 欠采样
4. 使用其他评估指标（如F1分数）

## 7. 实用小贴士

1. **数据预处理**
   - 处理缺失值
   - 特征缩放
   - 处理类别特征
   - 检测和处理异常值

2. **特征选择**
   - 使用L1正则化
   - 使用特征重要性
   - 使用领域知识
   - 考虑特征交互

3. **模型调优**
   - 使用网格搜索找最佳参数
   - 使用交叉验证评估模型
   - 尝试不同的正则化方法
   - 监控训练过程

4. **模型解释**
   - 分析特征系数
   - 使用SHAP值
   - 绘制ROC曲线
   - 分析混淆矩阵

## 8. 总结

逻辑回归是一个简单但强大的分类算法，它的优点是：
- 简单易解释
- 训练速度快
- 预测速度快
- 可以输出概率

但也有一些局限性：
- 只能学习线性决策边界
- 对异常值敏感
- 需要处理特征之间的相关性
- 可能需要大量特征工程

选择是否使用逻辑回归，需要考虑：
- 是否需要概率输出
- 是否需要模型可解释性
- 数据是否大致呈线性可分
- 计算资源是否受限

在这些场景下，逻辑回归往往是一个很好的选择：
- 风险评估（如信用评分）
- 疾病诊断
- 垃圾邮件检测
- 用户行为预测
