# 监督学习算法详解

## 线性回归

### 1. 理论基础
- **模型定义**
  - 数学表达式
  - 损失函数
  - 优化目标
- **假设条件**
  - 线性性
  - 独立性
  - 同方差性

### 2. 实现方法
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成示例数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 4.0, 6.3, 8.0, 9.9])

# 创建并训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 评估模型
print(f'系数: {model.coef_[0]:.2f}')
print(f'截距: {model.intercept_:.2f}')
print(f'R2分数: {r2_score(y, y_pred):.2f}')
```

### 3. 正则化方法
- **Ridge回归**
  - L2正则化
  - 参数调优
- **Lasso回归**
  - L1正则化
  - 特征选择
- **弹性网络**
  - 混合正则化
  - 应用场景

## 逻辑回归

### 1. 基本原理
- **sigmoid函数**
- **决策边界**
- **最大似然估计**

### 2. 实现示例
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成分类数据
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 评估模型
print(f'准确率: {clf.score(X_test, y_test):.2f}')
```

### 3. 多分类扩展
- **一对多方法**
- **一对一方法**
- **多项式逻辑回归**

## 决策树

### 1. 决策树构建
- **特征选择**
  - 信息增益
  - 基尼系数
  - 增益比
- **树的生长**
  - 分裂标准
  - 停止条件

### 2. 实现示例
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 创建决策树
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# 可视化决策树
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=[f'特征{i+1}' for i in range(X.shape[1])])
plt.show()
```

### 3. 剪枝技术
- **预剪枝**
  - 深度限制
  - 样本数限制
- **后剪枝**
  - 错误率降低剪枝
  - 代价复杂度剪枝

## 支持向量机(SVM)

### 1. 理论基础
- **最大间隔**
- **核函数**
- **软间隔**

### 2. 实现示例
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练SVM
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled, y_train)

# 评估模型
print(f'准确率: {svm.score(X_test_scaled, y_test):.2f}')
```

### 3. 核函数选择
- **线性核**
- **多项式核**
- **RBF核**
- **sigmoid核**

## 集成学习

### 1. Bagging方法
- **随机森林**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 特征重要性
importances = rf.feature_importances_
```

### 2. Boosting方法
- **AdaBoost**
- **梯度提升树**
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
```

### 3. Stacking
- **基学习器选择**
- **元学习器设计**

## 模型评估与选择

### 1. 评估指标
- **分类问题**
  - 准确率
  - 精确率
  - 召回率
  - F1分数
- **回归问题**
  - MSE
  - MAE
  - R2分数

### 2. 交叉验证
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5)
print(f'交叉验证分数: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')
```

### 3. 超参数调优
```python
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svm = SVC()
clf = GridSearchCV(svm, parameters)
clf.fit(X_train, y_train)
```

## 实战项目：信用卡欺诈检测

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 加载数据
df = pd.read_csv('creditcard.csv')

# 数据预处理
X = df.drop('Class', axis=1)
y = df['Class']

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测和评估
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 课后练习

1. 实现简单的线性回归和逻辑回归算法
2. 使用不同的核函数训练SVM，比较性能
3. 构建决策树并进行剪枝
4. 实现一个简单的集成学习系统

## 延伸阅读

1. Trevor Hastie, Robert Tibshirani, Jerome Friedman. The Elements of Statistical Learning
2. Richard O. Duda, Peter E. Hart, David G. Stork. Pattern Classification
3. Christopher M. Bishop. Pattern Recognition and Machine Learning

## 下一步学习

- 深入学习非监督学习算法
- 探索深度学习模型
- 实践更复杂的机器学习项目
- 参与Kaggle竞赛