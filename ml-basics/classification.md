---
title: "分类算法详解"
slug: "classification"
description: "深入理解常用分类算法的原理、实现和应用"
is_published: true
estimated_minutes: 60
language: "zh-CN"
---

![分类算法](assets/images/ml-basics/classification-header.png)
*分类是机器学习中最常见的任务之一，需要掌握多种算法的特点和应用场景*

# 分类算法详解

## 学习目标
完成本节后，你将能够：
- 理解主要分类算法的原理和特点
- 实现和使用各种分类器
- 选择合适的分类算法
- 评估分类模型的性能
- 处理不平衡数据集问题

## 先修知识
学习本节内容需要：
- Python编程基础
- 机器学习基础概念
- 数学和统计学基础
- 数据预处理知识

## 逻辑回归

### 原理与实现
逻辑回归是最基础的分类算法之一，常用于二分类问题。

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                         random_state=42)

# 创建和训练模型
model = LogisticRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)
y_prob = model.predict_proba(X)
```

### 决策边界可视化
```python
def plot_decision_boundary(X, y, model, title="决策边界"):
    """
    绘制二维数据的决策边界
    """
    import matplotlib.pyplot as plt
    
    h = .02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("特征1")
    plt.ylabel("特征2")
    plt.title(title)
    plt.show()
```

## K近邻算法(KNN)

### 原理与实现
KNN是一种基于实例的学习算法，通过计算样本间距离进行分类。

```python
from sklearn.neighbors import KNeighborsClassifier

# 创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# 预测
y_pred_knn = knn.predict(X)
```

### 选择最优K值
```python
from sklearn.model_selection import cross_val_score

def find_best_k(X, y, k_range):
    """
    通过交叉验证选择最优的K值
    """
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5)
        k_scores.append(scores.mean())
    
    return k_scores

# 测试不同的K值
k_range = range(1, 31)
k_scores = find_best_k(X, y, k_range)

# 可视化结果
plt.plot(k_range, k_scores)
plt.xlabel('K值')
plt.ylabel('交叉验证得分')
plt.title('不同K值的模型性能')
plt.show()
```

## 支持向量机(SVM)

### 线性SVM
```python
from sklearn.svm import SVC

# 创建线性SVM
svm_linear = SVC(kernel='linear')
svm_linear.fit(X, y)

# 预测
y_pred_svm = svm_linear.predict(X)
```

### 核方法
```python
# RBF核SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X, y)

# 多项式核SVM
svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X, y)
```

## 决策树

### 决策树分类器
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# 创建决策树
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X, y)

# 可视化决策树
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=[f'feature_{i}' for i in range(X.shape[1])])
plt.show()
```

### 特征重要性
```python
def plot_feature_importance(model, feature_names):
    """
    可视化特征重要性
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10,6))
    plt.title("特征重要性")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.show()
```

## 集成方法

### 随机森林
```python
from sklearn.ensemble import RandomForestClassifier

# 创建随机森林
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(X, y)

# 预测
y_pred_rf = rf.predict(X)
```

### 梯度提升
```python
from sklearn.ensemble import GradientBoostingClassifier

# 创建梯度提升分类器
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(X, y)

# 预测
y_pred_gb = gb.predict(X)
```

## 模型评估

### 评估指标
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_classifier(y_true, y_pred):
    """
    评估分类器性能
    """
    print("准确率:", accuracy_score(y_true, y_pred))
    print("精确率:", precision_score(y_true, y_pred))
    print("召回率:", recall_score(y_true, y_pred))
    print("F1分数:", f1_score(y_true, y_pred))
    print("\n混淆矩阵:\n", confusion_matrix(y_true, y_pred))
    print("\n分类报告:\n", classification_report(y_true, y_pred))
```

### ROC曲线
```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_prob):
    """
    绘制ROC曲线
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('接收者操作特征曲线')
    plt.legend(loc="lower right")
    plt.show()
```

## 处理不平衡数据

### 重采样方法
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# SMOTE过采样
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 随机欠采样
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X, y)

# 结合过采样和欠采样
pipeline = Pipeline([
    ('smote', SMOTE()),
    ('rus', RandomUnderSampler())
])
X_resampled, y_resampled = pipeline.fit_resample(X, y)
```

### 类别权重
```python
# 使用类别权重
weighted_model = LogisticRegression(class_weight='balanced')
weighted_model.fit(X, y)
```

## 实战项目：信用卡欺诈检测

### 数据准备
```python
# 加载数据
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, n_features=20,
                         n_classes=2, weights=[0.99, 0.01],
                         random_state=42)

# 数据分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

### 模型训练与评估
```python
# 创建模型管道
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测和评估
y_pred = pipeline.predict(X_test)
evaluate_classifier(y_test, y_pred)
```

## 练习与作业
1. 基础练习：
   - 实现逻辑回归的梯度下降
   - 使用不同核函数的SVM
   - 构建和可视化决策树

2. 进阶练习：
   - 实现交叉验证和网格搜索
   - 处理不平衡数据集
   - 比较不同分类器的性能

3. 项目实践：
   - 选择一个真实数据集进行分类
   - 实现完整的分类流程
   - 尝试不同的模型优化方法

## 常见问题
Q1: 如何选择合适的分类算法？
A1: 需要考虑以下因素：
- 数据规模和维度
- 特征的线性可分性
- 计算资源限制
- 模型可解释性需求
- 预测速度要求

Q2: 如何处理过拟合问题？
A2: 可以采用以下方法：
- 增加训练数据
- 使用正则化
- 减少模型复杂度
- 使用集成方法
- 特征选择

## 扩展阅读
- [scikit-learn分类算法指南](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [不平衡学习](https://imbalanced-learn.org/stable/)
- [集成学习方法](https://scikit-learn.org/stable/modules/ensemble.html)

## 下一步学习
- 深度学习分类
- 序列分类
- 多标签分类
- 半监督学习
