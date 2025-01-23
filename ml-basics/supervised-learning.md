---
title: "监督学习基础"
slug: "supervised-learning"
sequence: 1
description: "掌握监督学习的核心概念、算法原理和实践应用"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

![Supervised Learning](images/supervised-learning-header.png)
*监督学习是机器学习最常用的范式之一*

# 监督学习基础

## 学习目标
完成本模块学习后，你将能够：
- 理解监督学习的基本概念和原理
- 掌握常见的监督学习算法
- 学会评估和优化模型性能
- 解决实际的监督学习问题

## 先修知识
- Python编程基础
- 数学基础(线性代数、微积分、概率统计)
- 数据分析基础

## 1. 监督学习概述

### 1.1 基本概念
监督学习是机器学习的主要范式之一，其核心思想是通过已标记的训练数据来学习输入到输出的映射关系。在这个过程中：
- 算法通过观察训练样本(特征和标签)来学习模式
- 学习到的模式用于预测新样本的标签
- 预测结果与实际标签的差异用于指导模型改进

### 1.2 应用场景
```python
# 示例：房价预测
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
df = pd.read_csv('house_prices.csv')
X = df[['size', 'bedrooms', 'location']]
y = df['price']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'均方误差: {mse:.2f}')
print(f'R2分数: {r2:.2f}')
```

## 2. 经典监督学习算法

### 2.1 线性回归
线性回归是最基础的监督学习算法，用于建立特征和目标变量之间的线性关系。

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 创建Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# 训练模型
pipeline.fit(X_train, y_train)

# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(pipeline.named_steps['regressor'].coef_)
})
print("特征重要性:")
print(feature_importance.sort_values('importance', ascending=False))
```

### 2.2 逻辑回归
逻辑回归是一种用于分类问题的监督学习算法。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 创建和训练模型
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# 预测和评估
y_pred = clf.predict(X_test)
print("分类报告:")
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, 
            fmt='d',
            cmap='Blues')
plt.title('混淆矩阵')
plt.show()
```

### 2.3 决策树
决策树是一种直观且易于解释的监督学习算法。

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 创建和训练模型
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# 可视化决策树
plt.figure(figsize=(20,10))
plot_tree(dt, 
         feature_names=X.columns,
         class_names=['0', '1'],
         filled=True,
         rounded=True)
plt.title('决策树可视化')
plt.show()

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
})
print("\n特征重要性:")
print(feature_importance.sort_values('importance', ascending=False))
```

## 3. 模型评估与优化

### 3.1 评估指标
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

def evaluate_model(y_true, y_pred, y_prob=None):
    """评估分类模型性能"""
    print("模型评估指标:")
    print(f"准确率: {accuracy_score(y_true, y_pred):.3f}")
    print(f"精确率: {precision_score(y_true, y_pred):.3f}")
    print(f"召回率: {recall_score(y_true, y_pred):.3f}")
    print(f"F1分数: {f1_score(y_true, y_pred):.3f}")
    
    if y_prob is not None:
        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('接收者操作特征(ROC)曲线')
        plt.legend(loc="lower right")
        plt.show()
```

### 3.2 交叉验证
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5):
    """绘制学习曲线"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    # 计算平均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='训练集得分')
    plt.plot(train_sizes, test_mean, label='验证集得分')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1)
    plt.xlabel('训练样本数')
    plt.ylabel('得分')
    plt.title('学习曲线')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

### 3.3 超参数调优
```python
from sklearn.model_selection import GridSearchCV

def optimize_model(model, param_grid, X, y, cv=5):
    """使用网格搜索优化模型超参数"""
    grid_search = GridSearchCV(model, param_grid, cv=cv, 
                             scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    print("最佳参数:", grid_search.best_params_)
    print("最佳得分:", grid_search.best_score_)
    
    # 参数重要性可视化
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='param_max_depth', y='mean_test_score', data=results)
    plt.xlabel('最大深度')
    plt.ylabel('交叉验证得分')
    plt.title('不同最大深度的模型性能')
    plt.show()
    
    return grid_search.best_estimator_
```

## 4. 实战案例：信用卡欺诈检测

### 4.1 数据准备
```python
# 加载数据
df = pd.read_csv('credit_card_fraud.csv')

# 处理类别不平衡
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 特征工程
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
```

### 4.2 模型训练与评估
```python
# 创建模型
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# 优化模型
best_model = optimize_model(rf, param_grid, X_scaled, y_resampled)

# 预测和评估
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 评估模型性能
evaluate_model(y_test, y_pred, y_prob)
```

## 常见问题解答

Q: 如何处理类别不平衡问题？
A: 可以使用过采样(SMOTE)、欠采样或调整类别权重等方法。选择方法时需要考虑数据规模和业务需求。

Q: 如何选择合适的评估指标？
A: 根据问题类型和业务目标选择。比如在欺诈检测中，召回率可能比准确率更重要；在医疗诊断中，精确率可能更重要。

Q: 如何避免过拟合？
A: 可以使用交叉验证、正则化、早停等技术。同时，确保训练数据的质量和代表性也很重要。

## 扩展阅读
- [统计学习方法](https://book.douban.com/subject/33437381/)
- [机器学习实战](https://book.douban.com/subject/24703171/)
- [Scikit-learn官方文档](https://scikit-learn.org/stable/)
