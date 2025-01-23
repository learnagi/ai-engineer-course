---
title: "高级回归方法"
slug: "advanced-regression"
sequence: 2
description: "深入学习高级回归技术，包括正则化方法、非线性回归和集成方法"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

![Advanced Regression](images/advanced-regression-header.png)
*掌握高级回归方法，应对复杂预测问题*

# 高级回归方法

## 学习目标
完成本模块学习后，你将能够：
- 理解并实现各种高级回归算法
- 掌握正则化技术来防止过拟合
- 处理非线性回归问题
- 使用集成方法提高回归性能

## 先修知识
- Python编程基础
- 线性代数基础
- 基本统计概念
- 机器学习基础

## 1. 从零实现线性回归

### 1.1 基础理论
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

### 1.2 可视化与分析
```python
def plot_regression_line(X, y, model):
    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
```

## 2. 高级回归方法

### 2.1 岭回归（L2正则化）
```python
from sklearn.linear_model import Ridge

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
```

### 2.2 Lasso回归（L1正则化）
```python
from sklearn.linear_model import Lasso

class LassoRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Lasso(alpha=self.alpha)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
```

### 2.3 弹性网络
```python
from sklearn.linear_model import ElasticNet

class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
```

## 3. 正则化技术

### 3.1 L1正则化（Lasso）
- 特点：产生稀疏解
- 用途：特征选择
- 数学表达式：
```python
def l1_penalty(weights, alpha):
    """计算L1正则化项"""
    return alpha * np.sum(np.abs(weights))
```

### 3.2 L2正则化（Ridge）
- 特点：防止过拟合
- 用途：处理多重共线性
- 数学表达式：
```python
def l2_penalty(weights, alpha):
    """计算L2正则化项"""
    return alpha * np.sum(weights ** 2)
```

### 3.3 正则化参数选择
```python
from sklearn.model_selection import GridSearchCV

def find_best_alpha(X, y, model_class):
    """网格搜索找最佳正则化参数"""
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model_class(), param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_params_['alpha']
```

## 4. 非线性回归

### 4.1 多项式回归
```python
from sklearn.preprocessing import PolynomialFeatures

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)
        self.linear_regression = LinearRegression()
        
    def fit(self, X, y):
        X_poly = self.poly_features.fit_transform(X)
        self.linear_regression.fit(X_poly, y)
        
    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return self.linear_regression.predict(X_poly)
```

### 4.2 样条回归
```python
from scipy.interpolate import make_interp_spline

def spline_regression(X, y, n_knots=5):
    """使用样条进行非线性回归"""
    spl = make_interp_spline(X.flatten(), y, k=3)
    X_new = np.linspace(X.min(), X.max(), 200)
    y_new = spl(X_new)
    return X_new, y_new
```

## 5. 实战案例：房价预测

### 5.1 数据准备
```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def prepare_housing_data():
    """准备加州房价数据集"""
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
```

### 5.2 模型比较
```python
def compare_regression_models(X, y):
    """比较不同回归模型的性能"""
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }
    
    results = {}
    for name, model in models.items():
        # 交叉验证评分
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        results[name] = rmse
        
    return results
```

## 常见问题解答

Q: 如何选择合适的正则化方法？
A: 根据问题特点选择：
- L1正则化：当需要特征选择时
- L2正则化：当处理多重共线性时
- 弹性网络：当两种效果都需要时

Q: 如何处理非线性关系？
A: 可以使用以下方法：
- 多项式回归
- 样条回归
- 核方法
- 神经网络

Q: 如何避免过拟合？
A: 可以采用以下策略：
- 使用正则化
- 减少模型复杂度
- 增加训练数据
- 使用交叉验证

## 扩展阅读
- [《Statistical Learning with Sparsity》](https://web.stanford.edu/~hastie/StatLearnSparsity/)
- [《Elements of Statistical Learning》](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Scikit-learn Regression Tutorial](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
