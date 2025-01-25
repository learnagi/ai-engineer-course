---
title: "机器学习模型优化指南"
slug: "model-optimization"
sequence: 5
description: "全面介绍机器学习模型优化技术，包括正则化、超参数调优、模型集成等方法"
is_published: true
estimated_minutes: 90
language: "zh-CN"
---

# 机器学习模型优化指南

## 学习目标
完成本章学习后，你将能够：
- 理解过拟合和欠拟合的概念
- 掌握常用的正则化技术
- 熟练使用超参数优化方法
- 实践模型集成技术

## 1. 模型优化基础

### 1.1 过拟合与欠拟合
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def plot_model_complexity():
    """可视化不同复杂度模型的拟合效果"""
    # 生成数据
    np.random.seed(0)
    X = np.sort(np.random.rand(30))
    y = np.cos(1.5 * np.pi * X) + np.random.normal(0, 0.1, 30)
    X = X.reshape(-1, 1)
    
    # 尝试不同程度的多项式
    degrees = [1, 4, 15]  # 分别对应欠拟合、合适、过拟合
    plt.figure(figsize=(15, 5))
    
    for i, degree in enumerate(degrees):
        plt.subplot(1, 3, i + 1)
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression)
        ])
        pipeline.fit(X, y)
        
        # 绘制结果
        X_test = np.linspace(0, 1, 100).reshape(-1, 1)
        plt.scatter(X, y, label='Training points')
        plt.plot(X_test, pipeline.predict(X_test), label=f'Degree {degree}')
        plt.legend()
        plt.title(f'Polynomial Degree {degree}')
    
    plt.tight_layout()
    plt.show()
```

## 2. 正则化技术

### 2.1 L1正则化（Lasso）
```python
from sklearn.linear_model import Lasso
import numpy as np

def lasso_regression_example():
    """L1正则化示例"""
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(100, 20)
    true_coef = np.zeros(20)
    true_coef[:5] = np.random.randn(5)
    y = np.dot(X, true_coef) + np.random.randn(100) * 0.1
    
    # 使用不同的alpha值
    alphas = [0.0001, 0.001, 0.01, 0.1, 1]
    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        print(f'Alpha: {alpha}')
        print(f'Number of non-zero coefficients: {np.sum(lasso.coef_ != 0)}')
        print(f'R2 score: {lasso.score(X, y)}\n')
```

### 2.2 L2正则化（Ridge）
```python
from sklearn.linear_model import Ridge

def ridge_regression_example():
    """L2正则化示例"""
    # 使用不同的alpha值
    alphas = [0.0001, 0.001, 0.01, 0.1, 1]
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        print(f'Alpha: {alpha}')
        print(f'Coefficient norm: {np.linalg.norm(ridge.coef_)}')
        print(f'R2 score: {ridge.score(X, y)}\n')
```

### 2.3 弹性网络
```python
from sklearn.linear_model import ElasticNet

def elastic_net_example():
    """弹性网络示例"""
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic.fit(X, y)
    print('Number of non-zero coefficients:', np.sum(elastic.coef_ != 0))
    print('R2 score:', elastic.score(X, y))
```

### 2.4 Dropout（深度学习）
```python
import tensorflow as tf

def create_model_with_dropout():
    """创建带有Dropout的神经网络"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    return model
```

## 3. 超参数优化

### 3.1 网格搜索
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def grid_search_example(X, y):
    """网格搜索示例"""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        rf, param_grid, cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    return grid_search
```

### 3.2 随机搜索
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def random_search_example(X, y):
    """随机搜索示例"""
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 20),
        'max_features': uniform(0.1, 0.9)
    }
    
    rf = RandomForestClassifier()
    random_search = RandomizedSearchCV(
        rf, param_distributions,
        n_iter=100, cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    print("Best parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)
    return random_search
```

### 3.3 贝叶斯优化
```python
from skopt import BayesSearchCV

def bayes_search_example(X, y):
    """贝叶斯优化示例"""
    from skopt.space import Real, Integer
    
    search_spaces = {
        'n_estimators': Integer(100, 500),
        'max_depth': Integer(10, 50),
        'min_samples_split': Integer(2, 20),
        'max_features': Real(0.1, 0.9)
    }
    
    rf = RandomForestClassifier()
    bayes_search = BayesSearchCV(
        rf, search_spaces, n_iter=50,
        cv=5, scoring='accuracy',
        n_jobs=-1
    )
    
    bayes_search.fit(X, y)
    print("Best parameters:", bayes_search.best_params_)
    print("Best score:", bayes_search.best_score_)
    return bayes_search
```

## 4. 模型集成

### 4.1 投票集成
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def voting_ensemble_example(X, y):
    """投票集成示例"""
    # 创建基础模型
    clf1 = LogisticRegression()
    clf2 = DecisionTreeClassifier()
    clf3 = SVC(probability=True)
    
    # 创建投票集成
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', clf1),
            ('dt', clf2),
            ('svc', clf3)
        ],
        voting='soft'
    )
    
    return voting_clf
```

### 4.2 Bagging
```python
from sklearn.ensemble import BaggingClassifier

def bagging_example(X, y):
    """Bagging示例"""
    base_estimator = DecisionTreeClassifier()
    bagging = BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8
    )
    
    bagging.fit(X, y)
    return bagging
```

### 4.3 Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def boosting_comparison(X, y):
    """比较不同的Boosting算法"""
    # GBDT
    gbdt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1
    )
    
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1
    )
    
    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1
    )
    
    models = {
        'GBDT': gbdt,
        'XGBoost': xgb,
        'LightGBM': lgb
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        score = model.score(X, y)
        results[name] = score
    
    return results
```

### 4.4 Stacking
```python
from sklearn.ensemble import StackingClassifier

def stacking_example(X, y):
    """Stacking集成示例"""
    estimators = [
        ('rf', RandomForestClassifier()),
        ('gbdt', GradientBoostingClassifier()),
        ('xgb', XGBClassifier())
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression()
    )
    
    stacking.fit(X, y)
    return stacking
```

## 5. 实战案例：信用卡欺诈检测

### 5.1 完整优化流程
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def credit_card_fraud_detection():
    """信用卡欺诈检测完整示例"""
    # 1. 加载数据
    data = pd.read_csv('creditcard.csv')
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # 2. 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 4. 创建和优化模型
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'learning_rate': [0.01, 0.1]
    }
    
    xgb = XGBClassifier()
    grid_search = GridSearchCV(
        xgb, param_grid, cv=3,
        scoring='f1', n_jobs=-1
    )
    
    # 5. 训练模型
    grid_search.fit(X_train, y_train)
    
    # 6. 评估结果
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return grid_search
```

## 常见问题解答

Q: 如何选择正则化方法？
A: 考虑以下因素：
1. L1正则化（Lasso）适合特征选择
2. L2正则化（Ridge）适合处理多重共线性
3. 弹性网络结合两者优点
4. 数据量和特征数也是重要考虑因素

Q: 如何避免过拟合？
A: 常用方法：
1. 增加训练数据
2. 使用正则化
3. 简化模型
4. 使用交叉验证
5. 特征选择
6. 提前停止

Q: 集成学习的优缺点？
A: 优点：
1. 提高模型性能
2. 减少过拟合
3. 提高稳定性

缺点：
1. 计算成本高
2. 模型复杂度增加
3. 可解释性降低

## 扩展阅读
1. [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
2. [XGBoost Documentation](https://xgboost.readthedocs.io/)
3. [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
