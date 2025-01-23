---
title: "模型评估与验证"
slug: "model-evaluation"
sequence: 4
description: "学习机器学习模型的评估方法、验证技术和性能指标"
is_published: true
estimated_minutes: 90
language: "zh-CN"
---

![Model Evaluation](images/model-evaluation-header.png)
*准确的模型评估是机器学习项目成功的关键*

# 模型评估与验证

## 学习目标
完成本模块学习后，你将能够：
- 理解模型评估的重要性
- 掌握常用的评估指标
- 使用交叉验证方法
- 处理过拟合和欠拟合问题

## 先修知识
- Python编程基础
- 机器学习基础概念
- 基本的统计知识
- 数据处理技能

## 1. 评估指标

### 1.1 分类问题指标
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """计算分类问题的评估指标"""
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC曲线（如果有概率预测）
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc': (fpr, tpr, roc_auc)
    }

def plot_confusion_matrix(cm, classes):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
```

### 1.2 回归问题指标
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_regression_metrics(y_true, y_pred):
    """计算回归问题的评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def plot_regression_results(y_true, y_pred):
    """绘制回归结果对比图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Regression Results')
    plt.show()
```

## 2. 交叉验证

### 2.1 K折交叉验证
```python
from sklearn.model_selection import KFold, cross_val_score

def k_fold_cross_validation(model, X, y, k=5):
    """执行K折交叉验证"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_val)
        score = model.score(X_val, y_val)
        scores.append(score)
        
        print(f'Fold {fold + 1}: {score:.3f}')
    
    print(f'Average Score: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})')
    return scores
```

### 2.2 留一法交叉验证
```python
from sklearn.model_selection import LeaveOneOut

def leave_one_out_cv(model, X, y):
    """执行留一法交叉验证"""
    loo = LeaveOneOut()
    scores = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

## 3. 学习曲线与验证曲线

### 3.1 学习曲线
```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, cv=5):
    """绘制学习曲线"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    # 计算平均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    
    # 添加标准差区域
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

### 3.2 验证曲线
```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(model, X, y, param_name, param_range, cv=5):
    """绘制验证曲线"""
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, n_jobs=-1)
    
    # 计算平均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # 绘制验证曲线
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label='Training score')
    plt.plot(param_range, val_mean, label='Cross-validation score')
    
    # 添加标准差区域
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(param_range, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title('Validation Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

## 4. 过拟合与欠拟合诊断

### 4.1 偏差方差分解
```python
def bias_variance_decomposition(model, X, y, test_size=0.2, n_iterations=100):
    """计算模型的偏差和方差"""
    predictions = np.zeros((len(X), n_iterations))
    
    for i in range(n_iterations):
        # 随机划分训练集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        predictions[:, i] = model.predict(X)
    
    # 计算偏差和方差
    expected_pred = np.mean(predictions, axis=1)
    bias = np.mean((y - expected_pred) ** 2)
    variance = np.mean(np.var(predictions, axis=1))
    
    return bias, variance

def plot_bias_variance_trade_off(model, X, y, param_range, param_name):
    """绘制偏差方差权衡图"""
    biases = []
    variances = []
    
    for param_value in param_range:
        # 设置模型参数
        setattr(model, param_name, param_value)
        
        # 计算偏差和方差
        bias, variance = bias_variance_decomposition(model, X, y)
        biases.append(bias)
        variances.append(variance)
    
    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, biases, label='Bias')
    plt.plot(param_range, variances, label='Variance')
    plt.plot(param_range, np.array(biases) + np.array(variances),
             label='Total Error')
    
    plt.xlabel(param_name)
    plt.ylabel('Error')
    plt.title('Bias-Variance Trade-off')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 5. 模型选择

### 5.1 网格搜索
```python
from sklearn.model_selection import GridSearchCV

def grid_search_cv(model, param_grid, X, y, cv=5):
    """执行网格搜索"""
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1,
                              scoring='accuracy', verbose=1)
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    # 绘制参数搜索结果
    results = pd.DataFrame(grid_search.cv_results_)
    for param in param_grid:
        plt.figure(figsize=(10, 6))
        plt.errorbar(results[f'param_{param}'],
                    results['mean_test_score'],
                    yerr=results['std_test_score'])
        plt.xlabel(param)
        plt.ylabel('Score')
        plt.title(f'Grid Search Results for {param}')
        plt.grid(True)
        plt.show()
    
    return grid_search.best_estimator_
```

### 5.2 随机搜索
```python
from sklearn.model_selection import RandomizedSearchCV

def random_search_cv(model, param_distributions, X, y, n_iter=100, cv=5):
    """执行随机搜索"""
    random_search = RandomizedSearchCV(model, param_distributions,
                                     n_iter=n_iter, cv=cv, n_jobs=-1,
                                     scoring='accuracy', verbose=1)
    random_search.fit(X, y)
    
    print("Best parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)
    
    return random_search.best_estimator_
```

## 常见问题解答

Q: 如何选择合适的评估指标？
A: 根据问题类型选择：
- 分类问题：准确率、精确率、召回率、F1分数
- 回归问题：MSE、RMSE、MAE、R²
- 不平衡数据：AUC-ROC、PR曲线

Q: 如何处理过拟合问题？
A: 可以采用以下方法：
- 增加训练数据
- 减少模型复杂度
- 使用正则化
- 使用交叉验证
- 特征选择

Q: 如何选择交叉验证方法？
A: 根据数据特点选择：
- 数据量大：K折交叉验证
- 数据量小：留一法交叉验证
- 时间序列数据：时间序列交叉验证

## 扩展阅读
- [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Model Selection and Evaluation](https://scikit-learn.org/stable/model_selection.html)
