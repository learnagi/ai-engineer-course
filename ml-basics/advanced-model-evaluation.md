---
title: "高级模型评估技术"
slug: "advanced-model-evaluation"
sequence: 5
description: "深入学习高级模型评估方法，包括复杂交叉验证、超参数调优和模型诊断技术"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

![Advanced Model Evaluation](images/advanced-model-evaluation-header.png)
*掌握高级模型评估技术，构建更可靠的机器学习模型*

# 高级模型评估技术

## 学习目标
完成本模块学习后，你将能够：
- 使用高级交叉验证技术评估模型
- 掌握现代超参数调优方法
- 处理不平衡数据集的评估
- 进行深入的模型诊断

## 先修知识
- 基础模型评估概念
- Python编程和机器学习基础
- scikit-learn库的使用经验

## 1. 高级交叉验证技术

### 1.1 分层交叉验证
```python
from sklearn.model_selection import StratifiedKFold

def stratified_cross_validation(model, X, y, k=5):
    """执行分层交叉验证，保持每折中类别分布一致"""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
        
        print(f'Fold {fold + 1}: {score:.3f}')
    
    return np.mean(scores), np.std(scores)
```

### 1.2 时间序列交叉验证
```python
from sklearn.model_selection import TimeSeriesSplit

def timeseries_cross_validation(model, X, y, n_splits=5):
    """执行时间序列交叉验证"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

## 2. 现代超参数调优

### 2.1 贝叶斯优化
```python
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def bayesian_optimization(model, param_space, X, y, n_iter=50):
    """使用贝叶斯优化进行超参数调优"""
    bayes_search = BayesSearchCV(
        model,
        param_space,
        n_iter=n_iter,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    bayes_search.fit(X, y)
    
    print("最佳参数:", bayes_search.best_params_)
    print("最佳得分:", bayes_search.best_score_)
    
    return bayes_search
```

### 2.2 Optuna框架
```python
import optuna

def objective(trial, model_class, X, y):
    """定义Optuna优化目标"""
    # 定义超参数搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.0)
    }
    
    # 创建模型
    model = model_class(**params)
    
    # 交叉验证评估
    score = cross_val_score(model, X, y, cv=5).mean()
    
    return score

def optimize_hyperparameters(model_class, X, y, n_trials=100):
    """使用Optuna进行超参数优化"""
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_class, X, y), 
                  n_trials=n_trials)
    
    print("最佳参数:", study.best_params)
    print("最佳得分:", study.best_value)
    
    return study
```

## 3. 不平衡数据集评估

### 3.1 特殊评估指标
```python
from sklearn.metrics import balanced_accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_curve

def evaluate_imbalanced_dataset(y_true, y_pred, y_prob):
    """评估不平衡数据集的性能"""
    # 计算平衡准确率
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # 计算PR曲线下的面积
    average_precision = average_precision_score(y_true, y_prob)
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    return {
        'balanced_accuracy': balanced_acc,
        'average_precision': average_precision,
        'pr_curve': (precision, recall)
    }
```

### 3.2 分层采样评估
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def evaluate_with_resampling(model, X, y, cv=5):
    """使用重采样技术进行评估"""
    # 创建包含重采样的管道
    pipeline = ImbPipeline([
        ('sampler', SMOTE()),
        ('classifier', model)
    ])
    
    # 使用分层交叉验证
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='balanced_accuracy')
    
    return np.mean(scores), np.std(scores)
```

## 4. 高级模型诊断

### 4.1 学习曲线分析
```python
def plot_advanced_learning_curve(model, X, y, cv=5, n_jobs=-1):
    """绘制详细的学习曲线，包含训练时间分析"""
    from time import time
    from sklearn.model_selection import learning_curve
    
    # 定义训练样本数量
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # 记录开始时间
    start_time = time()
    
    # 计算学习曲线
    train_sizes, train_scores, val_scores, fit_times, _ = \
        learning_curve(model, X, y, cv=cv, n_jobs=n_jobs,
                      train_sizes=train_sizes,
                      return_times=True)
    
    # 计算统计量
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制学习曲线
    ax1.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    ax1.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    ax1.plot(train_sizes, train_mean, label='训练得分')
    ax1.plot(train_sizes, val_mean, label='验证得分')
    ax1.set_xlabel('训练样本数')
    ax1.set_ylabel('得分')
    ax1.legend(loc='best')
    ax1.set_title('学习曲线')
    
    # 绘制时间复杂度曲线
    ax2.plot(train_sizes, fit_times_mean, 'o-')
    ax2.fill_between(train_sizes, fit_times_mean - fit_times_std,
                     fit_times_mean + fit_times_std, alpha=0.1)
    ax2.set_xlabel('训练样本数')
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('可扩展性分析')
    
    plt.tight_layout()
    plt.show()
```

### 4.2 特征重要性分析
```python
def analyze_feature_importance(model, X, feature_names=None):
    """分析特征重要性并可视化"""
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        raise ValueError("模型不支持特征重要性分析")
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title('Top 20 特征重要性')
    plt.tight_layout()
    plt.show()
    
    return importance_df
```

## 5. 实践建议

### 5.1 评估策略选择
- 小数据集（<1000样本）：使用留一法交叉验证
- 中等数据集：使用5-10折交叉验证
- 大数据集（>100000样本）：使用简单的训练/验证/测试集划分
- 时间序列数据：使用时间序列交叉验证
- 类别不平衡数据：使用分层交叉验证

### 5.2 超参数调优建议
- 先使用随机搜索确定参数的大致范围
- 再使用贝叶斯优化进行精细调优
- 对计算资源要求高的模型，考虑使用Optuna等现代框架
- 始终使用交叉验证来评估超参数的性能

### 5.3 模型诊断清单
- 检查学习曲线判断过拟合/欠拟合
- 分析特征重要性，考虑特征选择
- 对于不平衡数据集，重点关注少数类的性能
- 考虑模型的训练时间和推理时间
- 定期监控模型在新数据上的表现

## 参考资源

- [scikit-learn高级模型选择](https://scikit-learn.org/stable/model_selection.html)
- [Optuna文档](https://optuna.readthedocs.io/)
- [不平衡学习](https://imbalanced-learn.org/)
- [特征选择指南](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Python机器学习实战](https://www.manning.com/books/python-machine-learning)
