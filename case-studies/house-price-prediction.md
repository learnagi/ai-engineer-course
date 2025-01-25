---
title: "实战案例：波士顿房价预测"
slug: "house-price-prediction"
description: "通过波士顿房价数据集，实践特征工程和模型优化的完整机器学习流程"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

# 实战案例：波士顿房价预测

## 项目概述
本案例将使用波士顿房价数据集，完整演示机器学习项目的各个环节，包括：
- 数据分析和预处理
- 特征工程
- 模型选择和优化
- 模型评估和解释

## 1. 环境准备
```python
# 所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import shap

# 设置随机种子
np.random.seed(42)
```

## 2. 数据加载和探索

### 2.1 加载数据
```python
# 加载波士顿房价数据集
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

print("数据集形状:", data.shape)
print("\n特征说明:")
for name, desc in zip(boston.feature_names, boston.DESCR.split('\n')[20:33]):
    print(f"{name}: {desc.strip()}")
```

### 2.2 探索性数据分析
```python
def explore_data(data):
    """探索性数据分析"""
    # 基本统计信息
    print("基本统计信息:")
    print(data.describe())
    
    # 相关性分析
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("特征相关性热力图")
    plt.show()
    
    # 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data['PRICE'], bins=30)
    plt.title("房价分布")
    plt.show()
    
    # 箱线图
    plt.figure(figsize=(15, 6))
    data.boxplot()
    plt.xticks(rotation=45)
    plt.title("特征箱线图")
    plt.show()

explore_data(data)
```

## 3. 特征工程

### 3.1 特征处理
```python
def feature_engineering(data):
    """特征工程"""
    # 创建特征副本
    df = data.copy()
    
    # 处理偏态分布
    skewed_features = ['CRIM', 'ZN', 'RM', 'DIS', 'B', 'LSTAT']
    for feature in skewed_features:
        df[feature] = np.log1p(df[feature])
    
    # 创建交互特征
    df['RM_LSTAT'] = df['RM'] * df['LSTAT']
    df['RM_AGE'] = df['RM'] * df['AGE']
    df['LSTAT_AGE'] = df['LSTAT'] * df['AGE']
    
    # 多项式特征
    df['RM_sq'] = df['RM'] ** 2
    df['LSTAT_sq'] = df['LSTAT'] ** 2
    
    return df

# 应用特征工程
df_engineered = feature_engineering(data)
```

### 3.2 特征选择
```python
from sklearn.feature_selection import SelectKBest, f_regression

def select_features(X, y, k=15):
    """特征选择"""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print("选中的特征:", selected_features)
    return X_selected, selected_features

# 准备数据
X = df_engineered.drop('PRICE', axis=1)
y = df_engineered['PRICE']

# 特征选择
X_selected, selected_features = select_features(X, y)
```

## 4. 模型构建和优化

### 4.1 基准模型
```python
def evaluate_models(X, y):
    """评估多个基准模型"""
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义模型
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42)
    }
    
    # 评估模型
    results = {}
    for name, model in models.items():
        # 训练和预测
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=5, scoring='r2'
        )
        
        results[name] = {
            'MSE': mse,
            'R2': r2,
            'CV_mean': cv_scores.mean(),
            'CV_std': cv_scores.std()
        }
    
    return results, models

# 评估模型
results, models = evaluate_models(X_selected, y)
print("\n模型评估结果:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
```

### 4.2 超参数优化
```python
from sklearn.model_selection import RandomizedSearchCV

def optimize_lightgbm(X, y):
    """优化LightGBM模型"""
    # 参数空间
    param_dist = {
        'n_estimators': np.arange(100, 1000, 100),
        'max_depth': np.arange(3, 10),
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': np.arange(20, 100, 10),
        'min_child_samples': np.arange(10, 50, 10),
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
    }
    
    # 创建模型
    lgb_model = lgb.LGBMRegressor(random_state=42)
    
    # 随机搜索
    random_search = RandomizedSearchCV(
        lgb_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    
    # 训练
    random_search.fit(X, y)
    
    print("最佳参数:", random_search.best_params_)
    print("最佳得分:", -random_search.best_score_)
    
    return random_search.best_estimator_

# 优化LightGBM模型
best_model = optimize_lightgbm(X_selected, y)
```

## 5. 模型解释

### 5.1 特征重要性
```python
def plot_feature_importance(model, feature_names):
    """绘制特征重要性"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title("特征重要性")
    plt.show()
    
    return importance

# 分析特征重要性
importance = plot_feature_importance(best_model, selected_features)
```

### 5.2 SHAP值分析
```python
def analyze_shap_values(model, X):
    """SHAP值分析"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 特征重要性
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X)
    
    # 依赖图
    for feature in X.columns[:3]:  # 展示前三个重要特征
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X)

# SHAP值分析
analyze_shap_values(best_model, pd.DataFrame(X_selected, columns=selected_features))
```

## 6. 模型部署

### 6.1 保存模型
```python
import joblib

def save_model(model, scaler, file_prefix='boston_house'):
    """保存模型和预处理器"""
    joblib.dump(model, f'{file_prefix}_model.pkl')
    joblib.dump(scaler, f'{file_prefix}_scaler.pkl')

# 保存模型
save_model(best_model, scaler)
```

### 6.2 预测函数
```python
def predict_price(features, model_path='boston_house_model.pkl', 
                 scaler_path='boston_house_scaler.pkl'):
    """预测房价"""
    # 加载模型和预处理器
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 预处理
    features_scaled = scaler.transform(features)
    
    # 预测
    prediction = model.predict(features_scaled)
    
    return prediction

# 示例预测
sample_features = X_test_scaled[:1]
predicted_price = predict_price(sample_features)
print(f"预测房价: ${predicted_price[0]:,.2f}")
```

## 7. 模型监控

### 7.1 性能监控
```python
def monitor_performance(y_true, y_pred, threshold=0.1):
    """监控模型性能"""
    # 计算误差
    errors = np.abs(y_true - y_pred) / y_true
    
    # 性能指标
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(errors),
        'Error_std': np.std(errors),
        'Over_threshold': np.mean(errors > threshold)
    }
    
    return metrics

# 监控性能
performance = monitor_performance(y_test, best_model.predict(X_test_scaled))
print("\n性能指标:")
for metric, value in performance.items():
    print(f"{metric}: {value:.4f}")
```

## 常见问题解答

Q: 如何处理新特征？
A: 新特征需要经过与训练数据相同的处理流程：
1. 特征工程（如对数变换）
2. 标准化
3. 特征选择

Q: 模型效果不好怎么办？
A: 可以尝试以下方法：
1. 收集更多数据
2. 尝试更复杂的特征工程
3. 使用集成学习
4. 调整超参数
5. 分析预测错误的案例

Q: 如何处理过拟合？
A: 常用方法：
1. 减少特征数量
2. 增加正则化
3. 使用交叉验证
4. 收集更多数据
5. 简化模型

## 扩展阅读
1. [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
2. [Applied Predictive Modeling](http://appliedpredictivemodeling.com/)
3. [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
