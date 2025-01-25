---
title: "特征工程实践指南"
slug: "feature-engineering"
sequence: 4
description: "全面介绍特征工程的原理和技术，包括特征选择、特征提取、特征变换等关键技术"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

# 特征工程实践指南

## 学习目标
完成本章学习后，你将能够：
- 理解特征工程在机器学习中的重要性
- 掌握常用的特征选择和特征提取方法
- 熟练运用特征变换和特征组合技术
- 构建高效的特征工程pipeline

## 1. 特征工程概述

### 1.1 什么是特征工程
特征工程是将原始数据转换为更好的特征表示的过程，使机器学习算法能够更好地工作。它是机器学习中最重要的步骤之一，往往对模型性能有决定性影响。

### 1.2 特征工程的重要性
```python
# 示例：特征工程对模型性能的影响
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# 加载数据
boston = load_boston()
X, y = boston.data, boston.target

# 不进行特征工程
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
print("原始特征得分:", model.score(X_test, y_test))

# 进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
model.fit(X_train, y_train)
print("标准化后得分:", model.score(X_test, y_test))
```

## 2. 特征选择

### 2.1 过滤法（Filter Methods）
基于统计指标选择特征。

```python
from sklearn.feature_selection import SelectKBest, f_classif

def filter_features(X, y, k=5):
    """使用F检验选择最重要的k个特征"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support()
    return X_selected, selected_features

# 示例使用
X_selected, mask = filter_features(X, y)
selected_features = X.columns[mask]  # 假设X是DataFrame
```

### 2.2 包装法（Wrapper Methods）
使用目标算法的性能来评估特征子集。

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def wrapper_feature_selection(X, y, n_features):
    """使用递归特征消除法选择特征"""
    estimator = RandomForestClassifier(n_estimators=100)
    selector = RFE(estimator, n_features_to_select=n_features)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.support_
```

### 2.3 嵌入法（Embedded Methods）
在模型训练过程中进行特征选择。

```python
from sklearn.linear_model import Lasso

def embedded_feature_selection(X, y, alpha=1.0):
    """使用Lasso进行特征选择"""
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    return lasso.coef_ != 0
```

## 3. 特征提取

### 3.1 主成分分析（PCA）
```python
from sklearn.decomposition import PCA

def apply_pca(X, n_components):
    """应用PCA降维"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    return X_pca, explained_variance

# 可视化PCA结果
def plot_pca_components(pca, feature_names):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    components = pd.DataFrame(
        pca.components_,
        columns=feature_names
    )
    sns.heatmap(components, cmap='coolwarm')
    plt.title('PCA Components')
    plt.show()
```

### 3.2 t-SNE
```python
from sklearn.manifold import TSNE

def apply_tsne(X, n_components=2):
    """应用t-SNE进行可视化"""
    tsne = TSNE(n_components=n_components)
    X_tsne = tsne.fit_transform(X)
    return X_tsne
```

### 3.3 自编码器
```python
import tensorflow as tf

def create_autoencoder(input_dim, encoding_dim):
    """创建简单的自编码器"""
    # 编码器
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    
    # 解码器
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    # 完整的自编码器
    autoencoder = tf.keras.Model(input_layer, decoded)
    
    # 仅编码器
    encoder = tf.keras.Model(input_layer, encoded)
    
    return autoencoder, encoder
```

## 4. 特征变换

### 4.1 数值特征变换
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class FeatureTransformer:
    """特征变换工具类"""
    
    @staticmethod
    def standard_scale(X):
        """标准化"""
        return StandardScaler().fit_transform(X)
    
    @staticmethod
    def minmax_scale(X):
        """归一化"""
        return MinMaxScaler().fit_transform(X)
    
    @staticmethod
    def robust_scale(X):
        """鲁棒缩放"""
        return RobustScaler().fit_transform(X)
    
    @staticmethod
    def log_transform(X):
        """对数变换"""
        return np.log1p(X)
    
    @staticmethod
    def power_transform(X, power):
        """幂变换"""
        return np.power(X, power)
```

### 4.2 类别特征编码
```python
class CategoryEncoder:
    """类别特征编码工具类"""
    
    @staticmethod
    def one_hot_encode(X, columns):
        """独热编码"""
        return pd.get_dummies(X, columns=columns)
    
    @staticmethod
    def label_encode(X):
        """标签编码"""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        return le.fit_transform(X)
    
    @staticmethod
    def target_encode(X, y, column):
        """目标编码"""
        means = X.groupby(column)[y].mean()
        return X[column].map(means)
```

## 5. 特征组合

### 5.1 多项式特征
```python
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(X, degree=2):
    """创建多项式特征"""
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)
```

### 5.2 交叉特征
```python
def create_interaction_features(df, feature1, feature2, operation='multiply'):
    """创建交叉特征"""
    if operation == 'multiply':
        return df[feature1] * df[feature2]
    elif operation == 'add':
        return df[feature1] + df[feature2]
    elif operation == 'subtract':
        return df[feature1] - df[feature2]
    elif operation == 'divide':
        return df[feature1] / (df[feature2] + 1e-7)
```

## 6. 特征工程Pipeline

### 6.1 构建Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """创建特征预处理pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
```

### 6.2 自定义Transformer
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    """自定义特征转换器"""
    
    def __init__(self, power=2):
        self.power = power
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        # 添加自定义转换逻辑
        return X_transformed
```

## 7. 实战案例：房价预测

### 7.1 特征工程完整流程
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def housing_price_prediction():
    # 1. 加载数据
    data = pd.read_csv('housing.csv')
    
    # 2. 分离特征和目标
    X = data.drop('price', axis=1)
    y = data['price']
    
    # 3. 识别特征类型
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # 4. 创建预处理pipeline
    preprocessor = create_preprocessing_pipeline(
        numeric_features,
        categorical_features
    )
    
    # 5. 创建完整pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])
    
    # 6. 训练和评估
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    full_pipeline.fit(X_train, y_train)
    score = full_pipeline.score(X_test, y_test)
    
    return full_pipeline, score
```

## 常见问题解答

Q: 如何处理高基数类别特征？
A: 可以采用以下方法：
1. 频率编码
2. 目标编码
3. 分箱
4. 选择top-k类别，其他归为"其他"

Q: 如何处理时间特征？
A: 常用方法包括：
1. 提取年、月、日、星期等
2. 创建周期性特征
3. 计算时间差
4. 提取时间窗口统计特征

Q: 特征选择和降维有什么区别？
A: 主要区别：
- 特征选择保留原始特征的子集
- 降维创建新的特征组合
- 特征选择更易解释
- 降维可能获得更好的表示

## 扩展阅读
1. [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
2. [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
3. [Feature Engineering Made Easy](https://www.packtpub.com/product/feature-engineering-made-easy/9781787287600)
