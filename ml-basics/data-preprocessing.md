---
title: "数据预处理与特征工程"
slug: "data-preprocessing-and-feature-engineering"
sequence: 2
description: "学习数据清洗、预处理和特征工程的关键技术，为模型训练准备高质量数据"
is_published: true
estimated_minutes: 60
language: "zh-CN"
---

![数据预处理与特征工程](assets/images/ml-basics/preprocessing-header.png)
*数据是机器学习的燃料，而特征工程是提炼这些燃料的关键工艺*

# 数据预处理与特征工程

## 学习目标
完成本节后，你将能够：
- 使用Pandas进行数据加载和基础处理
- 识别和处理数据质量问题（缺失值、异常值等）
- 掌握特征工程的各种技术（编码、缩放、创建等）
- 构建端到端的数据处理Pipeline

## 先修知识
学习本节内容需要：
- Python基础编程
- Pandas库的基本使用
- 机器学习基础概念
- 基本的统计知识

## Pandas数据处理基础

### 数据读取与查看
数据分析的第一步是了解数据的基本情况，包括数据结构、特征类型和统计特征。

```python
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('data.csv')

# 基本信息查看
print("数据基本信息:")
print(df.info())

# 数据预览
print("\n前5行数据:")
print(df.head())

# 基本统计信息
print("\n基本统计信息:")
print(df.describe())
```

### 数据选择与过滤
熟练的数据操作能力是数据分析的基础。

```python
# 列选择
selected_columns = df[['column1', 'column2']]

# 条件过滤
filtered_data = df[df['age'] > 25]

# 多条件过滤
complex_filter = df[(df['age'] > 25) & (df['salary'] > 50000)]

# 索引操作
df.loc[0:5, 'column1':'column3']  # 标签索引
df.iloc[0:5, 0:3]                 # 位置索引
```

### 数据转换
数据转换是将原始数据转化为更适合分析的形式。

```python
# 类型转换
df['age'] = df['age'].astype('int32')

# 重命名列
df = df.rename(columns={'old_name': 'new_name'})

# 排序
df_sorted = df.sort_values('column1', ascending=False)

# 分组操作
grouped = df.groupby('category').agg({
    'value': ['mean', 'count'],
    'other_value': 'sum'
})
```

## 数据清洗

### 缺失值处理
缺失值是数据分析中最常见的问题之一，需要谨慎处理。

```python
# 检查缺失值
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

# 处理缺失值
# 删除
df_cleaned = df.dropna()  # 删除含有缺失值的行
df_cleaned = df.dropna(subset=['important_column'])  # 特定列

# 填充
df['column'].fillna(df['column'].mean())  # 均值填充
df['column'].fillna(df['column'].median())  # 中位数填充
df['column'].fillna(df['column'].mode()[0])  # 众数填充
df['column'].fillna(method='ffill')  # 前向填充
df['column'].fillna(method='bfill')  # 后向填充
```

### 异常值检测与处理
异常值可能是错误数据，也可能是重要的异常情况。

```python
from scipy import stats

def detect_outliers_zscore(data, threshold=3):
    """
    使用Z-score方法检测异常值
    
    参数:
        data: 数值型数据
        threshold: Z-score阈值，默认为3
    
    返回:
        布尔数组，True表示异常值
    """
    z_scores = stats.zscore(data)
    return abs(z_scores) > threshold

def detect_outliers_iqr(data):
    """
    使用IQR方法检测异常值
    
    参数:
        data: 数值型数据
    
    返回:
        布尔数组，True表示异常值
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def cap_outliers(data):
    """
    将异常值截断到上下限范围内
    
    参数:
        data: 数值型数据
    
    返回:
        处理后的数据
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(data, lower_bound, upper_bound)

# 使用示例
# 删除异常值
df = df[~detect_outliers_zscore(df['column'])]

# 截断异常值
df['column'] = cap_outliers(df['column'])
```

### 数据一致性检查
确保数据的质量和一致性是数据清洗的重要部分。

```python
# 重复值检查
duplicates = df.duplicated().sum()
df_unique = df.drop_duplicates()

def check_range(data, column, min_val, max_val):
    """
    检查数据是否在指定范围内
    
    参数:
        data: DataFrame
        column: 要检查的列名
        min_val: 最小允许值
        max_val: 最大允许值
    
    返回:
        超出范围的数据
    """
    invalid = data[(data[column] < min_val) | (data[column] > max_val)]
    return invalid

def check_format(data, column, pattern):
    """
    检查数据是否符合指定格式
    
    参数:
        data: DataFrame
        column: 要检查的列名
        pattern: 正则表达式模式
    
    返回:
        不符合格式的数据
    """
    import re
    invalid = data[~data[column].str.match(pattern)]
    return invalid
```

## 特征工程

### 特征缩放
不同尺度的特征可能会影响模型性能，需要进行适当的缩放。

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 标准化 (Z-score标准化)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 归一化 (Min-Max缩放)
min_max_scaler = MinMaxScaler()
df_normalized = min_max_scaler.fit_transform(df)

# 稳健缩放 (处理异常值)
robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df)
```

### 特征编码
将分类特征转换为数值形式，使机器学习算法能够处理。

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 标签编码
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# One-Hot编码
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(df[['category']])

# 自定义映射
mapping = {'low': 0, 'medium': 1, 'high': 2}
df['category_mapped'] = df['category'].map(mapping)

# 频率编码
frequency_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(frequency_map)
```

### 特征创建与转换
创建新特征或转换现有特征可以帮助模型捕捉更多信息。

```python
# 多项式特征
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 交互特征
df['interaction'] = df['feature1'] * df['feature2']

# 时间特征
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# 分箱
df['age_group'] = pd.qcut(df['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

## 实战：构建数据处理Pipeline

### 项目描述
- 目标：构建一个端到端的数据处理pipeline
- 数据：包含数值和类别特征的数据集
- 技术要点：数据清洗、特征工程、Pipeline构建

### 完整代码实现

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 定义数值和类别特征
numeric_features = ['age', 'salary']
categorical_features = ['department', 'position']

# 数值特征处理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别特征处理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first'))
])

# 组合转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 创建完整pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# 训练pipeline
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
```

## 练习与作业
- 基础练习：
  * 使用Pandas读取一个CSV文件
  * 进行基本的数据探索
  * 处理文件中的缺失值

- 提高练习：
  * 实现一个函数，能够自动检测和处理异常值
  * 对数值特征进行不同方式的缩放，比较效果

- 挑战练习：
  * 构建一个完整的数据处理pipeline
  * 包含缺失值处理、异常值处理、特征工程等步骤
  * 使用交叉验证评估pipeline的效果

## 常见问题
Q1: 如何选择合适的缺失值处理方法？
A1: 需要考虑以下因素：
- 缺失值的比例
- 缺失的原因
- 数据的分布特征
- 业务场景的要求

Q2: 什么时候应该使用One-Hot编码，什么时候使用标签编码？
A2: 
- One-Hot编码适用于：
  * 类别之间没有顺序关系
  * 类别数量较少
  * 模型对特征独立性要求高
- 标签编码适用于：
  * 类别之间有顺序关系
  * 类别数量很多
  * 树模型等对特征独立性要求不高的模型

## 小测验
- 以下哪些是处理缺失值的有效方法？为什么？
  * 删除缺失值
  * 均值填充
  * 中位数填充
  * 模型预测填充

- 解释为什么要进行特征缩放？不同的缩放方法有什么区别？

- 在实际项目中，如何选择合适的特征工程方法？需要考虑哪些因素？

## 扩展阅读
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [scikit-learn预处理指南](https://scikit-learn.org/stable/modules/preprocessing.html)
- [数据清洗实践指南](https://www.kaggle.com/learn/data-cleaning)

## 下一步学习
- 机器学习算法基础
- 模型评估与选择
- 模型调优技术
- 特征选择方法
