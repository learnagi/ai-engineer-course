---
title: "数据分析与处理"
slug: "data-analysis"
sequence: 1
description: "掌握数据分析的核心技能，包括Pandas高级操作、数据清洗、探索性分析等实用技巧"
is_published: true
estimated_minutes: 90
language: "zh-CN"
---

![Data Analysis](images/data-analysis-header.png)
*数据分析是AI工程的基石，让我们从数据中发现价值*

# 数据分析与处理

## 学习目标
完成本模块学习后，你将能够：
- 熟练使用Pandas进行数据操作和分析
- 掌握数据清洗和预处理技巧
- 进行探索性数据分析
- 处理真实世界的数据问题

## 先修知识
- Python基础编程
- 基本的统计学概念
- NumPy基础操作

## 1. Pandas高级操作

### 1.1 数据操作
```python
import pandas as pd
import numpy as np

# 高效的数据读取
df = pd.read_csv('data.csv', 
                 usecols=['user_id', 'timestamp', 'value'],
                 parse_dates=['timestamp'],
                 dtype={'user_id': 'category'})

# 链式操作
result = (df
    .groupby('user_id')
    .agg({
        'value': ['mean', 'std'],
        'timestamp': ['min', 'max']
    })
    .round(2)
    .reset_index())

# 多层索引操作
df.columns = pd.MultiIndex.from_product([['A', 'B'], ['val1', 'val2']])
df.loc[:, ('A', 'val1')]  # 访问特定列

# 高效的数据筛选
mask = (df['value'] > df['value'].mean()) & \
       (df['timestamp'].dt.year == 2024)
filtered_df = df.loc[mask]
```

### 1.2 数据转换
```python
# 类型转换和优化
df['category'] = df['category'].astype('category')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 透视表操作
pivot_table = df.pivot_table(
    values='value',
    index='user_id',
    columns='category',
    aggfunc=['mean', 'count'],
    fill_value=0
)

# 时间序列重采样
daily_stats = df.set_index('timestamp').resample('D').agg({
    'value': ['mean', 'sum', 'count'],
    'user_id': 'nunique'
})

# 自定义转换函数
def normalize_column(series):
    """标准化数值列"""
    return (series - series.mean()) / series.std()

df['value_normalized'] = df.groupby('category')['value'].transform(normalize_column)
```

## 2. 数据清洗与预处理

### 2.1 缺失值处理
```python
# 检查缺失值
missing_stats = df.isnull().sum() / len(df) * 100
print("缺失值比例(%):")
print(missing_stats[missing_stats > 0])

# 智能填充缺失值
df['value'] = df['value'].fillna(df.groupby('category')['value'].transform('mean'))

# 处理异常值
def remove_outliers(group, column, n_std=3):
    """使用Z分数法删除异常值"""
    z_scores = np.abs((group[column] - group[column].mean()) / group[column].std())
    return group[z_scores < n_std]

clean_df = df.groupby('category').apply(
    lambda x: remove_outliers(x, 'value')
).reset_index(drop=True)
```

### 2.2 数据质量提升
```python
# 重复值处理
duplicates = df[df.duplicated(subset=['user_id', 'timestamp'], keep=False)]
df = df.drop_duplicates(subset=['user_id', 'timestamp'], keep='last')

# 数据一致性检查
def validate_data(df):
    """数据验证函数"""
    checks = {
        'missing_values': df.isnull().sum().to_dict(),
        'negative_values': (df < 0).sum().to_dict(),
        'unique_counts': df.nunique().to_dict(),
        'value_ranges': {
            col: {'min': df[col].min(), 
                  'max': df[col].max()}
            for col in df.select_dtypes('number').columns
        }
    }
    return pd.DataFrame(checks)

# 数据清理pipeline
class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
    
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
    
    def handle_missing_values(self):
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self
    
    def remove_outliers(self, columns, n_std=3):
        for col in columns:
            self.df = self.df[np.abs(self.df[col] - self.df[col].mean()) <= 
                            (n_std * self.df[col].std())]
        return self
    
    def get_clean_data(self):
        return self.df

# 使用pipeline
cleaner = DataCleaner(df)
clean_df = (cleaner
    .remove_duplicates()
    .handle_missing_values()
    .remove_outliers(['value'])
    .get_clean_data())
```

## 3. 探索性数据分析

### 3.1 统计描述
```python
# 基本统计量
summary = df.describe(include='all').round(2)

# 相关性分析
correlation = df.corr()
correlation_matrix = df.corr().round(2)

# 分组统计
group_stats = df.groupby('category').agg({
    'value': ['count', 'mean', 'std', 'min', 'max'],
    'user_id': 'nunique'
}).round(2)

# 时间模式分析
time_patterns = df.set_index('timestamp').groupby([
    df['timestamp'].dt.hour,
    df['timestamp'].dt.dayofweek
])['value'].mean().unstack()
```

### 3.2 数据可视化
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 设置可视化风格
plt.style.use('seaborn')
sns.set_palette("husl")

# 分布可视化
def plot_distribution(df, column):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 直方图
    sns.histplot(data=df, x=column, kde=True, ax=ax1)
    ax1.set_title(f'{column}分布')
    
    # 箱线图
    sns.boxplot(data=df, y=column, ax=ax2)
    ax2.set_title(f'{column}箱线图')
    
    plt.tight_layout()
    return fig

# 相关性热力图
def plot_correlation_matrix(df, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f')
    plt.title('特征相关性矩阵')
    
# 时间序列趋势
def plot_time_trends(df, value_col, date_col='timestamp'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 日趋势
    daily = df.set_index(date_col).resample('D')[value_col].mean()
    daily.plot(ax=ax1, title='日均值趋势')
    
    # 月趋势
    monthly = df.set_index(date_col).resample('M')[value_col].mean()
    monthly.plot(ax=ax2, title='月均值趋势')
    
    plt.tight_layout()
    return fig
```

## 4. 实战案例：用户行为分析

### 4.1 数据准备
```python
# 加载数据
df = pd.read_csv('user_behavior.csv')

# 数据清理
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['user_id'] = df['user_id'].astype('category')
df['event_type'] = df['event_type'].astype('category')

# 特征工程
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
```

### 4.2 用户行为分析
```python
# 用户活跃度分析
user_activity = df.groupby('user_id').agg({
    'timestamp': ['count', 'min', 'max'],
    'event_type': lambda x: x.nunique()
}).round(2)

# 事件频率分析
event_frequency = df.groupby(['hour', 'day_of_week'])['event_type'].count().unstack()

# 用户留存分析
def calculate_retention(df, time_col='timestamp', user_col='user_id'):
    """计算用户留存率"""
    # 获取用户首次活跃时间
    first_activity = df.groupby(user_col)[time_col].min().reset_index()
    first_activity['cohort'] = first_activity[time_col].dt.to_period('M')
    
    # 计算每个用户在每个月的活跃情况
    df['activity_month'] = df[time_col].dt.to_period('M')
    monthly_activity = df.groupby([user_col, 'activity_month']).size().reset_index()
    
    # 合并首次活跃信息
    retention_data = pd.merge(monthly_activity, first_activity, on=user_col)
    retention_data['months_since_first'] = (
        retention_data['activity_month'] - retention_data['cohort']
    ).apply(lambda x: x.n)
    
    # 计算留存率
    cohort_sizes = retention_data.groupby('cohort').size()
    retention_rates = retention_data.pivot_table(
        index='cohort',
        columns='months_since_first',
        values=user_col,
        aggfunc='nunique'
    ).divide(cohort_sizes, axis=0) * 100
    
    return retention_rates.round(2)

# 计算并可视化留存率
retention_rates = calculate_retention(df)
plt.figure(figsize=(12, 8))
sns.heatmap(retention_rates, 
            annot=True, 
            fmt='.1f', 
            cmap='YlOrRd')
plt.title('用户留存率热力图 (%)')
```

## 常见问题解答

Q: 如何处理大规模数据集？
A: 使用分块读取（chunk）、内存优化（如类型转换）和并行处理等技术。可以考虑使用dask或vaex等专门处理大数据的库。

Q: 如何选择合适的数据清洗策略？
A: 根据数据特点和业务需求选择。对于关键特征，可能需要更保守的方法（如手动检查）；对于次要特征，可以使用自动化方法（如统计填充）。

Q: 探索性数据分析的关键步骤是什么？
A: 首先了解数据基本情况（大小、类型、缺失值等），然后进行统计描述，最后通过可视化深入分析数据特征和关系。要特别关注异常值和特殊模式。
