---
title: "非监督学习基础"
slug: "unsupervised-learning"
sequence: 2
description: "掌握非监督学习的核心概念、聚类算法和降维技术"
is_published: true
estimated_minutes: 90
language: "zh-CN"
---

![Unsupervised Learning](images/unsupervised-learning-header.png)
*非监督学习帮助我们发现数据中的隐藏模式*

# 非监督学习基础

## 学习目标
完成本模块学习后，你将能够：
- 理解非监督学习的基本原理
- 掌握常用的聚类算法
- 学习数据降维技术
- 应用非监督学习解决实际问题

## 先修知识
- Python编程基础
- 数学基础(线性代数、概率统计)
- 数据预处理技能

## 1. 非监督学习概述

### 1.1 基本概念
非监督学习是机器学习的重要分支，其特点是：
- 不需要标记数据
- 自动发现数据中的模式和结构
- 常用于数据探索和特征学习

### 1.2 主要应用
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 数据准备示例
def generate_sample_data(n_samples=1000):
    """生成示例数据"""
    np.random.seed(42)
    
    # 生成两个高斯分布的数据
    cluster1 = np.random.normal(0, 1, (n_samples//2, 2))
    cluster2 = np.random.normal(3, 1.5, (n_samples//2, 2))
    
    # 合并数据
    X = np.vstack([cluster1, cluster2])
    
    return X

# 数据可视化
def plot_clusters(X, labels=None, title="数据分布"):
    """绘制聚类结果"""
    plt.figure(figsize=(10, 6))
    if labels is None:
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.show()
```

## 2. 聚类算法

### 2.1 K-Means聚类
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_clustering(X, n_clusters=3):
    """K-Means聚类分析"""
    # 创建和训练模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, labels)
    print(f"轮廓系数: {silhouette_avg:.3f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 4))
    
    # 原始数据的聚类结果
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:, 1], 
                marker='x', s=200, linewidths=3, 
                color='r', label='中心点')
    plt.title('K-Means聚类结果')
    plt.legend()
    
    # 肘部法则图
    plt.subplot(122)
    inertias = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k值')
    plt.ylabel('簇内平方和')
    plt.title('肘部法则图')
    
    plt.tight_layout()
    plt.show()
    
    return kmeans

```

### 2.2 DBSCAN聚类
```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def dbscan_clustering(X, eps=0.5, min_samples=5):
    """DBSCAN聚类分析"""
    # 确定最佳eps参数
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # 绘制k-距离图
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(distances[:, -1]))
    plt.xlabel('样本')
    plt.ylabel(f'{min_samples}-距离')
    plt.title('K-距离图')
    plt.show()
    
    # 执行DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # 计算轮廓系数
    if len(np.unique(labels)) > 1:  # 确保不止一个簇
        silhouette_avg = silhouette_score(X, labels)
        print(f"轮廓系数: {silhouette_avg:.3f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title('DBSCAN聚类结果')
    plt.colorbar(label='簇标签')
    plt.show()
    
    # 统计信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f'估计的簇数量: {n_clusters}')
    print(f'估计的噪声点数量: {n_noise}')
    
    return dbscan
```

### 2.3 层次聚类
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def hierarchical_clustering(X, n_clusters=3):
    """层次聚类分析"""
    # 计算链接矩阵
    linkage_matrix = linkage(X, method='ward')
    
    # 绘制树状图
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('层次聚类树状图')
    plt.xlabel('样本索引')
    plt.ylabel('距离')
    plt.show()
    
    # 执行层次聚类
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, labels)
    print(f"轮廓系数: {silhouette_avg:.3f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title('层次聚类结果')
    plt.colorbar(label='簇标签')
    plt.show()
    
    return clustering
```

## 3. 降维技术

### 3.1 主成分分析(PCA)
```python
from sklearn.decomposition import PCA

def pca_analysis(X, n_components=2):
    """PCA降维分析"""
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 计算解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # 可视化结果
    plt.figure(figsize=(12, 4))
    
    # 降维结果
    plt.subplot(121)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title('PCA降维结果')
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    
    # 解释方差比
    plt.subplot(122)
    plt.bar(range(1, len(explained_variance_ratio) + 1), 
            explained_variance_ratio)
    plt.xlabel('主成分')
    plt.ylabel('解释方差比')
    plt.title('各主成分解释方差比')
    
    plt.tight_layout()
    plt.show()
    
    print("各主成分解释方差比:")
    for i, ratio in enumerate(explained_variance_ratio, 1):
        print(f"主成分 {i}: {ratio:.3f}")
    
    return pca, X_pca
```

### 3.2 t-SNE
```python
from sklearn.manifold import TSNE

def tsne_analysis(X, n_components=2, perplexity=30):
    """t-SNE降维分析"""
    # 执行t-SNE
    tsne = TSNE(n_components=n_components, 
                perplexity=perplexity, 
                random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.title('t-SNE降维结果')
    plt.xlabel('t-SNE特征1')
    plt.ylabel('t-SNE特征2')
    plt.colorbar()
    plt.show()
    
    return tsne, X_tsne
```

## 4. 实战案例：客户分群分析

### 4.1 数据准备
```python
# 加载数据
df = pd.read_csv('customer_data.csv')

# 特征工程
features = ['recency', 'frequency', 'monetary']
X = df[features].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 4.2 客户分群
```python
# 使用K-Means进行客户分群
def customer_segmentation(X, n_clusters=4):
    """客户分群分析"""
    # K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 添加聚类标签
    df['Cluster'] = labels
    
    # 分析各簇的特征
    cluster_stats = df.groupby('Cluster')[features].mean()
    
    # 可视化结果
    fig = plt.figure(figsize=(15, 5))
    
    # 散点图
    ax1 = fig.add_subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    
    # 箱线图
    ax2 = fig.add_subplot(132)
    df.boxplot(column=features[0], by='Cluster')
    plt.title(f'{features[0]} by Cluster')
    
    # 热力图
    ax3 = fig.add_subplot(133)
    sns.heatmap(cluster_stats, annot=True, cmap='YlOrRd')
    plt.title('Cluster Characteristics')
    
    plt.tight_layout()
    plt.show()
    
    return kmeans, cluster_stats
```

### 4.3 客户画像分析
```python
def analyze_clusters(df, features, labels):
    """分析客户群体特征"""
    # 计算每个簇的基本统计量
    cluster_stats = df.groupby('Cluster')[features].agg([
        'mean', 'std', 'min', 'max'
    ]).round(2)
    
    # 计算每个簇的大小
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    
    # 可视化簇的大小
    plt.figure(figsize=(10, 6))
    cluster_sizes.plot(kind='bar')
    plt.title('各客户群体规模')
    plt.xlabel('客户群体')
    plt.ylabel('客户数量')
    plt.show()
    
    # 特征分布
    fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
    for i, feature in enumerate(features):
        sns.boxplot(x='Cluster', y=feature, data=df, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return cluster_stats
```

## 常见问题解答

Q: 如何选择合适的聚类算法？
A: 根据数据特点和业务需求选择：
- K-Means适用于球形簇
- DBSCAN适用于不规则形状的簇
- 层次聚类适用于需要层次结构的场景

Q: 如何确定最佳聚类数？
A: 可以使用以下方法：
- 肘部法则
- 轮廓系数
- Gap统计量
- 业务知识

Q: PCA和t-SNE的区别是什么？
A: PCA是线性降维方法，保持全局结构；t-SNE是非线性降维方法，更好地保持局部结构。PCA计算快但可能丢失非线性关系，t-SNE计算慢但能发现复杂的数据模式。

## 扩展阅读
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Scikit-learn聚类算法](https://scikit-learn.org/stable/modules/clustering.html)
- [降维技术综述](https://arxiv.org/abs/1403.2090)
