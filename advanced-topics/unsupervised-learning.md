# 无监督学习算法详解

## 聚类算法

### 1. K-means聚类

#### 1.1 算法原理
- **基本步骤**
  - 初始化聚类中心
  - 分配样本
  - 更新中心点
  - 迭代优化
- **算法变体**
  - K-means++
  - Mini-batch K-means
  - 核K-means

#### 1.2 实现示例
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成示例数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60)

# 训练模型
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title('K-means聚类结果')
plt.show()
```

#### 1.3 最优K值选择
- **肘部法则**
```python
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('畸变度')
plt.title('肘部法则确定最优K值')
plt.show()
```

### 2. 层次聚类

#### 2.1 算法类型
- **凝聚式层次聚类**
  - 自底向上
  - 合并策略
- **分裂式层次聚类**
  - 自顶向下
  - 分裂准则

#### 2.2 实现示例
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 训练模型
model = AgglomerativeClustering(n_clusters=3)
model.fit(X)

# 绘制树状图
linkage_matrix = linkage(X, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('层次聚类树状图')
plt.show()
```

### 3. DBSCAN

#### 3.1 算法原理
- **核心概念**
  - 核心点
  - 边界点
  - 噪声点
- **参数选择**
  - eps（邻域半径）
  - minPts（最小点数）

#### 3.2 实现示例
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 数据标准化
X_scaled = StandardScaler().fit_transform(X)

# 训练模型
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X_scaled)

# 可视化结果
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan.labels_)
plt.title('DBSCAN聚类结果')
plt.show()
```

## 降维技术

### 1. 主成分分析(PCA)

#### 1.1 数学原理
- **协方差矩阵**
- **特征值分解**
- **主成分选择**

#### 1.2 实现示例
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 加载数据
digits = load_digits()
X = digits.data

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化结果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target)
plt.colorbar()
plt.title('PCA降维结果')
plt.show()

# 查看解释方差比
print('解释方差比：', pca.explained_variance_ratio_)
```

### 2. t-SNE

#### 2.1 算法原理
- **概率分布计算**
- **KL散度优化**
- **梯度下降**

#### 2.2 实现示例
```python
from sklearn.manifold import TSNE

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target)
plt.colorbar()
plt.title('t-SNE降维结果')
plt.show()
```

### 3. 其他降维方法

#### 3.1 LDA
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, digits.target)
```

#### 3.2 UMAP
```python
import umap

reducer = umap.UMAP()
X_umap = reducer.fit_transform(X)
```

## 异常检测

### 1. 统计方法

#### 1.1 Z-score方法
```python
from scipy import stats

z_scores = stats.zscore(X)
outliers = (abs(z_scores) > 3).any(axis=1)
```

#### 1.2 IQR方法
```python
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
```

### 2. 基于密度的方法

#### 2.1 局部异常因子(LOF)
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
y_pred = lof.fit_predict(X)
```

### 3. 孤立森林

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1)
y_pred = iso_forest.fit_predict(X)
```

## 实战项目：客户分群分析

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 加载数据
df = pd.read_csv('customer_data.csv')

# 数据预处理
X = df.drop(['CustomerID'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 聚类
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 可视化结果
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.scatter(X_pca[df['Cluster'] == i, 0],
                X_pca[df['Cluster'] == i, 1],
                label=f'Cluster {i}')
plt.legend()
plt.title('客户分群结果')
plt.show()

# 分析每个群体的特征
print(df.groupby('Cluster').mean())
```

## 课后练习

1. 实现基本的K-means算法
2. 比较不同降维方法的效果
3. 设计异常检测系统
4. 完成客户分群项目

## 延伸阅读

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
2. Aggarwal, C. C. (2015). Data Mining: The Textbook
3. [scikit-learn聚类算法指南](https://scikit-learn.org/stable/modules/clustering.html)

## 下一步学习

- 探索深度学习中的自编码器
- 学习生成对抗网络
- 实践更复杂的无监督学习项目
- 参与数据挖掘竞赛