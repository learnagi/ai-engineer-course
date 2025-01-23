---
title: "经典机器学习算法"
slug: "classic-ml"
sequence: 8
description: "常用机器学习算法的原理和实现，包括监督学习、无监督学习算法及其实践应用"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

# 经典机器学习算法

## 课程介绍
本模块深入讲解常用机器学习算法的原理和实现，通过实际案例帮助你掌握各类算法的特点和应用场景。

## 学习目标
完成本模块学习后，你将能够：
- 理解主流机器学习算法的原理
- 掌握算法选择的方法
- 实现基本的机器学习算法
- 应用算法解决实际问题

## 1. 监督学习算法

### 1.1 线性模型
```python
# 📈 实战案例：线性模型实现
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class LinearRegressionFromScratch(BaseEstimator, RegressorMixin):
    """从零实现线性回归"""
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = []
    
    def fit(self, X, y):
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.n_iterations):
            # 预测
            y_pred = self._forward(X)
            
            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 记录损失
            loss = np.mean((y_pred - y) ** 2)
            self.history.append(loss)
        
        return self
    
    def _forward(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        return self._forward(X)

# 使用示例
def linear_model_demo():
    """线性模型示例"""
    from sklearn.datasets import make_regression
    
    # 生成数据
    X, y = make_regression(n_samples=100, n_features=1, noise=10)
    
    # 训练模型
    model = LinearRegressionFromScratch()
    model.fit(X, y)
    
    # 可视化结果
    plt.figure(figsize=(10, 5))
    
    # 绘制数据点
    plt.subplot(1, 2, 1)
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), 'r')
    plt.title('拟合结果')
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(model.history)
    plt.title('损失曲线')
    
    return plt.gcf()
```

### 1.2 决策树与集成学习
```python
# 🌳 实战案例：决策树与随机森林
class DecisionTreeFromScratch:
    """从零实现决策树"""
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1:
            return {'value': np.bincount(y).argmax()}
        
        # 寻找最佳分割
        best_gain = -1
        best_split = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)
        
        if best_split is None:
            return {'value': np.bincount(y).argmax()}
        
        # 分割数据
        feature, threshold = best_split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_tree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _information_gain(self, y, X_column, threshold):
        """计算信息增益"""
        parent_entropy = self._entropy(y)
        
        # 分割数据
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
        
        # 计算子节点熵
        n = len(y)
        n_l, n_r = len(y[left_mask]), len(y[right_mask])
        e_l, e_r = self._entropy(y[left_mask]), self._entropy(y[right_mask])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        """计算熵"""
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        """遍历决策树进行预测"""
        if 'value' in node:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])

# 随机森林实现
class RandomForestFromScratch:
    """从零实现随机森林"""
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # 随机抽样（Bootstrap）
            indices = np.random.choice(len(X), len(X), replace=True)
            sample_X = X[indices]
            sample_y = y[indices]
            
            # 训练决策树
            tree = DecisionTreeFromScratch(max_depth=self.max_depth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)
        return self
    
    def predict(self, X):
        # 收集所有树的预测
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # 多数投票
        return np.array([np.bincount(pred).argmax() 
                        for pred in predictions.T])
```

## 2. 无监督学习算法

### 2.1 K-means聚类
```python
# 🎯 实战案例：K-means聚类实现
class KMeansFromScratch:
    """从零实现K-means聚类"""
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
    
    def fit(self, X):
        # 随机初始化质心
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # 分配样本到最近的质心
            distances = self._calculate_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # 更新质心
            new_centroids = np.array([X[labels == k].mean(axis=0) 
                                    for k in range(self.n_clusters)])
            
            # 检查收敛
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
        
        return self
    
    def _calculate_distances(self, X):
        """计算样本到所有质心的距离"""
        distances = np.zeros((len(X), self.n_clusters))
        for k, centroid in enumerate(self.centroids):
            distances[:, k] = np.sqrt(((X - centroid) ** 2).sum(axis=1))
        return distances
    
    def predict(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)

# 聚类可视化
def visualize_clusters(X, labels):
    """可视化聚类结果"""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('聚类结果')
    return plt.gcf()
```

### 2.2 主成分分析(PCA)
```python
# 📊 实战案例：PCA实现
class PCAFromScratch:
    """从零实现PCA"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # 中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 选择前n_components个特征向量
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]
        
        # 计算解释方差比
        self.explained_variance_ratio_ = eigenvalues[idx[:self.n_components]] / \
                                       np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components.T) + self.mean

# PCA可视化
def visualize_pca(X, y):
    """可视化PCA结果"""
    pca = PCAFromScratch(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制降维结果
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('PCA降维结果')
    
    # 绘制解释方差比
    plt.subplot(1, 2, 2)
    plt.bar(range(len(pca.explained_variance_ratio_)), 
            pca.explained_variance_ratio_)
    plt.title('解释方差比')
    
    return plt.gcf()
```

## 实战项目：客户分类系统

### 项目描述
构建一个客户分类系统，结合监督和无监督学习方法，实现客户群体的分析和预测。

### 项目代码框架
```python
class CustomerClassificationSystem:
    def __init__(self):
        self.clustering_model = None
        self.classification_model = None
        self.pca_model = None
    
    def preprocess_data(self, data):
        """数据预处理"""
        # 处理缺失值
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)
        
        # 标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)
        
        return data_scaled
    
    def cluster_customers(self, X, n_clusters=5):
        """客户聚类"""
        # 降维
        self.pca_model = PCAFromScratch(n_components=2)
        X_pca = self.pca_model.fit_transform(X)
        
        # 聚类
        self.clustering_model = KMeansFromScratch(n_clusters=n_clusters)
        clusters = self.clustering_model.fit_predict(X_pca)
        
        return clusters, X_pca
    
    def train_classifier(self, X, y):
        """训练分类器"""
        # 使用随机森林分类器
        self.classification_model = RandomForestFromScratch(
            n_trees=100, max_depth=10
        )
        self.classification_model.fit(X, y)
    
    def analyze_customer_groups(self, X, clusters):
        """分析客户群体特征"""
        analysis = {}
        for cluster in range(max(clusters) + 1):
            cluster_data = X[clusters == cluster]
            analysis[f'Cluster_{cluster}'] = {
                'size': len(cluster_data),
                'mean': np.mean(cluster_data, axis=0),
                'std': np.std(cluster_data, axis=0)
            }
        return analysis
    
    def visualize_results(self, X_pca, clusters, feature_names=None):
        """可视化结果"""
        plt.figure(figsize=(15, 5))
        
        # 聚类结果
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('客户群体分布')
        
        # 特征重要性
        if self.classification_model and feature_names is not None:
            importances = np.mean([tree.feature_importances_ 
                                 for tree in self.classification_model.trees], 
                                axis=0)
            plt.subplot(1, 3, 2)
            plt.bar(range(len(importances)), importances)
            plt.xticks(range(len(importances)), feature_names, rotation=45)
            plt.title('特征重要性')
        
        # 群体大小对比
        plt.subplot(1, 3, 3)
        cluster_sizes = np.bincount(clusters)
        plt.pie(cluster_sizes, labels=[f'Group {i}' 
                for i in range(len(cluster_sizes))])
        plt.title('群体规模对比')
        
        return plt.gcf()
```

## 练习与作业
1. 实现逻辑回归算法
2. 优化决策树的分裂策略
3. 实现DBSCAN聚类算法

## 扩展阅读
- [统计学习方法](https://book.douban.com/subject/33437381/)
- [机器学习实战](https://book.douban.com/subject/24703171/)
- [scikit-learn算法文档](https://scikit-learn.org/stable/supervised_learning.html)

## 小测验
1. 决策树的优缺点是什么？
2. K-means算法的局限性有哪些？
3. 如何选择合适的聚类算法？

## 下一步学习
- 深度学习基础
- 集成学习进阶
- 模型部署实践

## 常见问题解答
Q: 如何处理决策树的过拟合问题？
A: 可以通过设置最大深度、最小样本数、剪枝等方法来控制决策树的复杂度。

Q: 什么情况下应该使用无监督学习？
A: 当数据没有标签、需要发现数据内在结构、或需要降维时，可以考虑使用无监督学习方法。
