import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

def setup_style():
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def save_figure(name, dpi=300):
    """保存图片到指定目录，确保目录存在"""
    save_path = os.path.join('images', 'knn', f'{name}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def generate_knn_decision_boundary():
    """生成KNN决策边界示例图"""
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2)),  # 类别A
        np.random.normal(loc=[6, 6], scale=0.5, size=(30, 2))   # 类别B
    ])
    y = np.concatenate([np.zeros(30), np.ones(30)])
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='类别A')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='类别B')
    plt.title('KNN决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    
    plt.subplot(122)
    test_point = np.array([[4, 4]])
    distances, indices = knn.kneighbors(test_point)
    
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='类别A')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='类别B')
    plt.scatter(test_point[:, 0], test_point[:, 1], c='green', marker='*', s=200, label='测试样本')
    
    circle = plt.Circle((test_point[0, 0], test_point[0, 1]), 
                       radius=distances[0].max(),
                       fill=False, linestyle='--', color='green')
    plt.gca().add_artist(circle)
    plt.title('K=5的邻居范围')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    
    plt.tight_layout()
    save_figure('decision-boundary')

def generate_k_effect():
    """生成不同K值效果对比图"""
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2)),
        np.random.normal(loc=[6, 6], scale=0.5, size=(30, 2))
    ])
    y = np.concatenate([np.zeros(30), np.ones(30)])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    k_values = [1, 3, 7, 15]
    
    for k, ax in zip(k_values, axes.ravel()):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='类别A')
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='类别B')
        ax.set_title(f'K = {k}')
        ax.legend()
    
    plt.tight_layout()
    save_figure('k-effect')

def generate_scaling_effect():
    """生成特征缩放效果对比图"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    X[:, 0] = X[:, 0] * 100  # 第一个特征的尺度较大
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='类别A')
    ax1.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='类别B')
    ax1.set_title('原始数据')
    ax1.set_xlabel('特征1 (大尺度)')
    ax1.set_ylabel('特征2 (小尺度)')
    ax1.legend()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ax2.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], c='blue', label='类别A')
    ax2.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], c='red', label='类别B')
    ax2.set_title('标准化后的数据')
    ax2.set_xlabel('特征1 (标准化)')
    ax2.set_ylabel('特征2 (标准化)')
    ax2.legend()
    
    plt.tight_layout()
    save_figure('scaling-effect')

def generate_curse_of_dimensionality():
    """生成维度灾难示意图"""
    dimensions = np.arange(1, 11)
    avg_distances = []
    
    for d in dimensions:
        points = np.random.uniform(0, 1, (1000, d))
        distances = []
        for i in range(100):
            dist = np.linalg.norm(points[i] - points, axis=1)
            distances.extend(dist)
        avg_distances.append(np.mean(distances))
    
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, avg_distances, marker='o')
    plt.title('维度灾难：随着维度增加，点与点之间的平均距离增大')
    plt.xlabel('维度')
    plt.ylabel('平均距离')
    plt.grid(True)
    save_figure('curse-of-dimensionality')

def generate_header():
    """生成KNN算法教程的封面图片"""
    np.random.seed(42)
    
    # 生成示例数据
    X = np.concatenate([
        np.random.normal(loc=[3, 3], scale=0.5, size=(50, 2)),  # 类别A
        np.random.normal(loc=[7, 7], scale=0.5, size=(50, 2))   # 类别B
    ])
    y = np.concatenate([np.zeros(50), np.ones(50)])
    
    plt.figure(figsize=(16, 9))  # 16:9的宽屏比例
    
    # 绘制样本点
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='#2ecc71', s=100, label='类别A', alpha=0.6)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='#e74c3c', s=100, label='类别B', alpha=0.6)
    
    # 添加一个突出的测试点
    test_point = np.array([[5, 5]])
    plt.scatter(test_point[:, 0], test_point[:, 1], c='#3498db', 
               marker='*', s=500, label='新样本', zorder=10)
    
    # 添加K近邻示意圈
    circle = plt.Circle((test_point[0, 0], test_point[0, 1]), 
                       radius=2.5,
                       fill=False, linestyle='--', color='#3498db', 
                       linewidth=2, alpha=0.8)
    plt.gca().add_artist(circle)
    
    # 设置图表样式
    plt.title('K近邻算法', fontsize=24, pad=20)
    plt.xlabel('特征1', fontsize=14)
    plt.ylabel('特征2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    save_figure('header')

def main():
    """生成所有图片"""
    setup_style()
    
    # 生成所有图片
    generate_header()
    generate_knn_decision_boundary()
    generate_k_effect()
    generate_scaling_effect()
    generate_curse_of_dimensionality()

if __name__ == '__main__':
    main()
