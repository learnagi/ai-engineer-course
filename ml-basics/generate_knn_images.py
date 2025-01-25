import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def generate_images():
    # 设置matplotlib样式
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 支持中文
    
    # 1. 生成封面图片
    def create_header_image():
        # 生成示例数据
        X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
        
        # 创建测试点
        test_point = np.array([[0, 0]])
        
        # 训练KNN模型
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        
        # 绘制散点图
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.scatter(test_point[:, 0], test_point[:, 1], color='red', 
                   marker='*', s=200, label='待分类样本')
        
        # 找到最近的K个邻居
        distances, indices = knn.kneighbors(test_point)
        
        # 绘制到最近邻的连线
        for idx in indices[0]:
            plt.plot([test_point[0, 0], X[idx, 0]], 
                     [test_point[0, 1], X[idx, 1]], 
                     'r--', alpha=0.3)
        
        plt.title('K近邻算法示例', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.savefig('images/knn/header.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    pad_inches=0.1)
        plt.close()

    def create_distance_comparison():
        # 生成两个点
        point1 = np.array([0, 0])
        point2 = np.array([3, 4])

        # 创建图形
        plt.figure(figsize=(12, 4))

        # 1. 欧氏距离
        plt.subplot(131)
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', label='欧氏距离')
        plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], c='blue')
        plt.grid(True, alpha=0.3)
        plt.title('欧氏距离')

        # 2. 曼哈顿距离
        plt.subplot(132)
        plt.plot([point1[0], point1[0], point2[0]], 
                 [point1[1], point2[1], point2[1]], 'g-', label='曼哈顿距离')
        plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], c='blue')
        plt.grid(True, alpha=0.3)
        plt.title('曼哈顿距离')

        # 3. 闵可夫斯基距离
        plt.subplot(133)
        t = np.linspace(0, 1, 100)
        p = 3  # p=3的闵可夫斯基距离
        x = point1[0] + (point2[0] - point1[0]) * t
        y = point1[1] + (point2[1] - point1[1]) * t
        plt.plot(x, y, 'b-', label=f'闵可夫斯基距离 (p={p})')
        plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], c='blue')
        plt.grid(True, alpha=0.3)
        plt.title(f'闵可夫斯基距离 (p={p})')

        plt.tight_layout()
        plt.savefig('images/knn/distance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_k_value_impact():
        # 生成数据
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                 n_clusters_per_class=1, random_state=42)
        
        # 创建图形
        plt.figure(figsize=(15, 5))
        k_values = [1, 5, 15]
        
        for i, k in enumerate(k_values, 1):
            plt.subplot(1, 3, i)
            
            # 训练模型
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X, y)
            
            # 创建网格点
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            # 预测
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # 绘制决策边界
            plt.contourf(xx, yy, Z, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
            plt.title(f'K = {k}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/knn/k_value_impact.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_scaling_importance():
        # 生成数据
        np.random.seed(42)
        X = np.random.randn(100, 2)
        X[:, 1] = X[:, 1] * 10  # 放大第二个特征的尺度
        y = (X[:, 0] + X[:, 1]/10 > 0).astype(int)  # 创建标签

        # 创建图形
        plt.figure(figsize=(12, 5))

        # 1. 未缩放的数据
        plt.subplot(121)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.title('未进行特征缩放')
        plt.grid(True, alpha=0.3)

        # 2. 缩放后的数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        plt.subplot(122)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
        plt.title('特征缩放后')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('images/knn/scaling_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 确保目录存在
    import os
    os.makedirs('images/knn', exist_ok=True)
    
    # 生成所有图片
    create_header_image()
    create_distance_comparison()
    create_k_value_impact()
    create_scaling_importance()

if __name__ == '__main__':
    generate_images()