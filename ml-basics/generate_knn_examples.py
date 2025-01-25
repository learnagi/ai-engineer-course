"""
生成KNN教程中的所有示例和可视化图片
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import seaborn as sns
from collections import Counter
import os

# 设置matplotlib样式
plt.style.use('default')  # 使用默认样式
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置字体以支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def ensure_directory(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def generate_distance_comparison():
    """生成不同距离度量方法的比较图"""
    # 创建网格点
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # 中心点
    center = np.array([0, 0])
    
    # 计算不同距离
    euclidean = np.sqrt(X**2 + Y**2)
    manhattan = np.abs(X) + np.abs(Y)
    minkowski = np.power(np.power(np.abs(X), 3) + np.power(np.abs(Y), 3), 1/3)
    
    # 绘制等值线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.contour(X, Y, euclidean, levels=[1, 2, 3])
    plt.title('欧氏距离')
    plt.axis('equal')
    
    plt.subplot(132)
    plt.contour(X, Y, manhattan, levels=[1, 2, 3])
    plt.title('曼哈顿距离')
    plt.axis('equal')
    
    plt.subplot(133)
    plt.contour(X, Y, minkowski, levels=[1, 2, 3])
    plt.title('闵可夫斯基距离(p=3)')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('images/knn/distance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_k_value_impact():
    """生成K值对决策边界的影响图"""
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # 添加一些噪声点
    noise = np.random.randn(20, 2)
    X = np.vstack([X, noise])
    y = np.hstack([y, np.random.randint(0, 2, 20)])
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # 不同的K值
    k_values = [1, 5, 15]
    
    plt.figure(figsize=(15, 5))
    for i, k in enumerate(k_values, 1):
        # 训练模型
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        
        # 预测
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        plt.subplot(1, 3, i)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title(f'K = {k}')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
    
    plt.tight_layout()
    plt.savefig('images/knn/k_value_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_scaling_importance():
    """生成特征缩放重要性的图"""
    # 生成不同尺度的数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    X[:, 1] = X[:, 1] * 10  # 放大第二个特征
    y = (X[:, 0] + X[:, 1]/10 > 0).astype(int)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    plt.figure(figsize=(12, 5))
    
    # 原始数据
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
    plt.title('原始数据')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    # 标准化后的数据
    plt.subplot(122)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, alpha=0.6)
    plt.title('标准化后的数据')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    plt.tight_layout()
    plt.savefig('images/knn/scaling_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_cv_results():
    """生成交叉验证结果图"""
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # 不同的K值
    k_values = list(range(1, 21, 2))
    cv_scores = []
    
    # 计算每个K值的交叉验证分数
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5)
        cv_scores.append(scores.mean())
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv_scores, 'o-')
    plt.xlabel('K值')
    plt.ylabel('交叉验证准确率')
    plt.title('不同K值的模型性能比较')
    plt.grid(True)
    
    plt.savefig('images/knn/cv_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """生成所有图片"""
    # 确保图片目录存在
    ensure_directory('images/knn')
    
    # 生成所有图片
    generate_distance_comparison()
    generate_k_value_impact()
    generate_scaling_importance()
    generate_cv_results()
    
    print("所有图片生成完成！")

if __name__ == "__main__":
    main()
