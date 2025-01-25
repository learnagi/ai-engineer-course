import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import os

# 设置matplotlib样式
plt.style.use('default')  # 使用默认样式
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['axes.grid'] = True  # 显示网格
plt.rcParams['grid.alpha'] = 0.3  # 网格透明度

def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_sigmoid():
    """生成sigmoid函数图像"""
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.grid(True)
    plt.title('Sigmoid函数')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.text(-8, 0.8, 'σ(z) = 1 / (1 + e^(-z))')
    
    save_path = 'images/classification/sigmoid-function.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_logistic_decision_boundary():
    """生成逻辑回归决策边界"""
    # 生成数据
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, random_state=42)
    
    # 训练模型
    model = LogisticRegression()
    model.fit(X, y)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # 预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('逻辑回归决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    save_path = 'images/classification/logistic-decision-boundary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_decision_tree_boundary():
    """生成决策树决策边界"""
    # 生成非线性数据
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    
    # 训练模型
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # 预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('决策树决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    save_path = 'images/classification/decision-tree-boundary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_svm_boundary():
    """生成SVM决策边界"""
    # 生成非线性数据
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    
    # 训练模型
    model = SVC(kernel='rbf')
    model.fit(X, y)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # 预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('SVM决策边界 (RBF核)')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    save_path = 'images/classification/svm-boundary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_knn_example():
    """生成KNN分类示例图"""
    # 生成示例数据
    X_train = np.array([
        [10, 2], [8, 3],    # 动作片
        [2, 8], [3, 10],    # 爱情片
    ])
    X_test = np.array([[5, 5]])  # 待分类的电影
    y_train = np.array([0, 0, 1, 1])  # 0表示动作片，1表示爱情片
    
    plt.figure(figsize=(10, 6))
    
    # 绘制已知点
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
               c='blue', marker='o', s=100, label='动作片')
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
               c='red', marker='s', s=100, label='爱情片')
    
    # 绘制待分类点
    plt.scatter(X_test[0, 0], X_test[0, 1], 
               c='gray', marker='*', s=200, label='待分类')
    
    # 绘制到最近的3个点的连线
    nbrs = NearestNeighbors(n_neighbors=3).fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)
    
    for idx in indices[0]:
        plt.plot([X_test[0, 0], X_train[idx, 0]], 
                [X_test[0, 1], X_train[idx, 1]], 
                'k--', alpha=0.3)
    
    # 添加标签和标题
    plt.xlabel('动作场景数量')
    plt.ylabel('爱情场景数量')
    plt.title('KNN分类示例 (K=3)')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    save_path = 'images/classification/knn-example.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """生成所有图片"""
    # 确保图片目录存在
    ensure_directory('images/classification')
    
    # 生成图片
    generate_sigmoid()
    generate_logistic_decision_boundary()
    generate_decision_tree_boundary()
    generate_svm_boundary()
    generate_knn_example()
    
    print("所有图片生成完成！")

if __name__ == '__main__':
    main()
