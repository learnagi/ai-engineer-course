import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

def generate_sample_data(n_samples=50, noise=0.5):
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 0.5 * X.ravel() + np.sin(X.ravel()) + np.random.normal(0, noise, n_samples)
    return X, y

def plot_regularization_comparison():
    # 生成数据
    X, y = generate_sample_data(n_samples=50, noise=0.3)
    X_poly = PolynomialFeatures(degree=5).fit_transform(X)
    
    # 创建不同的模型
    models = {
        '无正则化': LinearRegression(),
        'Ridge回归': Ridge(alpha=1.0),
        'Lasso回归': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    for idx, (name, model) in enumerate(models.items(), 1):
        plt.subplot(2, 2, idx)
        model.fit(X_poly, y)
        
        # 生成平滑的预测线
        X_test = np.linspace(0, 10, 200).reshape(-1, 1)
        X_test_poly = PolynomialFeatures(degree=5).fit_transform(X_test)
        y_pred = model.predict(X_test_poly)
        
        plt.scatter(X, y, color='blue', alpha=0.5, label='数据点')
        plt.plot(X_test, y_pred, 'r-', label='预测线')
        plt.title(f'{name}的拟合效果')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        
        # 添加模型系数信息
        coef_text = f'非零系数数量: {np.sum(np.abs(model.coef_) > 1e-10)}'
        plt.text(0.05, 0.95, coef_text, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('images/regularization_comparison.png')
    plt.close()

def plot_glm_concepts():
    # 生成数据
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    
    # 创建图形
    plt.figure(figsize=(15, 5))
    
    # 1. 逻辑函数
    plt.subplot(131)
    y_logistic = 1 / (1 + np.exp(-X))
    plt.plot(X, y_logistic)
    plt.title('逻辑函数 (Logistic)')
    plt.xlabel('X')
    plt.ylabel('概率')
    plt.grid(True)
    
    # 2. 泊松分布
    plt.subplot(132)
    lambda_vals = [1, 2, 4]
    x = np.arange(0, 10)
    for lambda_val in lambda_vals:
        pmf = np.exp(-lambda_val) * lambda_val**x / np.array([np.math.factorial(i) for i in x])
        plt.plot(x, pmf, label=f'λ={lambda_val}')
    plt.title('泊松分布')
    plt.xlabel('事件数量')
    plt.ylabel('概率')
    plt.legend()
    plt.grid(True)
    
    # 3. GAM示例
    plt.subplot(133)
    X = np.linspace(0, 10, 100)
    y1 = np.sin(X)
    y2 = 0.5 * X
    y_combined = y1 + y2
    plt.plot(X, y1, '--', label='非线性项')
    plt.plot(X, y2, '--', label='线性项')
    plt.plot(X, y_combined, '-', label='组合效果')
    plt.title('广义加性模型(GAM)示例')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('images/glm_concepts.png')
    plt.close()

def plot_overfitting_detection():
    # 生成训练数据
    np.random.seed(42)
    X_train = np.sort(np.random.uniform(0, 10, 20)).reshape(-1, 1)
    y_train = 0.5 * X_train.ravel() + np.sin(X_train.ravel()) + np.random.normal(0, 0.2, 20)
    
    # 生成测试数据
    X_test = np.sort(np.random.uniform(0, 10, 50)).reshape(-1, 1)
    y_test = 0.5 * X_test.ravel() + np.sin(X_test.ravel()) + np.random.normal(0, 0.2, 50)
    
    # 创建不同程度的多项式特征
    degrees = [1, 3, 15]
    plt.figure(figsize=(15, 5))
    
    for idx, degree in enumerate(degrees, 1):
        plt.subplot(1, 3, idx)
        
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # 计算训练集和测试集得分
        train_score = model.score(X_train_poly, y_train)
        test_score = model.score(X_test_poly, y_test)
        
        # 生成平滑的预测线
        X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
        X_plot_poly = poly_features.transform(X_plot)
        y_plot = model.predict(X_plot_poly)
        
        plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='训练数据')
        plt.scatter(X_test, y_test, color='green', alpha=0.5, label='测试数据')
        plt.plot(X_plot, y_plot, 'r-', label='模型预测')
        
        plt.title(f'{degree}次多项式\n训练R²: {train_score:.3f}\n测试R²: {test_score:.3f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/overfitting_detection.png')
    plt.close()

if __name__ == '__main__':
    # 确保images目录存在
    import os
    os.makedirs('images', exist_ok=True)
    
    # 生成所有图片
    plot_regularization_comparison()
    plot_glm_concepts()
    plot_overfitting_detection()
