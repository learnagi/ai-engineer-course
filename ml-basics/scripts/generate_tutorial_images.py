import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 确保可以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def create_house_price_example():
    """创建房屋价格线性回归示例图"""
    # 生成示例数据
    np.random.seed(42)
    areas = np.linspace(50, 200, 20)
    prices = 2000 * areas + 100000 + np.random.normal(0, 50000, 20)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.scatter(areas, prices, color='blue', alpha=0.6, label='房屋数据')
    
    # 添加趋势线
    model = LinearRegression()
    model.fit(areas.reshape(-1, 1), prices)
    line_x = np.array([0, 250])
    line_y = model.predict(line_x.reshape(-1, 1))
    plt.plot(line_x, line_y, color='red', linestyle='--', label='价格趋势')
    
    plt.xlabel('房屋面积（平方米）')
    plt.ylabel('房屋价格（元）')
    plt.title('房屋面积与价格的线性关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/linear-regression-house-example.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_formula_explanation():
    """创建线性回归公式的几何解释图"""
    # 生成数据
    np.random.seed(42)
    x = np.linspace(0, 10, 10)
    y = 2 * x + 1 + np.random.normal(0, 1, 10)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制数据点
    plt.scatter(x, y, color='blue', s=100, label='数据点')
    
    # 绘制拟合线
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    line_y = model.predict(x.reshape(-1, 1))
    plt.plot(x, line_y, color='red', linewidth=2, label='拟合线 y = wx + b')
    
    # 添加标注
    plt.annotate('w: 斜率', xy=(5, 11), xytext=(6, 13),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('b: 截距', xy=(0, model.intercept_), xytext=(1, 0),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('x (特征)')
    plt.ylabel('y (目标)')
    plt.title('线性回归公式的几何意义')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/linear-regression-formula.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_mse_explanation():
    """创建均方误差(MSE)的可视化解释"""
    # 生成数据
    np.random.seed(42)
    x = np.linspace(0, 10, 8)
    y_true = 2 * x + 1
    y_pred = 1.8 * x + 2
    
    plt.figure(figsize=(12, 6))
    
    # 绘制真实值和预测值
    plt.scatter(x, y_true, color='blue', s=100, label='实际值')
    plt.plot(x, y_pred, color='red', linewidth=2, label='预测线')
    
    # 绘制误差
    for i in range(len(x)):
        plt.vlines(x[i], y_true[i], y_pred[i], colors='gray', linestyles='--', alpha=0.5)
        error = y_pred[i] - y_true[i]
        plt.annotate(f'误差²: {error**2:.1f}', 
                    xy=(x[i], (y_true[i] + y_pred[i])/2),
                    xytext=(x[i]+0.2, (y_true[i] + y_pred[i])/2))
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('均方误差(MSE)的计算')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/mse-explanation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_overfitting_visualization():
    """创建过拟合、欠拟合和良好拟合的对比图"""
    # 生成数据
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y = 3 * np.sin(x/2) + np.random.normal(0, 0.5, 20)
    
    plt.figure(figsize=(15, 5))
    
    # 欠拟合（使用线性模型）
    plt.subplot(131)
    plt.scatter(x, y, color='blue', alpha=0.6, label='数据点')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r-', label='欠拟合模型')
    plt.title('欠拟合')
    plt.legend()
    
    # 良好拟合
    plt.subplot(132)
    plt.scatter(x, y, color='blue', alpha=0.6, label='数据点')
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r-', label='良好拟合模型')
    plt.title('良好拟合')
    plt.legend()
    
    # 过拟合
    plt.subplot(133)
    plt.scatter(x, y, color='blue', alpha=0.6, label='数据点')
    z = np.polyfit(x, y, 15)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r-', label='过拟合模型')
    plt.title('过拟合')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/overfitting-underfitting.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_multiple_regression_visualization():
    """创建多元线性回归的3D可视化"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # 生成示例数据
    np.random.seed(42)
    x1 = np.random.rand(100) * 10
    x2 = np.random.rand(100) * 10
    y = 2 * x1 + 3 * x2 + np.random.normal(0, 1, 100)
    
    # 创建3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制数据点
    scatter = ax.scatter(x1, x2, y, c='blue', alpha=0.6)
    
    # 创建网格点
    x1_grid, x2_grid = np.meshgrid(np.linspace(0, 10, 10),
                                  np.linspace(0, 10, 10))
    y_grid = 2 * x1_grid + 3 * x2_grid
    
    # 绘制拟合平面
    surface = ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3,
                            cmap='viridis')
    
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.set_zlabel('目标值')
    ax.set_title('多元线性回归的3D可视化')
    
    plt.savefig('images/multiple-linear-regression.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 生成所有图片
    create_house_price_example()
    create_formula_explanation()
    create_mse_explanation()
    create_overfitting_visualization()
    create_multiple_regression_visualization()
    print("所有教程图片已生成完成！")
