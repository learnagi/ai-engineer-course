import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_header_image():
    """创建线性回归的题图，包含多个子图展示不同概念"""
    
    # 创建一个大图，包含多个子图
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 4, figure=fig)
    
    # 1. 简单线性回归示例
    ax1 = fig.add_subplot(gs[:, :2])
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 1.5, 50)
    ax1.scatter(x, y, color='blue', alpha=0.6, label='数据点')
    ax1.plot(x, 2*x + 1, color='red', linestyle='--', label='真实关系')
    ax1.set_title('简单线性回归', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 多元回归的3D可视化
    ax2 = fig.add_subplot(gs[0, 2:], projection='3d')
    x1 = np.linspace(0, 10, 20)
    x2 = np.linspace(0, 10, 20)
    X1, X2 = np.meshgrid(x1, x2)
    Y = 2 * X1 + 3 * X2
    ax2.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)
    ax2.set_title('多元线性回归', fontsize=14)
    
    # 3. 残差分布图
    ax3 = fig.add_subplot(gs[1, 2])
    residuals = y - (2*x + 1)
    sns.histplot(residuals, kde=True, ax=ax3)
    ax3.set_title('残差分布', fontsize=14)
    
    # 4. 正则化效果对比
    ax4 = fig.add_subplot(gs[1, 3])
    x_reg = np.linspace(0, 10, 100)
    y_reg = np.sin(x_reg) + np.random.normal(0, 0.1, 100)
    ax4.scatter(x_reg, y_reg, color='blue', alpha=0.3, s=30)
    
    # 添加不同程度的拟合曲线
    z1 = np.polyfit(x_reg, y_reg, 1)
    z2 = np.polyfit(x_reg, y_reg, 15)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    ax4.plot(x_reg, p1(x_reg), 'r-', label='简单模型')
    ax4.plot(x_reg, p2(x_reg), 'g-', label='复杂模型')
    ax4.set_title('模型复杂度对比', fontsize=14)
    ax4.legend()
    
    # 设置整体标题
    fig.suptitle('线性回归：从简单到复杂', fontsize=16, y=1.02)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('images/linear-regression-header.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

if __name__ == "__main__":
    create_header_image()
    print("题图已生成完成！")
