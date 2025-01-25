import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_math_concepts_image():
    """创建数学概念解释图"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. 平均值和方差可视化
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)
    sns.histplot(data, kde=True, ax=ax1)
    ax1.axvline(np.mean(data), color='r', linestyle='--', label='平均值')
    ax1.axvline(np.mean(data) + np.std(data), color='g', linestyle='--', label='标准差')
    ax1.axvline(np.mean(data) - np.std(data), color='g', linestyle='--')
    ax1.set_title('平均值和标准差')
    ax1.legend()
    
    # 2. 向量运算可视化
    ax2 = fig.add_subplot(gs[0, 1])
    vector1 = np.array([2, 3])
    vector2 = np.array([1, 2])
    ax2.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color='b', label='向量1')
    ax2.quiver(0, 0, vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1, color='r', label='向量2')
    ax2.quiver(0, 0, vector1[0]+vector2[0], vector1[1]+vector2[1], angles='xy', scale_units='xy', scale=1, color='g', label='向量和')
    ax2.set_xlim(-1, 5)
    ax2.set_ylim(-1, 6)
    ax2.grid(True)
    ax2.set_title('向量运算')
    ax2.legend()
    
    # 3. 矩阵乘法可视化
    ax3 = fig.add_subplot(gs[1, 0])
    matrix = np.array([[1, 2], [3, 4]])
    vector = np.array([2, 1])
    result = np.dot(matrix, vector)
    
    ax3.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='b', label='原始向量')
    ax3.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1, color='r', label='变换后')
    ax3.set_xlim(-1, 6)
    ax3.set_ylim(-1, 8)
    ax3.grid(True)
    ax3.set_title('矩阵变换')
    ax3.legend()
    
    # 4. 相关性可视化
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.linspace(0, 10, 100)
    y = 0.8 * x + np.random.normal(0, 1, 100)
    ax4.scatter(x, y, alpha=0.5)
    ax4.set_title(f'相关性: {np.corrcoef(x, y)[0,1]:.2f}')
    
    plt.tight_layout()
    plt.savefig('images/math-concepts.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_math_concepts_image()
    print("数学概念图已生成完成！")
