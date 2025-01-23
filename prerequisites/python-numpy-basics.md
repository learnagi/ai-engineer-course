---
title: "Python和NumPy基础"
slug: "python-numpy-basics"
sequence: 1
description: "掌握机器学习所需的Python编程基础和NumPy科学计算库"
is_published: true
estimated_minutes: 90
language: "zh-CN"
---

![Python and NumPy](images/python-numpy-header.png)
*Python和NumPy是机器学习的基础工具*

# Python和NumPy基础

## 学习目标
完成本模块学习后，你将能够：
- 掌握Python基本语法和数据结构
- 理解NumPy的核心概念和操作
- 使用NumPy进行科学计算
- 处理多维数组数据

## 1. Python基础

### 1.1 基本数据类型
```python
# 数值类型
x = 42           # 整数
y = 3.14         # 浮点数
z = 1 + 2j      # 复数

# 字符串
text = "Hello, World!"
multiline = """
多行
文本
"""

# 布尔值
is_true = True
is_false = False

# 空值
nothing = None
```

### 1.2 数据结构
```python
# 列表
numbers = [1, 2, 3, 4, 5]
mixed = [1, "two", 3.0, [4, 5]]

# 元组
coordinates = (10, 20)
point = (1,)  # 单元素元组

# 字典
person = {
    "name": "Alice",
    "age": 25,
    "skills": ["Python", "ML"]
}

# 集合
unique_numbers = {1, 2, 3, 3, 2, 1}  # 结果：{1, 2, 3}
```

### 1.3 控制流
```python
# if条件语句
def check_number(x):
    if x > 0:
        return "正数"
    elif x < 0:
        return "负数"
    else:
        return "零"

# for循环
def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# while循环
def countdown(n):
    while n > 0:
        print(n)
        n -= 1
```

### 1.4 函数和类
```python
# 函数定义
def greet(name, greeting="Hello"):
    """函数文档字符串"""
    return f"{greeting}, {name}!"

# 类定义
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    @property
    def perimeter(self):
        return 2 * (self.width + self.height)
```

## 2. NumPy基础

### 2.1 创建数组
```python
import numpy as np

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# 特殊数组
zeros = np.zeros((3, 4))       # 全0数组
ones = np.ones((2, 3))         # 全1数组
rand = np.random.rand(2, 2)    # 随机数组
arange = np.arange(10)         # 序列数组
linspace = np.linspace(0, 1, 5)  # 等间隔数组
```

### 2.2 数组操作
```python
# 形状操作
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)          # 形状
print(arr.reshape(3, 2))  # 重塑
print(arr.T)             # 转置

# 索引和切片
print(arr[0, 1])         # 单元素
print(arr[:, 1])         # 列切片
print(arr[0, :])         # 行切片

# 条件索引
mask = arr > 3
print(arr[mask])         # 条件选择
```

### 2.3 数组运算
```python
# 算术运算
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)    # 加法
print(a * b)    # 乘法
print(a ** 2)   # 幂运算
print(np.sqrt(a))  # 开方

# 统计运算
print(arr.sum())       # 求和
print(arr.mean())      # 平均值
print(arr.std())       # 标准差
print(arr.min())       # 最小值
print(arr.max())       # 最大值
```

### 2.4 广播机制
```python
# 广播示例
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
scalar = 2
vector = np.array([10, 20, 30])

print(arr + scalar)      # 数组加标量
print(arr + vector)      # 数组加向量
```

### 2.5 线性代数运算
```python
# 矩阵运算
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(np.dot(a, b))      # 矩阵乘法
print(np.linalg.inv(a))  # 矩阵求逆
print(np.linalg.det(a))  # 行列式
print(np.linalg.eig(a))  # 特征值和特征向量
```

## 3. 实战练习

### 3.1 图像处理
```python
def process_image():
    # 创建简单图像
    image = np.zeros((100, 100))
    image[25:75, 25:75] = 1
    
    # 旋转
    rotated = np.rot90(image)
    
    # 翻转
    flipped = np.flipud(image)
    
    return image, rotated, flipped

# 显示图像
import matplotlib.pyplot as plt

def show_images(images):
    fig, axes = plt.subplots(1, len(images))
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()
```

### 3.2 数据分析
```python
def analyze_data():
    # 生成示例数据
    data = np.random.normal(size=(1000, 5))
    
    # 基本统计
    stats = {
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0),
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    
    # 相关性矩阵
    corr = np.corrcoef(data.T)
    
    return stats, corr

def plot_correlation_matrix(corr):
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(corr)))
    plt.yticks(range(len(corr)))
    plt.show()
```

## 4. 性能优化

### 4.1 向量化操作
```python
# 不推荐的循环方式
def slow_sum(arr1, arr2):
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        result[i] = arr1[i] + arr2[i]
    return result

# 推荐的向量化方式
def fast_sum(arr1, arr2):
    return arr1 + arr2
```

### 4.2 内存优化
```python
# 使用视图而不是复制
def optimize_memory():
    large_array = np.random.rand(1000000)
    
    # 视图 - 共享内存
    view = large_array.view()
    
    # 复制 - 新内存
    copy = large_array.copy()
    
    return view, copy
```

## 常见问题解答

Q: Python列表和NumPy数组的区别？
A: 主要区别：
- NumPy数组支持向量化操作
- NumPy数组内存效率更高
- NumPy数组元素类型必须相同
- NumPy提供更多数学运算功能

Q: 如何选择合适的NumPy数据类型？
A: 考虑以下因素：
- 数值范围
- 精度要求
- 内存限制
- 计算效率

Q: 如何提高NumPy代码性能？
A: 可以采用以下策略：
- 使用向量化操作
- 避免不必要的复制
- 使用适当的数据类型
- 利用内置函数

## 扩展阅读
- [Python官方文档](https://docs.python.org/3/)
- [NumPy官方文档](https://numpy.org/doc/)
- [《Python for Data Analysis》](https://wesmckinney.com/book/)
- [《From Python to NumPy》](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
