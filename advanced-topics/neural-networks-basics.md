# 神经网络基础

## 学习目标
1. 理解人工神经网络的基本概念和结构
2. 掌握前向传播和反向传播算法的原理
3. 能够实现简单的神经网络模型
4. 了解神经网络的训练技巧和优化方法

## 知识要点

### 1. 神经网络概述
#### 1.1 什么是人工神经网络
- 生物神经元与人工神经元的对比
- 神经网络的发展历史
- 神经网络的基本特点和应用场景

#### 1.2 神经网络的基本组成
- 神经元（节点）
- 权重和偏置
- 激活函数
- 网络层次结构

### 2. 前向传播算法
#### 2.1 单个神经元的计算
- 输入加权求和
- 偏置项的作用
- 激活函数的选择和特点
  - Sigmoid函数
  - ReLU函数
  - tanh函数

#### 2.2 多层网络的前向传播
- 输入层到隐藏层的计算
- 隐藏层到输出层的计算
- 矩阵形式的表达

### 3. 反向传播算法
#### 3.1 损失函数
- 均方误差（MSE）
- 交叉熵损失
- 损失函数的选择原则

#### 3.2 梯度下降优化
- 计算梯度
- 链式法则
- 参数更新规则

#### 3.3 反向传播过程
- 输出层误差计算
- 误差反向传播
- 权重和偏置更新

### 4. 神经网络训练
#### 4.1 训练流程
- 数据预处理
- 参数初始化
- 批量训练
- 学习率调整

#### 4.2 优化技巧
- 批量归一化
- 权重初始化方法
- 学习率调度
- 正则化技术

#### 4.3 过拟合处理
- Dropout技术
- L1/L2正则化
- 早停法

## 代码实践

### 1. 使用NumPy实现简单神经网络
```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        # 反向传播
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.z2_error = np.dot(self.output_delta, self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid_derivative(self.a1)
        
        # 更新权重和偏置
        self.W2 += np.dot(self.a1.T, self.output_delta)
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True)
        self.W1 += np.dot(X.T, self.z2_delta)
        self.b1 += np.sum(self.z2_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate=0.1):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss}')
```

### 2. 实践示例：XOR问题求解
```python
# 准备XOR数据
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 创建并训练神经网络
nn = SimpleNeuralNetwork(2, 4, 1)
nn.train(X, y, epochs=10000)

# 测试结果
print("\nPredictions:")
for i in range(len(X)):
    prediction = nn.forward(X[i:i+1])
    print(f"Input: {X[i]}, Predicted: {prediction[0]}, Actual: {y[i]}")
```

## 课后作业

1. 理论题
   - 解释神经网络中激活函数的作用，并比较不同激活函数的优缺点
   - 描述反向传播算法的工作原理，并说明其在神经网络训练中的重要性
   - 分析过拟合问题的产生原因，并总结常用的解决方法

2. 编程练习
   - 修改示例代码，尝试使用不同的激活函数（如ReLU、tanh）
   - 为神经网络添加一个隐藏层，观察模型性能的变化
   - 实现动态学习率调整机制

3. 实战项目
   - 使用MNIST数据集实现手写数字识别
   - 要求：
     * 构建多层神经网络
     * 实现批量训练
     * 添加正则化方法
     * 可视化训练过程
     * 评估模型性能

## 参考资源
- 《Neural Networks and Deep Learning》by Michael Nielsen
- Stanford CS231n 课程
- PyTorch官方教程
- TensorFlow神经网络指南

## 下一步学习
- 深入学习卷积神经网络（CNN）
- 探索循环神经网络（RNN）
- 研究高级优化算法
- 实践更复杂的神经网络架构