---
title: "神经网络基础"
slug: "neural-networks"
sequence: 1
description: "深度学习的核心概念和基础组件，包括神经网络结构、激活函数、反向传播等基础知识"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

# 神经网络基础

## 学习目标

完成本节学习后，你将能够：
- 理解神经网络的基本结构和工作原理
- 掌握前向传播和反向传播算法
- 实现常用的激活函数和损失函数
- 构建和训练简单的神经网络

## 1. 神经网络结构

### 1.1 神经元模型

神经元是神经网络的基本计算单元，其结构包括：
- 输入特征和权重
- 加权求和
- 激活函数

```python
import numpy as np

class Neuron:
    """单个神经元的实现"""
    def __init__(self, n_inputs):
        # 使用He初始化
        self.weights = np.random.randn(n_inputs) * np.sqrt(2.0/n_inputs)
        self.bias = 0
        
        # 存储中间值用于反向传播
        self.x = None
        self.output = None
    
    def forward(self, x):
        """前向传播"""
        self.x = x
        # 计算加权和
        z = np.dot(x, self.weights) + self.bias
        # 使用ReLU激活函数
        self.output = np.maximum(0, z)
        return self.output
    
    def backward(self, grad_output):
        """反向传播"""
        # ReLU的梯度
        grad_z = grad_output * (self.output > 0)
        
        # 计算梯度
        grad_weights = np.outer(self.x, grad_z)
        grad_bias = np.sum(grad_z)
        grad_x = np.dot(grad_z, self.weights.T)
        
        return grad_x, grad_weights, grad_bias
```

### 1.2 神经网络层

神经网络层是多个神经元的集合，常见的层类型包括：
- 全连接层
- 卷积层
- 池化层

```python
class Layer:
    """神经网络层的基类"""
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class FCLayer(Layer):
    """全连接层实现"""
    def __init__(self, n_inputs, n_units):
        super().__init__()
        # He初始化
        self.weights = np.random.randn(n_inputs, n_units) * np.sqrt(2.0/n_inputs)
        self.bias = np.zeros(n_units)
        
        # 优化器状态
        self.momentum_w = np.zeros_like(self.weights)
        self.momentum_b = np.zeros_like(self.bias)
    
    def forward(self, x):
        """前向传播"""
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def backward(self, grad_output):
        """反向传播"""
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input, grad_weights, grad_bias
```

## 2. 激活函数

激活函数为神经网络引入非线性变换，常用的激活函数包括：
- ReLU
- Sigmoid
- Tanh

```python
class Activation:
    """激活函数集合"""
    @staticmethod
    def relu(x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU导数"""
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid导数"""
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """Tanh激活函数"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Tanh导数"""
        return 1 - np.tanh(x)**2
```

## 3. 损失函数

损失函数用于衡量模型预测值与真实值之间的差异：

```python
class Loss:
    """损失函数集合"""
    @staticmethod
    def mse(y_pred, y_true):
        """均方误差损失"""
        return np.mean((y_pred - y_true) ** 2)
    
    @staticmethod
    def mse_derivative(y_pred, y_true):
        """均方误差损失导数"""
        return 2 * (y_pred - y_true) / y_pred.size
    
    @staticmethod
    def cross_entropy(y_pred, y_true):
        """交叉熵损失"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))
    
    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        """交叉熵损失导数"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred
```

## 4. 神经网络实现

完整的神经网络实现，包括前向传播和反向传播：

```python
class NeuralNetwork:
    """简单的神经网络实现"""
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None
    
    def add(self, layer):
        """添加层"""
        self.layers.append(layer)
    
    def set_loss(self, loss, loss_derivative):
        """设置损失函数"""
        self.loss = loss
        self.loss_derivative = loss_derivative
    
    def predict(self, x):
        """预测"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        """训练模型"""
        for epoch in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # 前向传播
                output = self.predict(x)
                error += self.loss(output, y)
                
                # 反向传播
                grad = self.loss_derivative(output, y)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            
            error /= len(x_train)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, error: {error}')
```

## 5. 实战示例：手写数字识别

使用MNIST数据集实现手写数字识别：

```python
def load_mnist():
    """加载MNIST数据集"""
    from sklearn.datasets import load_digits
    digits = load_digits()
    return digits.images, digits.target

def preprocess_data(X, y):
    """数据预处理"""
    # 归一化
    X = X.reshape(X.shape[0], -1) / 16.0
    
    # One-hot编码
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1
    
    return X, y_one_hot

def train_digit_classifier():
    """训练手写数字分类器"""
    # 加载数据
    X, y = load_mnist()
    X, y = preprocess_data(X, y)
    
    # 创建模型
    model = NeuralNetwork()
    model.add(FCLayer(64, 128))
    model.add(Activation.relu)
    model.add(FCLayer(128, 10))
    model.add(Activation.sigmoid)
    
    # 设置损失函数
    model.set_loss(Loss.cross_entropy, Loss.cross_entropy_derivative)
    
    # 训练模型
    model.fit(X, y, epochs=1000, learning_rate=0.1)
    
    return model
```

## 练习与作业

1. 实现不同的优化器（SGD、Adam）
2. 添加正则化方法（L1、L2、Dropout）
3. 尝试不同的网络架构

## 扩展阅读

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n课程笔记](https://cs231n.github.io/)

## 小测验

1. 神经网络中的激活函数有什么作用？
2. 反向传播算法的原理是什么？
3. 如何选择合适的网络架构？

## 下一步学习

- [深度学习框架](frameworks.md)
- [计算机视觉基础](computer-vision.md)
- [自然语言处理入门](natural-language-processing.md)
