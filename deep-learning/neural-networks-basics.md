---
title: "神经网络基础"
slug: "neural-networks"
sequence: 9
description: "深度学习的核心概念和基础组件，包括神经网络结构、激活函数、反向传播等基础知识"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

# 神经网络基础

## 课程介绍
本模块深入讲解神经网络的基础概念和核心组件，通过实现一个简单的神经网络来理解深度学习的基本原理。

## 学习目标
完成本模块学习后，你将能够：
- 理解神经网络的基本结构
- 掌握前向传播和反向传播
- 实现常用的激活函数
- 构建简单的神经网络

## 1. 神经网络结构

### 1.1 神经元模型
```python
# 🧠 实战案例：神经元实现
import numpy as np

class Neuron:
    """单个神经元的实现"""
    def __init__(self, n_inputs):
        # 随机初始化权重
        self.weights = np.random.randn(n_inputs) * 0.01
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

# 测试神经元
def test_neuron():
    """测试神经元的前向和反向传播"""
    # 创建神经元
    neuron = Neuron(n_inputs=3)
    
    # 测试数据
    x = np.array([1.0, 2.0, 3.0])
    
    # 前向传播
    output = neuron.forward(x)
    print(f"输出: {output}")
    
    # 反向传播
    grad_output = 1.0
    grad_x, grad_w, grad_b = neuron.backward(grad_output)
    print(f"输入梯度: {grad_x}")
    print(f"权重梯度: {grad_w}")
    print(f"偏置梯度: {grad_b}")
```

### 1.2 层的实现
```python
# 🔄 实战案例：全连接层实现
class FCLayer:
    """全连接层实现"""
    def __init__(self, n_inputs, n_units):
        self.weights = np.random.randn(n_inputs, n_units) * 0.01
        self.bias = np.zeros(n_units)
        
        self.x = None
        self.output = None
        
        # 优化器状态
        self.momentum_w = np.zeros_like(self.weights)
        self.momentum_b = np.zeros_like(self.bias)
    
    def forward(self, x):
        """前向传播"""
        self.x = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def backward(self, grad_output):
        """反向传播"""
        # 计算梯度
        grad_weights = np.dot(self.x.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        grad_x = np.dot(grad_output, self.weights.T)
        
        return grad_x, grad_weights, grad_bias
    
    def update(self, grad_weights, grad_bias, learning_rate=0.01, momentum=0.9):
        """使用动量更新参数"""
        # 更新动量
        self.momentum_w = momentum * self.momentum_w - learning_rate * grad_weights
        self.momentum_b = momentum * self.momentum_b - learning_rate * grad_bias
        
        # 更新参数
        self.weights += self.momentum_w
        self.bias += self.momentum_b
```

## 2. 激活函数

### 2.1 常用激活函数
```python
# ⚡️ 实战案例：激活函数实现
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

# 可视化激活函数
def plot_activations():
    """可视化不同的激活函数"""
    x = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(15, 5))
    
    # ReLU
    plt.subplot(1, 3, 1)
    plt.plot(x, Activation.relu(x), label='ReLU')
    plt.plot(x, Activation.relu_derivative(x), label='导数')
    plt.title('ReLU')
    plt.legend()
    
    # Sigmoid
    plt.subplot(1, 3, 2)
    plt.plot(x, Activation.sigmoid(x), label='Sigmoid')
    plt.plot(x, Activation.sigmoid_derivative(x), label='导数')
    plt.title('Sigmoid')
    plt.legend()
    
    # Tanh
    plt.subplot(1, 3, 3)
    plt.plot(x, Activation.tanh(x), label='Tanh')
    plt.plot(x, Activation.tanh_derivative(x), label='导数')
    plt.title('Tanh')
    plt.legend()
    
    return plt.gcf()
```

## 3. 前向传播与反向传播

### 3.1 简单神经网络实现
```python
# 🧮 实战案例：两层神经网络
class SimpleNN:
    """简单的两层神经网络"""
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden = FCLayer(input_size, hidden_size)
        self.output = FCLayer(hidden_size, output_size)
        self.activation = Activation()
    
    def forward(self, x):
        """前向传播"""
        # 第一层
        hidden_output = self.hidden.forward(x)
        hidden_activated = self.activation.relu(hidden_output)
        
        # 第二层
        output = self.output.forward(hidden_activated)
        return self.activation.sigmoid(output)
    
    def backward(self, x, y, output):
        """反向传播"""
        batch_size = x.shape[0]
        
        # 输出层梯度
        grad_output = (output - y) / batch_size
        grad_output *= self.activation.sigmoid_derivative(self.output.output)
        grad_h, grad_w2, grad_b2 = self.output.backward(grad_output)
        
        # 隐藏层梯度
        grad_h *= self.activation.relu_derivative(self.hidden.output)
        grad_x, grad_w1, grad_b1 = self.hidden.backward(grad_h)
        
        return (grad_w1, grad_b1), (grad_w2, grad_b2)
    
    def train_step(self, x, y, learning_rate=0.01):
        """训练一步"""
        # 前向传播
        output = self.forward(x)
        
        # 计算损失
        loss = -np.mean(y * np.log(output + 1e-8) + 
                       (1 - y) * np.log(1 - output + 1e-8))
        
        # 反向传播
        (grad_w1, grad_b1), (grad_w2, grad_b2) = self.backward(x, y, output)
        
        # 更新参数
        self.hidden.update(grad_w1, grad_b1, learning_rate)
        self.output.update(grad_w2, grad_b2, learning_rate)
        
        return loss
```

### 3.2 训练过程
```python
# 📈 实战案例：训练神经网络
def train_network():
    """训练神经网络示例"""
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    # 创建网络
    network = SimpleNN(input_size=2, hidden_size=4, output_size=1)
    
    # 训练参数
    epochs = 100
    batch_size = 32
    learning_rate = 0.1
    losses = []
    
    # 训练循环
    for epoch in range(epochs):
        epoch_losses = []
        
        # 批量训练
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            loss = network.train_step(batch_X, batch_y, learning_rate)
            epoch_losses.append(loss)
        
        # 记录平均损失
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return network, losses

# 可视化训练过程
def plot_training(losses):
    """可视化训练损失"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    return plt.gcf()
```

## 实战项目：手写数字识别

### 项目描述
使用简单的神经网络实现MNIST手写数字识别。

### 项目代码框架
```python
class DigitClassifier:
    def __init__(self):
        # 784 -> 128 -> 10
        self.network = SimpleNN(784, 128, 10)
        self.history = []
    
    def preprocess_data(self, X):
        """预处理数据"""
        # 归一化
        X = X.astype(float) / 255.0
        # 展平图像
        X = X.reshape(X.shape[0], -1)
        return X
    
    def one_hot_encode(self, y):
        """独热编码"""
        n_classes = 10
        n_samples = len(y)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def train(self, X_train, y_train, X_val, y_val, 
             epochs=10, batch_size=32, learning_rate=0.01):
        """训练模型"""
        # 预处理数据
        X_train = self.preprocess_data(X_train)
        X_val = self.preprocess_data(X_val)
        
        # 独热编码标签
        y_train = self.one_hot_encode(y_train)
        y_val = self.one_hot_encode(y_val)
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_losses = []
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                loss = self.network.train_step(
                    batch_X, batch_y, learning_rate
                )
                train_losses.append(loss)
            
            # 验证
            val_output = self.network.forward(X_val)
            val_loss = -np.mean(y_val * np.log(val_output + 1e-8) + 
                              (1 - y_val) * np.log(1 - val_output + 1e-8))
            
            # 计算准确率
            val_pred = np.argmax(val_output, axis=1)
            val_true = np.argmax(y_val, axis=1)
            accuracy = np.mean(val_pred == val_true)
            
            # 记录历史
            self.history.append({
                'train_loss': np.mean(train_losses),
                'val_loss': val_loss,
                'val_accuracy': accuracy
            })
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"train_loss: {np.mean(train_losses):.4f}")
            print(f"val_loss: {val_loss:.4f}")
            print(f"val_accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        """预测"""
        X = self.preprocess_data(X)
        output = self.network.forward(X)
        return np.argmax(output, axis=1)
    
    def visualize_predictions(self, X, y_true, n_samples=5):
        """可视化预测结果"""
        # 随机选择样本
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
        y_true = y_true[indices]
        
        # 预测
        y_pred = self.predict(X_sample)
        
        # 可视化
        plt.figure(figsize=(15, 3))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i+1)
            plt.imshow(X_sample[i], cmap='gray')
            plt.title(f'True: {y_true[i]}\nPred: {y_pred[i]}')
            plt.axis('off')
        
        return plt.gcf()
```

## 练习与作业
1. 实现不同的优化器（SGD、Adam）
2. 添加批量归一化层
3. 尝试不同的网络架构

## 扩展阅读
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n课程笔记](https://cs231n.github.io/)

## 小测验
1. 不同激活函数的优缺点是什么？
2. 反向传播算法的原理是什么？
3. 如何处理梯度消失问题？

## 下一步学习
- 卷积神经网络
- 循环神经网络
- 深度学习框架使用

## 常见问题解答
Q: 如何选择合适的网络架构？
A: 根据问题类型、数据规模和计算资源选择。图像任务通常用CNN，序列任务用RNN，简单任务可以用多层感知机。

Q: 如何调整学习率？
A: 可以从较小的值开始，观察训练曲线。如果收敛太慢可以增大，如果不稳定则减小。也可以使用学习率调度器。
