---
title: "高性能计算"
slug: "high-performance-computing"
sequence: 5
description: "AI开发中的高性能计算技术，包括NumPy优化、并行计算、GPU加速等关键技能"
is_published: true
estimated_minutes: 90
language: "zh-CN"
---

# 高性能计算

## 课程介绍
本模块聚焦AI开发中的高性能计算技术，教你如何优化代码性能，充分利用硬件资源。通过实际案例，掌握从CPU到GPU的各种加速技术。

## 学习目标
完成本模块学习后，你将能够：
- 使用NumPy进行高效计算
- 实现并行和分布式计算
- 使用GPU加速深度学习
- 优化大规模数据处理性能

## 1. NumPy优化技巧

### 1.1 向量化运算
```python
# ⚡️ 实战案例：图像处理优化
import numpy as np
import time

def process_image_loop(image):
    """使用循环处理图像（慢）"""
    height, width = image.shape
    result = np.zeros_like(image)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # 3x3卷积核
            result[i, j] = np.sum(image[i-1:i+2, j-1:j+2]) / 9
    
    return result

def process_image_vectorized(image):
    """使用向量化处理图像（快）"""
    # 使用卷积操作
    kernel = np.ones((3, 3)) / 9
    return np.correlate(image, kernel, mode='same')

# 性能对比
image = np.random.rand(1000, 1000)

start = time.time()
result_loop = process_image_loop(image)
print(f"Loop time: {time.time() - start:.2f}s")

start = time.time()
result_vectorized = process_image_vectorized(image)
print(f"Vectorized time: {time.time() - start:.2f}s")
```

### 1.2 内存优化
```python
# 💾 实战案例：内存优化
def optimize_array_memory():
    """内存使用优化技巧"""
    # 1. 使用适当的数据类型
    x = np.zeros(1000000, dtype=np.float32)  # 而不是float64
    
    # 2. 使用视图而不是复制
    y = x.view()  # 而不是x.copy()
    
    # 3. 使用内存映射
    data = np.memmap('large_array.npy', dtype=np.float32, mode='w+', shape=(1000000,))
    
    # 4. 使用生成器处理大数据
    def process_chunks(array, chunk_size=1000):
        for i in range(0, len(array), chunk_size):
            yield array[i:i + chunk_size]
```

## 2. 并行计算基础

### 2.1 多进程处理
```python
# 🔄 实战案例：并行数据处理
from multiprocessing import Pool
import numpy as np

def process_chunk(chunk):
    """处理数据块的函数"""
    # 示例：计算每个数字的平方根和平方的和
    return np.sum(np.sqrt(chunk) + np.square(chunk))

def parallel_processing(data, num_processes=4):
    """并行处理大数据集"""
    # 将数据分割成块
    chunks = np.array_split(data, num_processes)
    
    # 创建进程池
    with Pool(num_processes) as pool:
        # 并行处理每个块
        results = pool.map(process_chunk, chunks)
    
    # 合并结果
    return sum(results)

# 使用示例
data = np.random.rand(1000000)
result = parallel_processing(data)
```

### 2.2 异步处理
```python
# ⚡️ 实战案例：异步数据加载
import asyncio
import aiohttp
import aiofiles

async def download_file(url, filename):
    """异步下载文件"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                async with aiofiles.open(filename, mode='wb') as f:
                    await f.write(await response.read())
                return filename
            return None

async def process_file(filename):
    """异步处理文件"""
    async with aiofiles.open(filename, mode='r') as f:
        content = await f.read()
        # 处理文件内容
        return len(content)

async def main():
    """主异步函数"""
    # 下载任务
    download_tasks = [
        download_file(url, f"file_{i}.txt")
        for i, url in enumerate(urls)
    ]
    # 并发下载
    files = await asyncio.gather(*download_tasks)
    
    # 处理任务
    process_tasks = [
        process_file(f) for f in files if f is not None
    ]
    # 并发处理
    results = await asyncio.gather(*process_tasks)
    return results
```

## 3. GPU加速入门

### 3.1 CUDA基础
```python
# 🚀 实战案例：GPU加速计算
import torch

def cuda_acceleration_demo():
    """GPU加速示例"""
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建大矩阵
    size = 10000
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    # CPU计算
    start = time.time()
    c_cpu = torch.mm(a, b)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.2f}s")
    
    # GPU计算
    if device.type == 'cuda':
        a = a.to(device)
        b = b.to(device)
        
        start = time.time()
        c_gpu = torch.mm(a, b)
        gpu_time = time.time() - start
        print(f"GPU time: {gpu_time:.2f}s")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

### 3.2 CuPy使用
```python
# 📊 实战案例：CuPy数据处理
import cupy as cp

def cupy_processing_demo():
    """CuPy处理示例"""
    # 创建GPU数组
    x_gpu = cp.random.random((1000, 1000))
    y_gpu = cp.random.random((1000, 1000))
    
    # GPU上进行计算
    start = time.time()
    # 矩阵乘法
    z_gpu = cp.dot(x_gpu, y_gpu)
    # FFT变换
    fft_gpu = cp.fft.fft2(z_gpu)
    # 卷积操作
    kernel = cp.random.random((3, 3))
    conv_gpu = cp.correlate2d(z_gpu, kernel, mode='same')
    
    gpu_time = time.time() - start
    print(f"GPU processing time: {gpu_time:.2f}s")
    
    # 将结果转回CPU
    z_cpu = cp.asnumpy(z_gpu)
    return z_cpu
```

## 实战项目：图像处理加速器

### 项目描述
实现一个高性能的图像处理系统，结合CPU和GPU加速技术。

### 项目代码框架
```python
class ImageProcessor:
    def __init__(self, use_gpu=True):
        self.device = 'gpu' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
    
    def load_model(self):
        """加载预训练模型"""
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                             'resnet18', pretrained=True)
        if self.device == 'gpu':
            model = model.cuda()
        return model
    
    def process_batch(self, images):
        """批量处理图像"""
        # 转换为张量
        batch = torch.stack([self.preprocess(img) for img in images])
        if self.device == 'gpu':
            batch = batch.cuda()
        
        # 使用模型处理
        with torch.no_grad():
            output = self.model(batch)
        
        # 后处理
        results = self.postprocess(output)
        return results
    
    def parallel_process(self, image_list, batch_size=32, num_workers=4):
        """并行处理大量图像"""
        # 创建数据加载器
        dataset = ImageDataset(image_list)
        loader = DataLoader(dataset, batch_size=batch_size, 
                          num_workers=num_workers)
        
        results = []
        for batch in loader:
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
        
        return results
```

## 练习与作业
1. 实现矩阵运算的GPU加速
2. 创建并行数据处理pipeline
3. 优化大规模计算性能

## 扩展阅读
- [NumPy性能优化指南](https://numpy.org/doc/stable/user/basics.html)
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [并行计算实践](https://python-parallel-programming-cookbook.readthedocs.io/)

## 小测验
1. 向量化计算的优势是什么？
2. 如何选择合适的并行策略？
3. GPU加速的适用场景有哪些？

## 下一步学习
- 分布式计算
- 深度学习优化
- 模型部署加速

## 常见问题解答
Q: CPU和GPU如何选择？
A: 根据任务特点选择：CPU适合逻辑复杂的串行任务，GPU适合大规模并行的数值计算。

Q: 如何避免内存溢出？
A: 使用生成器、内存映射、分批处理等技术，避免一次加载过多数据。
