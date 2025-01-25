---
title: "模型部署与服务化"
slug: "model-deployment"
description: "学习机器学习模型的部署策略和服务化最佳实践"
is_published: true
estimated_minutes: 90
language: "zh-CN"
---

![模型部署](assets/images/engineering/model-deployment-header.png)
*模型部署是将机器学习模型投入生产环境的关键步骤*

# 模型部署与服务化

## 学习目标
完成本节后，你将能够：
- 理解模型部署的关键概念
- 掌握不同的部署策略
- 实现模型服务化
- 优化服务性能
- 监控模型运行状态

## 先修知识
学习本节内容需要：
- Python编程基础
- 机器学习基础
- Web服务开发基础
- Docker容器技术基础

## 模型序列化

### 模型保存与加载
```python
import joblib
import pickle
from typing import Any, Dict
import os

class ModelSerializer:
    """模型序列化工具"""
    
    @staticmethod
    def save_model(model: Any, path: str,
                  format: str = 'joblib') -> None:
        """
        保存模型到文件
        
        参数:
            model: 训练好的模型
            path: 保存路径
            format: 序列化格式 ('joblib' 或 'pickle')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if format == 'joblib':
            joblib.dump(model, path)
        elif format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_model(path: str, format: str = 'joblib') -> Any:
        """
        从文件加载模型
        """
        if format == 'joblib':
            return joblib.load(path)
        elif format == 'pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

# 使用示例
model_serializer = ModelSerializer()
model_serializer.save_model(
    model,
    'models/model.joblib',
    format='joblib'
)
```

### ONNX转换
```python
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

class ONNXConverter:
    """ONNX模型转换器"""
    
    @staticmethod
    def convert_to_onnx(model: Any, input_shape: tuple,
                       model_name: str) -> bytes:
        """
        将sklearn模型转换为ONNX格式
        """
        initial_type = [('float_input',
                        FloatTensorType(input_shape))]
        
        # 转换模型
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12,
            options={id(model): {'score_names': ['output']}}
        )
        
        return onnx_model.SerializeToString()
    
    @staticmethod
    def save_onnx(model_bytes: bytes, path: str) -> None:
        """
        保存ONNX模型
        """
        with open(path, 'wb') as f:
            f.write(model_bytes)
    
    @staticmethod
    def create_inference_session(model_path: str):
        """
        创建ONNX推理会话
        """
        return ort.InferenceSession(model_path)
```

## REST API服务

### Flask服务器
```python
from flask import Flask, request, jsonify
import numpy as np
from typing import Dict, Any

class ModelServer:
    """模型服务器"""
    
    def __init__(self, model_path: str):
        self.app = Flask(__name__)
        self.model = ModelSerializer.load_model(model_path)
        
        # 注册路由
        self.app.route('/predict',
                      methods=['POST'])(self.predict)
        self.app.route('/health',
                      methods=['GET'])(self.health_check)
    
    def predict(self):
        """预测接口"""
        try:
            # 获取输入数据
            data = request.get_json()
            
            # 验证输入
            if not self._validate_input(data):
                return jsonify({
                    'error': 'Invalid input format'
                }), 400
            
            # 预处理
            processed_data = self._preprocess_input(data)
            
            # 模型预测
            predictions = self.model.predict(processed_data)
            
            # 后处理
            result = self._postprocess_output(predictions)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({
                'error': str(e)
            }), 500
    
    def health_check(self):
        """健康检查接口"""
        return jsonify({'status': 'healthy'})
    
    def run(self, host: str = 'localhost',
            port: int = 5000):
        """启动服务"""
        self.app.run(host=host, port=port)
```

### FastAPI服务器
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class PredictionInput(BaseModel):
    """预测输入模型"""
    features: List[float]

class PredictionOutput(BaseModel):
    """预测输出模型"""
    prediction: float
    probability: float

class FastAPIModelServer:
    """FastAPI模型服务器"""
    
    def __init__(self, model_path: str):
        self.app = FastAPI()
        self.model = ModelSerializer.load_model(model_path)
        
        # 注册路由
        self.app.post('/predict',
                     response_model=PredictionOutput)(self.predict)
        self.app.get('/health')(self.health_check)
    
    async def predict(self, input_data: PredictionInput):
        """异步预测接口"""
        try:
            # 转换输入
            features = np.array(input_data.features).reshape(1, -1)
            
            # 预测
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0].max()
            
            return PredictionOutput(
                prediction=float(prediction),
                probability=float(probability)
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def health_check(self):
        """健康检查"""
        return {'status': 'healthy'}
    
    def run(self, host: str = 'localhost',
            port: int = 8000):
        """启动服务"""
        uvicorn.run(self.app, host=host, port=port)
```

## 容器化部署

### Dockerfile
```dockerfile
# 使用Python基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "server.py"]
```

### Docker Compose
```yaml
version: '3'

services:
  model-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.joblib
      - LOG_LEVEL=INFO
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 批处理服务

### 批量预测
```python
class BatchPredictor:
    """批量预测器"""
    
    def __init__(self, model_path: str):
        self.model = ModelSerializer.load_model(model_path)
    
    def predict_batch(self, data: pd.DataFrame,
                     batch_size: int = 1000) -> List[Any]:
        """
        批量预测
        
        参数:
            data: 输入数据
            batch_size: 批次大小
        """
        predictions = []
        
        # 分批处理
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_predictions = self.model.predict(batch)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def predict_file(self, input_file: str,
                    output_file: str,
                    batch_size: int = 1000) -> None:
        """
        处理文件数据
        """
        # 读取数据
        data = pd.read_csv(input_file)
        
        # 批量预测
        predictions = self.predict_batch(data, batch_size)
        
        # 保存结果
        data['prediction'] = predictions
        data.to_csv(output_file, index=False)
```

## 模型监控

### 性能监控
```python
import time
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PredictionMetrics:
    """预测指标"""
    latency: float
    input_shape: tuple
    output_shape: tuple
    timestamp: float

class ModelMonitor:
    """模型监控器"""
    
    def __init__(self):
        self.metrics: List[PredictionMetrics] = []
    
    def record_prediction(self, metrics: PredictionMetrics):
        """记录预测指标"""
        self.metrics.append(metrics)
    
    def get_average_latency(self,
                          window_size: int = 100) -> float:
        """获取平均延迟"""
        if not self.metrics:
            return 0.0
        
        recent_metrics = self.metrics[-window_size:]
        return np.mean([m.latency for m in recent_metrics])
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics:
            return {}
        
        return {
            'total_predictions': len(self.metrics),
            'avg_latency': self.get_average_latency(),
            'max_latency': max(m.latency for m in self.metrics),
            'min_latency': min(m.latency for m in self.metrics)
        }
```

### 数据漂移检测
```python
from scipy import stats

class DriftDetector:
    """数据漂移检测器"""
    
    def __init__(self, reference_data: np.ndarray):
        self.reference_data = reference_data
        self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data: np.ndarray) -> Dict[str, float]:
        """计算统计量"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'kurtosis': stats.kurtosis(data, axis=0),
            'skewness': stats.skew(data, axis=0)
        }
    
    def detect_drift(self, new_data: np.ndarray,
                    threshold: float = 0.05) -> bool:
        """
        检测数据漂移
        
        参数:
            new_data: 新数据
            threshold: p值阈值
        """
        # 执行KS检验
        drift_detected = False
        for i in range(new_data.shape[1]):
            statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                new_data[:, i]
            )
            
            if p_value < threshold:
                drift_detected = True
                break
        
        return drift_detected
```

## 实战项目：在线预测服务

### 项目结构
```
prediction-service/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── server.py
│   ├── model.py
│   ├── monitoring.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_server.py
│   └── test_model.py
└── models/
    └── model.joblib
```

### 服务实现
```python
class PredictionService:
    """预测服务实现"""
    
    def __init__(self, model_path: str):
        self.model = ModelSerializer.load_model(model_path)
        self.monitor = ModelMonitor()
        self.drift_detector = DriftDetector(
            self._load_reference_data()
        )
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        模型预测
        """
        start_time = time.time()
        
        try:
            # 预测
            prediction = self.model.predict(features)
            
            # 记录指标
            latency = time.time() - start_time
            self.monitor.record_prediction(
                PredictionMetrics(
                    latency=latency,
                    input_shape=features.shape,
                    output_shape=prediction.shape,
                    timestamp=time.time()
                )
            )
            
            # 检测漂移
            if self.drift_detector.detect_drift(features):
                # 触发告警
                self._alert_drift()
            
            return {
                'prediction': prediction.tolist(),
                'latency': latency
            }
        
        except Exception as e:
            # 记录错误
            self._log_error(e)
            raise
```

## 练习与作业
1. 基础练习：
   - 实现模型序列化
   - 创建REST API服务
   - 部署Docker容器

2. 进阶练习：
   - 实现批量预测
   - 添加性能监控
   - 实现漂移检测

3. 项目实践：
   - 构建完整的预测服务
   - 实现高可用部署
   - 添加监控和告警

## 常见问题
Q1: 如何选择部署方式？
A1: 需要考虑以下因素：
- 请求量和延迟要求
- 资源限制
- 运维能力
- 成本预算

Q2: 如何保证服务稳定性？
A2: 可以采用以下措施：
- 负载均衡
- 熔断降级
- 监控告警
- 自动扩缩容

## 扩展阅读
- [MLOps最佳实践](https://ml-ops.org/)
- [模型部署模式](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)
- [FastAPI文档](https://fastapi.tiangolo.com/)

## 下一步学习
- Kubernetes部署
- 模型压缩
- 边缘部署
- 服务网格
