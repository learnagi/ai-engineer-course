---
title: "实战案例：客户流失预测"
slug: "customer-churn-prediction"
description: "通过电信客户数据集，实践分类问题的完整机器学习流程，包括数据处理、特征工程、模型训练和评估"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

# 实战案例：客户流失预测

## 项目概述
本案例使用电信客户数据集，预测客户是否会流失。我们将完整演示二分类问题的解决流程，包括：
- 不平衡数据处理
- 分类特征编码
- 模型评估指标选择
- 模型解释和业务洞察

## 1. 环境准备
```python
# 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap

# 设置随机种子
np.random.seed(42)
```

## 2. 数据加载和探索

### 2.1 加载数据
```python
def load_data(file_path):
    """加载数据集"""
    df = pd.read_csv(file_path)
    print("数据集形状:", df.shape)
    print("\n数据集信息:")
    print(df.info())
    return df

# 加载数据
df = load_data('telco_customer_churn.csv')
```

### 2.2 探索性数据分析
```python
def explore_data(df):
    """探索性数据分析"""
    # 目标变量分布
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    plt.title("客户流失分布")
    plt.show()
    
    # 数值特征分析
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(numeric_features):
        plt.subplot(1, len(numeric_features), i+1)
        sns.boxplot(x='Churn', y=feature, data=df)
        plt.title(feature)
    plt.tight_layout()
    plt.show()
    
    # 类别特征分析
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        df_grouped = df.groupby(feature)['Churn'].mean().sort_values(ascending=False)
        df_grouped.plot(kind='bar')
        plt.title(f'{feature} vs Churn Rate')
        plt.xticks(rotation=45)
        plt.show()

explore_data(df)
```

## 3. 数据预处理

### 3.1 处理缺失值
```python
def handle_missing_values(df):
    """处理缺失值"""
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("缺失值统计:")
    print(missing_values[missing_values > 0])
    
    # 数值型特征用中位数填充
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    for feature in numeric_features:
        df[feature].fillna(df[feature].median(), inplace=True)
    
    # 类别型特征用众数填充
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        df[feature].fillna(df[feature].mode()[0], inplace=True)
    
    return df

df_cleaned = handle_missing_values(df)
```

### 3.2 特征工程
```python
def feature_engineering(df):
    """特征工程"""
    # 创建新特征
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['AvgMonthlyCharges'] = df['TotalCharges'] / df['tenure']
    df['ChargesPerService'] = df['MonthlyCharges'] / df['tenure']
    
    # 合约相关特征
    df['IsLongTermContract'] = df['Contract'].map(
        {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    )
    
    # 服务数量
    services = ['PhoneService', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[services].apply(
        lambda x: sum([1 for item in x if item not in ['No', 'No internet service']])
    )
    
    return df

df_engineered = feature_engineering(df_cleaned)
```

### 3.3 特征编码
```python
def encode_features(df):
    """特征编码"""
    # 标签编码
    label_encoders = {}
    categorical_features = df.select_dtypes(include=['object']).columns
    
    for feature in categorical_features:
        if feature != 'customerID':
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            label_encoders[feature] = le
    
    return df, label_encoders

df_encoded, label_encoders = encode_features(df_engineered)
```

## 4. 处理不平衡数据

### 4.1 SMOTE过采样
```python
def balance_data(X, y):
    """使用SMOTE处理不平衡数据"""
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print("原始数据集形状:", X.shape)
    print("平衡后数据集形状:", X_balanced.shape)
    
    return X_balanced, y_balanced

# 准备数据
X = df_encoded.drop(['Churn', 'customerID'], axis=1)
y = df_encoded['Churn']

# 平衡数据
X_balanced, y_balanced = balance_data(X, y)
```

## 5. 模型训练和评估

### 5.1 模型训练
```python
def train_models(X, y):
    """训练多个模型"""
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义模型
    models = {
        'XGBoost': xgb.XGBClassifier(
            scale_pos_weight=len(y[y==0])/len(y[y==1]),
            random_state=42
        ),
        'LightGBM': lgb.LGBMClassifier(
            scale_pos_weight=len(y[y==0])/len(y[y==1]),
            random_state=42
        )
    }
    
    # 训练和评估
    results = {}
    for name, model in models.items():
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # 评估指标
        results[name] = {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    return results, X_train_scaled, X_test_scaled

# 训练模型
results, X_train_scaled, X_test_scaled = train_models(X_balanced, y_balanced)
```

### 5.2 模型评估
```python
def evaluate_model(name, result):
    """评估模型性能"""
    y_test = result['y_test']
    y_pred = result['y_pred']
    y_prob = result['y_prob']
    
    # 打印分类报告
    print(f"\n{name} 分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title(f'{name} 混淆矩阵')
    plt.show()
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend()
    plt.show()
    
    # PR曲线
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{name} Precision-Recall Curve')
    plt.show()

# 评估每个模型
for name, result in results.items():
    evaluate_model(name, result)
```

## 6. 模型解释

### 6.1 特征重要性
```python
def plot_feature_importance(model, feature_names):
    """绘制特征重要性"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title("特征重要性")
    plt.show()
    
    return importance

# 分析XGBoost模型的特征重要性
importance = plot_feature_importance(
    results['XGBoost']['model'],
    X.columns
)
```

### 6.2 SHAP值分析
```python
def analyze_shap_values(model, X):
    """SHAP值分析"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 特征重要性
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X)
    
    # 依赖图
    for feature in X.columns[:3]:  # 展示前三个重要特征
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X)

# SHAP值分析
analyze_shap_values(
    results['XGBoost']['model'],
    pd.DataFrame(X_test_scaled, columns=X.columns)
)
```

## 7. 业务洞察

### 7.1 客户细分
```python
def customer_segmentation(df, predictions, probabilities):
    """客户细分分析"""
    df['Predicted_Churn'] = predictions
    df['Churn_Probability'] = probabilities
    
    # 高风险客户
    high_risk = df[df['Churn_Probability'] > 0.7]
    print("\n高风险客户特征:")
    print(high_risk.describe())
    
    # 客户特征分析
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Predicted_Churn', y='MonthlyCharges', data=df)
    plt.title('预测流失与月费关系')
    plt.show()
    
    return high_risk

# 分析高风险客户
high_risk_customers = customer_segmentation(
    df_encoded.copy(),
    results['XGBoost']['y_pred'],
    results['XGBoost']['y_prob']
)
```

### 7.2 干预建议
```python
def generate_recommendations(high_risk_df):
    """生成干预建议"""
    recommendations = []
    
    # 基于合约类型
    if high_risk_df['Contract'].mean() < 1:  # 大多是月付用户
        recommendations.append("提供长期合约优惠")
    
    # 基于服务数量
    if high_risk_df['TotalServices'].mean() < 3:
        recommendations.append("推荐捆绑服务套餐")
    
    # 基于月费
    if high_risk_df['MonthlyCharges'].mean() > df_encoded['MonthlyCharges'].mean():
        recommendations.append("考虑提供价格优惠")
    
    return recommendations

# 生成建议
recommendations = generate_recommendations(high_risk_customers)
print("\n干预建议:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")
```

## 8. 模型部署

### 8.1 保存模型
```python
import joblib

def save_model(model, scaler, encoders, file_prefix='churn_prediction'):
    """保存模型和预处理器"""
    joblib.dump(model, f'{file_prefix}_model.pkl')
    joblib.dump(scaler, f'{file_prefix}_scaler.pkl')
    joblib.dump(encoders, f'{file_prefix}_encoders.pkl')

# 保存最佳模型
save_model(
    results['XGBoost']['model'],
    scaler,
    label_encoders
)
```

### 8.2 预测函数
```python
def predict_churn(customer_data, model_path='churn_prediction_model.pkl',
                 scaler_path='churn_prediction_scaler.pkl',
                 encoders_path='churn_prediction_encoders.pkl'):
    """预测客户流失风险"""
    # 加载模型和预处理器
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    
    # 预处理数据
    processed_data = customer_data.copy()
    
    # 编码类别特征
    for feature, encoder in encoders.items():
        if feature in processed_data.columns:
            processed_data[feature] = encoder.transform(processed_data[feature])
    
    # 标准化
    processed_data = scaler.transform(processed_data)
    
    # 预测
    churn_prob = model.predict_proba(processed_data)[:, 1]
    
    return churn_prob

# 示例预测
sample_customer = X.iloc[[0]]
churn_risk = predict_churn(sample_customer)
print(f"\n客户流失风险: {churn_risk[0]:.2%}")
```

## 常见问题解答

Q: 如何处理新客户数据？
A: 新客户数据需要经过相同的预处理流程：
1. 特征工程
2. 类别编码
3. 标准化

Q: 如何选择阈值？
A: 阈值选择需要考虑：
1. 业务成本（误判成本）
2. 干预资源
3. 精确率和召回率的平衡
4. ROC曲线和PR曲线

Q: 模型更新策略？
A: 建议：
1. 定期重新训练（如每月）
2. 监控模型性能
3. 收集新数据
4. 根据业务变化调整特征

## 扩展阅读
1. [Customer Churn Prediction Guide](https://www.datascience.com/blog/churn-prediction-python)
2. [Handling Imbalanced Datasets](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
3. [Model Interpretation Techniques](https://christophm.github.io/interpretable-ml-book/)
