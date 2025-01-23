---
title: "机器学习基础概念"
slug: "ml-basics"
sequence: 7
description: "机器学习的核心概念和基础理论，包括学习范式、模型评估、特征工程等基础知识"
is_published: true
estimated_minutes: 90
language: "zh-CN"
---

# 机器学习基础概念

## 课程介绍
本模块介绍机器学习的核心概念和基础理论，帮助你建立完整的机器学习知识体系。通过实例讲解和代码实践，深入理解机器学习的本质。

## 学习目标
完成本模块学习后，你将能够：
- 理解机器学习的基本概念和类型
- 掌握模型评估的方法和指标
- 熟练进行特征工程
- 应对过拟合和欠拟合问题

## 1. 机器学习基础

### 1.1 学习范式
```python
# 🎯 实战案例：不同类型的学习任务
import numpy as np
from sklearn.model_selection import train_test_split

# 监督学习示例
def supervised_learning_demo():
    """分类任务示例"""
    # 生成数据
    X = np.random.randn(1000, 20)  # 特征
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 标签
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# 无监督学习示例
def unsupervised_learning_demo():
    """聚类任务示例"""
    from sklearn.cluster import KMeans
    
    # 生成数据
    X = np.random.randn(1000, 2)
    
    # 聚类
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(X)
    return X, clusters

# 强化学习示例
class SimpleEnvironment:
    """简单的强化学习环境"""
    def __init__(self):
        self.state = 0
        
    def step(self, action):
        """执行动作并返回奖励"""
        if action == 1:  # 向右移动
            self.state += 1
        else:  # 向左移动
            self.state -= 1
        
        # 计算奖励
        reward = -abs(self.state)  # 距离原点越远奖励越小
        done = abs(self.state) > 5  # 到达边界则结束
        
        return self.state, reward, done
```

### 1.2 模型评估
```python
# 📊 实战案例：模型评估方法
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y):
    """综合评估模型性能"""
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"交叉验证分数: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # 学习曲线
    def plot_learning_curve(model, X, y):
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, valid_scores = learning_curve(
            model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, n_jobs=-1
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), label='训练集')
        plt.plot(train_sizes, valid_scores.mean(axis=1), label='验证集')
        plt.xlabel('训练样本数')
        plt.ylabel('得分')
        plt.title('学习曲线')
        plt.legend()
        return plt.gcf()
    
    # 混淆矩阵
    def plot_confusion_matrix(y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        return plt.gcf()
    
    return {
        'cv_scores': cv_scores,
        'learning_curve': plot_learning_curve(model, X, y),
        'confusion_matrix': plot_confusion_matrix(y, model.predict(X))
    }
```

## 2. 特征工程

### 2.1 特征预处理
```python
# 🔧 实战案例：特征预处理Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def create_preprocessing_pipeline():
    """创建特征预处理Pipeline"""
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    # 组合Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    return preprocessor

def analyze_features(X):
    """分析特征质量"""
    # 缺失值分析
    missing = pd.DataFrame({
        'missing_count': X.isnull().sum(),
        'missing_ratio': X.isnull().sum() / len(X)
    })
    
    # 特征分布分析
    distributions = {}
    for col in X.select_dtypes(include=[np.number]).columns:
        distributions[col] = {
            'mean': X[col].mean(),
            'std': X[col].std(),
            'skew': X[col].skew()
        }
    
    # 特征相关性分析
    correlation = X.corr()
    
    return {
        'missing_analysis': missing,
        'distributions': distributions,
        'correlation': correlation
    }
```

### 2.2 特征选择
```python
# 🎯 实战案例：特征选择方法
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

def select_features(X, y):
    """综合特征选择"""
    # 1. 方差分析
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    
    # 2. 单变量特征选择
    selector = SelectKBest(score_func=f_classif, k=10)
    X_uni = selector.fit_transform(X, y)
    
    # 3. 基于模型的特征选择
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'variance_selection': X_var,
        'univariate_selection': X_uni,
        'feature_importance': importances
    }
```

## 3. 模型优化

### 3.1 过拟合与欠拟合
```python
# 🎯 实战案例：模型复杂度分析
def analyze_model_complexity():
    """分析模型复杂度对性能的影响"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    # 生成数据
    X = np.random.randn(1000, 1)
    y = (X[:, 0]**2 + np.random.randn(1000) * 0.1 > 0).astype(int)
    
    # 测试不同复杂度
    degrees = [1, 2, 3, 4, 5]
    train_scores = []
    test_scores = []
    
    for degree in degrees:
        # 创建多项式特征
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # 训练模型
        model = LogisticRegression()
        model.fit(X_poly, y)
        
        # 记录得分
        train_scores.append(model.score(X_poly, y))
        test_scores.append(np.mean(cross_val_score(model, X_poly, y, cv=5)))
    
    return degrees, train_scores, test_scores
```

### 3.2 正则化方法
```python
# 🛠️ 实战案例：正则化效果对比
def compare_regularization():
    """比较不同正则化方法的效果"""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    
    # 生成数据
    X = np.random.randn(100, 20)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1
    
    # 不同模型
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }
    
    # 比较结果
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        results[name] = {
            'coefficients': model.coef_,
            'score': model.score(X, y)
        }
    
    return results
```

## 实战项目：房价预测系统

### 项目描述
构建一个完整的房价预测系统，综合运用特征工程、模型选择和评估方法。

### 项目代码框架
```python
class HousePricePredictor:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.feature_importance = None
    
    def preprocess_data(self, X):
        """数据预处理"""
        # 创建预处理Pipeline
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(exclude=[np.number]).columns
        
        self.preprocessor = create_preprocessing_pipeline(
            numeric_features, categorical_features
        )
        
        return self.preprocessor.fit_transform(X)
    
    def select_features(self, X, y):
        """特征选择"""
        feature_selector = SelectKBest(f_regression, k=20)
        X_selected = feature_selector.fit_transform(X, y)
        
        # 记录特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_selector.scores_
        }).sort_values('importance', ascending=False)
        
        return X_selected
    
    def train_model(self, X, y):
        """训练模型"""
        # 创建模型Pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100))
        ])
        
        # 训练模型
        self.model.fit(X, y)
        
        # 评估性能
        scores = cross_val_score(self.model, X, y, cv=5)
        return np.mean(scores), np.std(scores)
    
    def predict(self, X):
        """预测房价"""
        return self.model.predict(X)
    
    def explain_prediction(self, X):
        """解释预测结果"""
        import shap
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        return shap_values
```

## 练习与作业
1. 实现完整的特征工程Pipeline
2. 比较不同正则化方法的效果
3. 构建模型评估报告系统

## 扩展阅读
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [scikit-learn文档](https://scikit-learn.org/stable/user_guide.html)

## 小测验
1. 机器学习的三种主要类型是什么？
2. 如何处理过拟合问题？
3. 特征选择的主要方法有哪些？

## 下一步学习
- 深度学习基础
- 模型部署
- AutoML技术

## 常见问题解答
Q: 如何选择合适的机器学习算法？
A: 根据数据类型、问题性质、样本量和计算资源等因素综合考虑。分类问题可以从简单的逻辑回归开始，回归问题可以从线性回归开始。

Q: 特征工程为什么重要？
A: 特征工程直接影响模型性能，好的特征可以简化模型结构，提高模型可解释性，改善泛化能力。
