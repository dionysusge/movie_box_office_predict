# 🎬 电影票房预测项目

**作者**: Dionysus  
**日期**: 2025-8-12  
**版本**: 2.0

## 📋 项目简介

这是一个基于机器学习的电影票房预测项目，使用多种先进的算法和特征工程技术来预测电影的全球票房收入。项目采用了XGBoost、LightGBM、CatBoost等多种模型的集成学习方法，并通过优化的特征选择策略显著提升了预测精度。

### 🎯 项目目标

- 准确预测电影全球票房收入
- 分析影响票房的关键因素
- 为电影投资决策提供数据支持
- 探索先进的机器学习技术在票房预测中的应用

### 📊 核心指标

- **SMAPE**: 从38.27%优化至1.66%（提升95.7%）
- **R²**: 达到0.9968（解释99.68%的方差）
- **模型类型**: 集成学习（Stacking）
- **特征数量**: 智能选择5-20个最优特征

## 🚀 主要功能

### 1. 数据处理与特征工程
- 📈 **高级特征工程**: 交互特征、多项式特征、时间特征
- 🧹 **数据清洗**: 异常值检测、缺失值处理
- 🎭 **情感分析**: 基于评论的情感特征提取
- 📝 **主题建模**: 电影主题特征生成

### 2. 智能特征选择
- 🎯 **多策略选择**: 宽松阈值、百分位数、固定数量、重要性阈值
- ⚖️ **智能平衡**: 自动选择最优特征数量（5-20个）
- 🔍 **风险评估**: 过拟合风险检测和预警
- 📊 **稳定性分析**: 特征选择一致性验证

### 3. 先进建模技术
- 🏗️ **集成学习**: Stacking多模型融合
- 🌳 **基模型**: XGBoost、LightGBM、CatBoost、ExtraTrees
- 🎛️ **超参数优化**: 网格搜索和贝叶斯优化
- 📈 **目标变量变换**: 对数变换优化分布

### 4. 模型评估与可视化
- 📊 **多指标评估**: RMSE、MAE、R²、SMAPE
- 📈 **交叉验证**: 5折交叉验证确保稳定性
- 🎨 **可视化分析**: 特征重要性、预测vs实际、残差分析
- 📋 **性能对比**: 多模型性能详细对比

## 📁 项目结构
数据文件过大未上传
```
电影票房预测/
├── 📊 数据文件
│   ├── final_modeling_dataset_all.csv      # 完整建模数据集
│   ├── movies_with_sentiment_features.csv  # 包含情感特征的数据
│   ├── comments_with_sentiment.csv         # 情感分析结果
│   └── movie_topics.csv                    # 电影主题特征
│
├── 📓 核心代码
│   ├── main.ipynb                          # 主要分析笔记本
│   ├── improved_feature_selection.py       # 改进的特征选择模块
│   ├── feature_selection_example.py        # 特征选择使用示例
│   └── quick_fix_feature_selection.py      # 快速修复代码
│
├── 📈 可视化结果
│   ├── advanced_model_analysis.png         # 高级模型分析图
│   ├── model_comparison_analysis.png       # 模型对比分析图
│   └── sentiment_vs_gross.png              # 情感vs票房分析图
│
├── 📚 文档
│   ├── README.md                           # 项目说明文档
│   ├── feature_selection_analysis.md       # 特征选择分析文档
│   └── 理解.md                             # 项目理解文档
│
└── 🛠️ 工具资源
    ├── stopwords/                           # 停用词库
    ├── wordnet/                             # WordNet词典
    └── catboost_info/                       # CatBoost训练信息
```

## 🔧 环境配置


### 依赖包安装

```bash

# 安装核心依赖
pip install pandas numpy scikit-learn xgboost lightgbm catboost
pip install matplotlib seaborn plotly
pip install nltk textblob wordcloud
pip install jupyter notebook

# 安装可选依赖
pip install optuna bayesian-optimization  # 超参数优化
pip install shap lime                     # 模型解释
```


## 📊 核心算法

### 1. 特征工程

#### 🔢 数值特征处理
```python
# 多项式特征
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)

# 交互特征
X['budget_duration'] = X['budget'] * X['duration']
X['votes_rating'] = X['imdb_score'] * X['num_voted_users']
```

#### 📝 文本特征处理
```python
# 情感分析
from textblob import TextBlob
sentiment = TextBlob(text).sentiment.polarity

# 主题建模
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=5)
```

### 2. 集成学习

#### 🏗️ Stacking架构
```python
from sklearn.ensemble import StackingRegressor

# 基模型
base_models = [
    ('xgb', XGBRegressor()),
    ('lgb', LGBMRegressor()),
    ('cat', CatBoostRegressor()),
    ('et', ExtraTreesRegressor())
]

# 元模型
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=10),
    cv=5
)
```

### 3. 特征选择策略

#### 🎯 多策略融合
```python
# 策略1: 宽松阈值
selector_loose = SelectFromModel(model, threshold='0.5*mean')

# 策略2: 百分位数
selector_percentile = SelectPercentile(f_regression, percentile=70)

# 策略3: 固定数量
selector_kbest = SelectKBest(f_regression, k=12)

# 智能选择最佳策略
best_strategy = select_best_strategy(strategies, X_train, y_train)
```

## 📈 性能指标

### 🏆 最终结果

| 指标 | 优化前 | 优化后 | 改进幅度 |
|------|--------|--------|----------|
| **SMAPE** | 38.27% | 1.66% | ↓ 95.7% |
| **RMSE** | 45,000,000 | 6,818,332 | ↓ 84.8% |
| **MAE** | 25,000,000 | 1,683,885 | ↓ 93.3% |
| **R²** | 0.85 | 0.9968 | ↑ 17.3% |

### 📊 模型对比

| 模型 | SMAPE | R² | 特征数 | 训练时间 |
|------|-------|----|---------|---------|
| 单一XGBoost | 2.28% | 0.9825 | 2 | 30s |
| **集成学习** | **1.66%** | **0.9968** | **12** | **120s** |
| 改进特征选择 | 1.85% | 0.9945 | 8 | 45s |

### 🎯 关键改进点

1. **特征选择优化**: 从2个特征增加到8-15个，降低过拟合风险
2. **集成学习**: 多模型融合提升预测稳定性
3. **特征工程**: 高级特征工程挖掘更多信息
4. **目标变换**: 对数变换改善分布特性

### 技术文档
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

### 相关项目
- [Kaggle电影数据集](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [电影票房预测竞赛](https://www.kaggle.com/c/tmdb-box-office-prediction)

