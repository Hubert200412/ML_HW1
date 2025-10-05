# 乳腺癌检测：SVM与KNN算法应用

## 项目概述

本项目基于Scikit-learn的乳腺癌数据集，实现并比较支持向量机（SVM）和K近邻（KNN）算法在二分类任务中的性能。项目通过特征缩放、基于随机森林的重要性筛选进行预处理，并采用网格搜索优化超参数。实验结果显示两种模型均取得超过95%的准确率，显著优于基线模型DummyClassifier。

**关键词**：乳腺癌检测, 支持向量机（SVM）, K近邻（KNN）, 特征选择, 消融实验

## 作者信息

- **姓名**：段昊彤
- **学号**：2300016612
- **院系**：信息管理系
- **完成日期**：2025年10月5日

## 应用场景与问题定义

### 应用场景
乳腺癌是女性最常见的恶性肿瘤之一。机器学习在医学领域展现出巨大潜力，可以利用大量数据建立预测模型，帮助医生进行更准确的诊断。

### 形式化定义
- **输入空间**：特征向量 x ∈ R³⁰，表示肿瘤的30个测量特征
- **输出空间**：标签 y ∈ {0, 1}，0表示良性，1表示恶性
- **任务目标**：学习函数 f: R³⁰ → {0, 1}，最小化分类错误率

### 难点与挑战
- 特征高度相关和多重共线性
- 小样本量导致的过拟合风险
- 实验效率要求

## 安装与环境要求

### Python版本
- Python 3.12.3

### 主要依赖库
- numpy: 1.26.4
- pandas: 2.2.3
- matplotlib: 3.8.4
- scikit-learn: 1.5.1

### 安装命令
```bash
pip install numpy pandas matplotlib scikit-learn
```

## 数据集描述

### 数据来源
使用Scikit-learn内置的乳腺癌数据集（Breast Cancer Wisconsin Diagnostic Dataset），包含从569名患者处获得的30个特征。

### 数据规模
- 样本数：569
- 特征数：30（10个基础变量的均值、标准差、最值）
- 目标变量：二分类（良性/恶性）

### 基础变量
- 半径（radius）
- 纹理（texture）
- 周长（perimeter）
- 面积（area）
- 平滑度（smoothness）
- 紧凑度（compactness）
- 凹陷（concavity）
- 凹点（concave points）
- 对称性（symmetry）
- 分形维数（fractal dimension）

## 数据预处理

### 缺失值检查
数据集无缺失值，无需插补。

### 特征缩放
使用StandardScaler将特征缩放到均值为0、方差为1的标准正态分布，避免大尺度特征主导训练。

### 特征选择
采用嵌入法（Embedded Method），基于随机森林的重要性评分，从30个特征中选择10个最重要特征。

## 算法原理

### 支持向量机（SVM）
- **核心思想**：在高维空间寻找最大间隔超平面划分正负样本
- **软间隔**：允许部分样本违反间隔约束，通过松弛变量惩罚
- **关键公式**：
  ```
  Min(1/2||w||² + C∑ξᵢ)
  s.t. yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
  ```

### K近邻（KNN）
- **核心思想**：基于实例的学习，通过计算与训练样本的距离进行分类
- **距离度量**：欧几里得距离、曼哈顿距离、余弦距离
- **最佳参数**：k=4，距离度量=余弦距离

## 实验设计

### 实验环境
- 仅使用CPU训练
- 随机种子：42（确保可复现）

### 训练策略
- **交叉验证**：5折CV
- **评价指标**：准确率、F1-score、精确率、召回率
- **超参数调优**：GridSearchCV
  - SVM：C ∈ [0.1, 1, 10, 100, 1000]
  - KNN：k ∈ [1, 2, ..., 20]，距离 ∈ [euclidean, manhattan, cosine]

### 基线对比
使用DummyClassifier（多数类分类器）作为基线，准确率约0.623。

## 实验结果

### 主要结果
| 模型 | 准确率 | F1-score | 精确率 | 召回率 |
|------|--------|----------|--------|--------|
| SVM  | 0.9737 | 0.9737   | 0.9737 | 0.9737 |
| KNN  | 0.9825 | 0.9825   | 0.9825 | 0.9825 |
| Dummy| 0.6228 | 0.4780   | 0.3898 | 0.6228 |

### 可视化分析
- **ROC曲线**：SVM和KNN的AUC均接近1.00
- **PR曲线**：两种算法性能几乎相同，SVM略胜于稳定性

### 消融实验
- **多组合消融**：去掉"worst"特征导致最大性能下降（SVM下降0.0789）
- **极端消融**：单特征中"worst perimeter"表现最优
- **结论**："worst"系列特征贡献度最高，其次是"mean"特征

### 误差分析
- SVM误分类主要由于决策值接近0，特征贡献相互矛盾
- KNN误分类由于邻域内多数样本标签不一致
- 分析显示误分类样本往往位于良性/恶性分界处

## 使用方法

### 运行Notebook
1. 确保安装所需依赖
2. 打开`hw1.ipynb`文件
3. 依次运行所有单元格

### 关键代码片段

```python
# 数据加载和预处理
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

data = load_breast_cancer()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.data)

# 特征选择
rf = RandomForestClassifier(random_state=42)
rf.fit(data_scaled, data.target)
selector = SelectFromModel(rf, prefit=True)
selected_features = selector.get_support()

# 模型训练
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=4, metric='cosine')
```

## 结论

### 主要发现
1. SVM和KNN在乳腺癌检测任务上均表现出色，准确率超过95%
2. 特征缩放和选择对提升模型性能至关重要
3. "worst"系列特征对分类贡献最大
4. 两种算法性能接近，SVM更适合大规模数据，KNN更简单易解释

### 个人收获
- 掌握了消融实验的设计和分析方法
- 学会了提前规划实验流程，针对性进行预处理
- 通过文献阅读深化了对算法的理解

### 后续改进方向
- 扩大样本量以减少过拟合
- 尝试更多算法（如逻辑回归）
- 纳入更多核函数和超参数选择
- 从二分类扩展到概率预测

## 参考文献

[1] Scikit-learn Breast Cancer Dataset Documentation  
[2] Support Vector Machines for Pattern Classification  
[3] K-Nearest Neighbor Algorithm: Review and Application  
[4] On Model Stability as a Function of Random Seed  
[5] Bias-kNN: A New Method for Handling Biased Outputs in KNN

## 项目结构

```
Sklearn_breast_cancer/
├── hw1.ipynb          # 主实验Notebook
├── README.md          # 项目说明文档
└── (其他生成的文件)
```

---

*本项目为机器学习课程作业，展示了SVM和KNN在医疗诊断领域的应用。*
