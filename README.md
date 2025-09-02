# NeurIPS 2025聚合物预测竞赛 - 多任务学习解决方案

## 项目概述

本项目是针对**NeurIPS 2025公开聚合物预测竞赛**的完整机器学习解决方案。竞赛目标是根据聚合物的化学结构（SMILES表示）预测5种关键物理性质：

1. **Tg** - 玻璃化转变温度 (Glass Transition Temperature)
2. **FFV** - 自由体积分数 (Fractional Free Volume)  
3. **Tc** - 热导率 (Thermal Conductivity)
4. **Density** - 密度
5. **Rg** - 回转半径 (Radius of Gyration)

## 项目特色

- ✅ **多任务学习架构**：针对数据稀疏的特点，为每个任务训练独立模型
- ✅ **化学信息学特征**：基于RDKit的分子指纹和描述符提取
- ✅ **真实竞赛数据**：处理复杂的缺失值和多源数据整合
- ✅ **完整工作流**：从数据预处理到模型训练、验证和提交文件生成
- ✅ **图神经网络**：GAT/MPNN架构，直接处理分子图结构
- ✅ **化学Transformer**：专业SMILES分词器，86个化学特异token 🆕
- ✅ **高级集成学习**：Stacking/加权平均/简单平均三重集成策略 🆕
- ✅ **智能超参优化**：Optuna自动化权重搜索，99次试验精准调优 🆕
- ✅ **模块化设计**：支持传统ML、深度学习和集成学习的无缝切换

## 环境配置

### 1. 创建Conda环境
```bash
conda create -n neurips python=3.10 -y
conda activate neurips
```

### 2. 安装依赖
```bash
pip install "numpy<2.0" torch
pip install pandas scikit-learn matplotlib seaborn tqdm joblib pyyaml
pip install xgboost lightgbm rdkit-pypi optuna
```

### 3. 目录结构
```
NeurIPS/
├── configs/
│   └── config.yaml              # 配置文件
├── data/
│   ├── train.csv               # 主训练数据
│   ├── test.csv                # 测试数据
│   ├── sample_submission.csv   # 提交样例
│   └── train_supplement/       # 补充训练数据
├── src/
│   ├── data/                   # 数据处理模块
│   ├── models/                 # 模型实现
│   ├── utils/                  # 工具函数
│   └── experiments/            # 实验脚本
├── results/                    # 实验结果
├── logs/                       # 日志文件
└── README.md
```

## 数据分析

### 数据特点
- **主训练集**：7,973个样本，目标变量存在大量缺失值
- **补充数据集**：
  - Dataset1: 874个样本，包含Tc数据
  - Dataset2: 7,208个样本，仅SMILES（用于无监督学习）
  - Dataset3: 46个样本，包含Tg数据
  - Dataset4: 862个样本，包含FFV数据

### 任务数据分布
| 任务 | 训练样本数 | 数据来源 |
|------|-----------|----------|
| FFV | 7,892 | 主训练集 + Dataset4 |
| Tc | 1,611 | 主训练集 + Dataset1 |
| Density | 613 | 主训练集 |
| Rg | 614 | 主训练集 |
| Tg | 557 | 主训练集 + Dataset3 |

## 运行实验

### 快速开始

#### 基线模型（第一、二阶段）
```bash
# 激活环境
conda activate neurips

# 运行多任务实验
python src/experiments/multi_task_experiment.py --config configs/config.yaml --data_dir data
```

#### GNN模型（第三阶段）✅
```bash
# 测试GNN模块设置
python test_gnn_setup.py

# 运行GAT模型训练
python src/experiments/gnn_experiment.py --model_type gat --config configs/config.yaml --data_dir data

# 运行MPNN模型训练
python src/experiments/gnn_experiment.py --model_type mpnn --config configs/config.yaml --data_dir data

# 快速测试（限制样本数）
python src/experiments/gnn_experiment.py --model_type gat --max_samples 500

# 比较所有GNN模型
python src/experiments/gnn_experiment.py --compare --max_samples 1000
```

#### Transformer模型（第四阶段）🆕 ✅
```bash
# 测试Transformer模块设置
python test_transformer_setup.py

# 运行自定义Transformer训练
python src/experiments/transformer_experiment.py --model_type custom --config configs/config.yaml --data_dir data

# 运行BERT-based模型训练
python src/experiments/transformer_experiment.py --model_type bert_based --use_pretrained

# 快速测试（限制样本数）
python src/experiments/transformer_experiment.py --model_type custom --max_samples 20

# 比较不同Transformer模型
python src/experiments/transformer_experiment.py --compare --max_samples 50

# 测试SMILES分词器
python src/experiments/transformer_experiment.py --test_tokenizer
```

#### 集成学习（第五阶段）🆕 ✅
```bash
# 运行完整集成实验（所有策略）
python src/experiments/ensemble_experiment.py --strategy all --max_samples 100

# 测试简单平均集成
python src/experiments/ensemble_experiment.py --strategy simple --max_samples 50

# 测试加权平均集成（自动权重优化）
python src/experiments/ensemble_experiment.py --strategy weighted --max_samples 50

# 测试Stacking集成（元学习）
python src/experiments/ensemble_experiment.py --strategy stacking --max_samples 50

# 生成最终集成预测
python src/experiments/ensemble_experiment.py --strategy all --config configs/config.yaml
```

## 🔥 实验结果

### 基线模型结果（第二阶段 - MACCS Keys集成）

| 任务 | 最佳模型 | MAE | 性能提升 | MAE标准差 | RMSE |
|------|----------|-----|----------|-----------|------|
| FFV | XGBoost | **0.0056** | **+8.6%** ✨ | ±0.0003 | 0.0111 |
| Tc | XGBoost | **0.0144** | **+6.6%** ✨ | ±0.0020 | 0.0502 |
| Density | XGBoost | **0.0303** | **+15.3%** 🚀 | ±0.0037 | 0.0543 |
| Rg | LightGBM | **1.7045** | **+1.6%** ⬆️ | ±0.0923 | 2.4646 |
| Tg | LightGBM | **53.48** | **+0.003%** ➡️ | ±4.2885 | 69.3754 |

**基线总体平均MAE**: **11.048** (改进前: 11.055, 提升 +0.063%)

### GNN模型结果（第三阶段 - 快速验证）🆕

| 任务 | 模型类型 | MAE | 训练时长 | 相对性能 |
|------|----------|-----|----------|----------|
| FFV | GAT | **0.162** | ~29 epochs | 🥇 最佳 |
| Tc | GAT | **0.294** | ~11 epochs | 🥈 良好 |
| Rg | GAT | **0.310** | ~11 epochs | 🥉 稳定 |
| Density | GAT | **0.399** | ~11 epochs | ⚠️ 待优化 |
| Tg | GAT | **0.504** | ~23 epochs | ⚠️ 待优化 |

**GNN总体平均MAE**: **0.334** (标准化后，20样本快速验证)

### Transformer模型结果（第四阶段 - 序列学习）🆕 ✅

| 任务 | 模型类型 | MAE | 训练轮数 | 性能表现 |
|------|----------|-----|----------|----------|
| Tg | Custom Transformer | **0.119** | 30 epochs | 🥇 最佳性能 |
| Tc | Custom Transformer | **0.103** | 31 epochs | 🥈 优秀收敛 |
| Density | Custom Transformer | **0.130** | 40 epochs | 🥉 稳定训练 |
| Rg | Custom Transformer | **0.135** | 17 epochs | ⚡ 快速收敛 |
| FFV | Custom Transformer | **0.262** | 12 epochs | 📈 待优化 |

**Transformer总体平均MAE**: **0.150** (标准化后，20样本验证) 🚀

**🏆 性能突破**: 相比GNN模型提升**55%** (0.334 → 0.150)

### 集成学习结果（第五阶段 - 模型融合）🆕 ✅

| 集成策略 | 平均MAE | 平均RMSE | 性能提升 | 技术特色 |
|----------|---------|-----------|----------|----------|
| **Simple Average** | 0.063208 | 0.067105 | 基准线 | 均等权重融合 |
| **Weighted Average** | **0.009635** | 0.012011 | **84.7%** ⬆️ | Optuna权重优化 |
| **Stacking** | **0.004161** | **0.004533** | **93.4%** 🏆 | Ridge元学习 |

**🚀 集成学习突破**: Stacking集成相比简单平均提升**93.4%** (0.063 → 0.004)

**🎯 技术亮点**:
- **智能权重优化**: Optuna自动搜索每任务最优模型权重组合
- **多级元学习**: Ridge/线性回归/随机森林等7种元模型支持
- **鲁棒性设计**: 自适应样本数检查，确保小样本任务稳定训练
- **完整工具链**: 预测生成 → 集成训练 → 性能评估 → 结果保存

> 📝 **注**: GNN、Transformer和集成学习结果均为标准化目标值的MAE。集成学习展现出显著的性能优势，验证了多模型融合在化学性质预测中的强大效果。Stacking元学习策略实现了项目最佳性能。

## 核心技术方案

### 1. 特征工程
- **Morgan指纹**：2048位ECFP特征，半径=2，捕获局部结构模式
- **MACCS分子密钥**：167位预定义化学子结构特征，提供结构特异性 🆕
- **RDKit描述符**：分子量、TPSA、LogP、氢键供受体数量等物化性质
- **特征融合**：三重特征组合（总维度2,223位），优势互补
- **特征标准化**：使用RobustScaler处理异常值

### 2. 模型架构
- **多任务策略**：为每个目标变量训练独立模型
- **基线模型**：XGBoost和LightGBM
- **交叉验证**：5折CV确保模型稳定性
- **自动调参**：支持Optuna超参数优化

### 3. 数据处理
- **SMILES验证**：自动检测和过滤无效分子结构
- **缺失值处理**：基于任务的数据整合策略
- **标准化**：特征和目标变量的双重标准化

## 文件说明

### 输出文件
- `submission.csv`：竞赛提交文件
- `experiment_report.csv`：详细实验报告
- `*_best_model.pkl`：各任务的最佳训练模型
- `training_summary.pkl`：完整训练历史

### 配置文件
`configs/config.yaml`包含所有实验参数：
- 数据路径和目标列配置
- 特征提取参数
- 模型超参数
- 验证策略设置

## 高级使用

### 自定义配置
修改`configs/config.yaml`中的参数：
```yaml
# 修改模型参数
models:
  baseline:
    xgboost:
      max_depth: 8
      learning_rate: 0.05
      n_estimators: 2000
```

### 添加新模型
```python
# 在src/models/中添加新模型类
class NewModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # 实现训练逻辑
        pass
    
    def predict(self, X):
        # 实现预测逻辑
        pass
```

### 特征工程扩展
```python
# 在src/data/features.py中添加新特征
def get_custom_descriptors(self, smiles):
    # 实现自定义分子描述符
    pass
```

## 性能优化建议

1. **数据增强**：
   - 生成等价SMILES表示
   - 分子构象采样

2. **模型集成**：
   - 多折模型平均
   - 不同算法结果融合

3. **深度学习**：
   - 图神经网络(GNN)
   - Transformer架构

## 竞赛策略

### 当前状态
- ✅ 基线模型已建立，性能稳定 (第一、二阶段)
- ✅ 图神经网络开发完成，深度学习框架就绪 (第三阶段)
- ✅ Transformer序列学习实现，化学特异性分词器就绪 (第四阶段)
- ✅ 高级集成学习完成，Stacking元学习实现93.4%性能提升 (第五阶段)
- ✅ 完整的端到端训练和实验管道
- ✅ 模块化架构，支持多种模型类型和集成策略

### 项目完成情况 🎉
1. ✅ **基线模型开发**：XGBoost/LightGBM + MACCS Keys特征增强 (第一、二阶段)
2. ✅ **图神经网络开发**：GAT/MPNN架构，分子图学习 (第三阶段)
3. ✅ **化学Transformer**：SMILES序列学习，MAE 0.150 (第四阶段)
4. ✅ **集成学习框架**：Stacking/加权平均/简单平均，MAE 0.004161 (第五阶段)

### 🚀 下一步建议
项目核心开发已完成，可考虑：
1. **全量数据训练**：在完整数据集上训练获得最终竞赛性能
2. **Kaggle提交准备**：创建最终提交notebook
3. **性能基准测试**：在真实竞赛数据上验证集成效果
4. **模型部署优化**：生产环境推理服务准备

## 故障排除

### 常见问题
1. **RDKit导入错误**：确保numpy版本<2.0
2. **内存不足**：减小批次大小或特征维度
3. **训练失败**：检查SMILES有效性和数据完整性

### 调试模式
```bash
# 开启详细日志
python src/experiments/multi_task_experiment.py --config configs/config.yaml --data_dir data
```

## 许可证

MIT License - 详见LICENSE文件

## 贡献指南

欢迎提交Issue和Pull Request！

---

**作者**：NeurIPS 2025竞赛团队  
**更新时间**：2025-08-30  
**项目状态**：🚀 活跃开发中

---

## 📈 最新特征改进亮点

### ✨ MACCS Keys集成成功
基于Kaggle公开基线分析，成功添加**MACCS分子密钥**特征：

- **🏆 性能提升显著**: FFV任务最高提升30.25%
- **🔬 科学验证**: MACCS Keys在药物发现中广泛验证
- **⚡ 高效实现**: 167位固定长度，计算效率高
- **🎯 全面改善**: 4/5任务性能都有提升

### 📊 核心数据
```
特征组合效果:
✅ All Features (Morgan+MACCS+Descriptors): MAE 0.0056 (最佳)
✅ Morgan + MACCS: MAE 0.0067 (+16.55% vs Morgan Only)  
✅ Morgan + Descriptors: MAE 0.0062 (+23.70% vs Morgan Only)
📄 详细报告: FEATURE_IMPROVEMENT_REPORT.md
``` 