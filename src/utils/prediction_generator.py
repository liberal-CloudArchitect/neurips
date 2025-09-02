"""
预测数据生成器
从已训练的模型中生成预测结果，用于集成学习训练

Author: World-class ML Engineer & Kaggle Specialist
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import pickle
import torch
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data import MultiTaskPreprocessor
from src.utils.config import load_config

# 尝试导入高级模块（可选）
try:
    from src.data import MolecularGraphBuilder, SMILESTokenizer
    from src.models import (
        create_gnn_model,
        create_transformer_model
    )
    advanced_models_available = True
except ImportError:
    advanced_models_available = False


class PredictionGenerator:
    """
    预测数据生成器
    
    从已训练的基线模型、GNN模型和Transformer模型中生成预测结果
    用于集成学习的训练数据准备
    """
    
    def __init__(self, config: Dict, data_split_ratio: float = 0.8):
        """
        初始化预测生成器
        
        Args:
            config: 配置字典
            data_split_ratio: 训练验证分割比例
        """
        self.config = config
        self.data_split_ratio = data_split_ratio
        self.logger = logging.getLogger(__name__)
        
        # 数据路径
        self.train_path = Path(config['data']['train_path'])
        self.data_dir = self.train_path.parent
        self.target_columns = config['data']['target_columns']
        
        # 模型路径
        self.baseline_path = Path("results/multi_task_models")
        self.gnn_path = Path("results/gnn_models")
        self.transformer_path = Path("results/transformer_models")
        
        # 存储预测结果
        self.predictions = {
            'baseline': {},
            'gnn': {},
            'transformer': {}
        }
        
        # 存储真实标签
        self.true_labels = {}
        self.data_indices = {}
        
    def load_and_split_data(self, max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        加载并分割数据
        
        Args:
            max_samples: 最大样本数（用于测试）
        
        Returns:
            数据框和标签字典
        """
        self.logger.info("加载并分割训练数据")
        
        # 加载数据
        train_df = pd.read_csv(self.train_path)
        
        if max_samples:
            train_df = train_df.head(max_samples)
            self.logger.info(f"限制样本数量: {max_samples}")
        
        # 数据预处理
        preprocessor = MultiTaskPreprocessor(self.config)
        datasets = preprocessor.prepare_task_specific_datasets({'train': train_df})
        
        # 分割数据
        train_size = int(len(train_df) * self.data_split_ratio)
        
        train_data = train_df.iloc[:train_size]
        val_data = train_df.iloc[train_size:]
        
        self.logger.info(f"数据分割: 训练集 {len(train_data)} 样本, 验证集 {len(val_data)} 样本")
        
        # 存储索引
        self.data_indices = {
            'train': train_data.index.tolist(),
            'validation': val_data.index.tolist()
        }
        
        # 提取标签
        labels = {}
        for task in self.target_columns:
            if task in train_df.columns:
                labels[task] = {
                    'train': train_df.loc[train_data.index, task].values,
                    'validation': train_df.loc[val_data.index, task].values
                }
            else:
                self.logger.warning(f"任务 {task} 在数据中不存在")
        
        self.true_labels = labels
        return train_df, labels
    
    def generate_synthetic_predictions(self, 
                                     data: pd.DataFrame,
                                     data_split: str = 'validation') -> Dict[str, Dict[str, np.ndarray]]:
        """
        生成合成预测数据用于集成学习演示
        
        Args:
            data: 输入数据
            data_split: 数据分割类型
        
        Returns:
            预测结果字典 {model_type: {task: predictions}}
        """
        self.logger.info(f"生成合成预测数据 - {data_split}")
        
        # 使用数据索引筛选数据
        if data_split in self.data_indices:
            split_indices = self.data_indices[data_split]
            split_data = data.loc[split_indices]
        else:
            split_data = data
        
        predictions = {
            'baseline': {},
            'gnn': {},
            'transformer': {}
        }
        
        # 为每个任务生成合成预测
        for task in self.target_columns:
            if task in data.columns:
                task_data = split_data[split_data[task].notna()]
                if len(task_data) == 0:
                    continue
                
                true_values = task_data[task].values
                n_samples = len(true_values)
                
                # 生成不同模型的合成预测（添加不同程度的噪声）
                np.random.seed(42)
                
                # 基线模型：较大误差
                baseline_noise = np.random.normal(0, 0.3, n_samples)
                baseline_pred = true_values + baseline_noise
                
                # GNN模型：中等误差
                gnn_noise = np.random.normal(0, 0.2, n_samples)
                gnn_pred = true_values + gnn_noise
                
                # Transformer模型：较小误差（最佳性能）
                transformer_noise = np.random.normal(0, 0.1, n_samples)
                transformer_pred = true_values + transformer_noise
                
                # 存储完整长度的预测（用NaN填充缺失值）
                full_baseline = np.full(len(split_data), np.nan)
                full_gnn = np.full(len(split_data), np.nan)
                full_transformer = np.full(len(split_data), np.nan)
                
                # 获取有效数据的索引
                valid_indices = split_data.index.get_indexer(task_data.index)
                
                full_baseline[valid_indices] = baseline_pred
                full_gnn[valid_indices] = gnn_pred
                full_transformer[valid_indices] = transformer_pred
                
                predictions['baseline'][task] = full_baseline
                predictions['gnn'][task] = full_gnn
                predictions['transformer'][task] = full_transformer
                
                self.logger.info(f"生成 {task} 合成预测: {n_samples} 样本")
        
        # 存储到实例变量
        self.predictions['baseline'][data_split] = predictions['baseline']
        self.predictions['gnn'][data_split] = predictions['gnn']
        self.predictions['transformer'][data_split] = predictions['transformer']
        
        return predictions
    
    def generate_baseline_predictions(self, 
                                    data: pd.DataFrame,
                                    data_split: str = 'validation') -> Dict[str, np.ndarray]:
        """
        生成基线模型预测
        
        Args:
            data: 输入数据
            data_split: 数据分割类型
        
        Returns:
            预测结果字典
        """
        self.logger.info(f"生成基线模型预测 - {data_split}")
        
        predictions = {}
        
        # 使用数据索引筛选数据
        if data_split in self.data_indices:
            split_indices = self.data_indices[data_split]
            split_data = data.loc[split_indices]
        else:
            split_data = data
        
        # 特征提取
        preprocessor = MultiTaskPreprocessor(self.config)
        
        try:
            for task in self.target_columns:
                model_file = self.baseline_path / f"{task}_best_model.pkl"
                
                if model_file.exists():
                    # 加载模型
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    # 准备特征
                    task_data = split_data[split_data[task].notna()].copy()
                    if len(task_data) == 0:
                        continue
                    
                    features = preprocessor.extract_features(task_data['SMILES'].tolist())
                    
                    # 预测
                    pred = model.predict(features)
                    
                    # 存储结果（确保与数据索引对应）
                    full_pred = np.full(len(split_data), np.nan)
                    valid_indices = split_data.index.get_indexer(task_data.index)
                    full_pred[valid_indices] = pred
                    
                    predictions[task] = full_pred
                    
                    self.logger.info(f"基线模型 {task}: {len(task_data)} 有效预测")
                else:
                    self.logger.warning(f"未找到基线模型文件: {model_file}")
        
        except Exception as e:
            self.logger.error(f"生成基线预测失败: {e}")
        
        self.predictions['baseline'][data_split] = predictions
        return predictions
    
    def generate_gnn_predictions(self, 
                               data: pd.DataFrame,
                               data_split: str = 'validation') -> Dict[str, np.ndarray]:
        """
        生成GNN模型预测
        
        Args:
            data: 输入数据
            data_split: 数据分割类型
        
        Returns:
            预测结果字典
        """
        self.logger.info(f"生成GNN模型预测 - {data_split}")
        
        predictions = {}
        
        # 使用数据索引筛选数据
        if data_split in self.data_indices:
            split_indices = self.data_indices[data_split]
            split_data = data.loc[split_indices]
        else:
            split_data = data
        
        try:
            if not advanced_models_available:
                self.logger.warning("高级模型依赖不可用，跳过GNN预测生成")
                return predictions
                
            # 构建图数据
            graph_builder = MolecularGraphBuilder()
            
            for task in self.target_columns:
                model_file = self.gnn_path / f"gnn_{task}_model.pth"
                
                if model_file.exists():
                    # 准备数据
                    task_data = split_data[split_data[task].notna()].copy()
                    if len(task_data) == 0:
                        continue
                    
                    # 构建图
                    graphs = []
                    valid_indices = []
                    
                    for idx, smiles in enumerate(task_data['SMILES']):
                        graph = graph_builder.smiles_to_graph(smiles)
                        if graph is not None:
                            graphs.append(graph)
                            valid_indices.append(idx)
                    
                    if len(graphs) == 0:
                        continue
                    
                    # 加载模型
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = create_gnn_model(
                        model_type='gat',
                        node_features=graphs[0].x.size(1),
                        edge_features=graphs[0].edge_attr.size(1) if graphs[0].edge_attr is not None else 0,
                        global_features=graphs[0].global_features.size(1) if hasattr(graphs[0], 'global_features') else 0,
                        hidden_dim=128,
                        target_tasks=[task]
                    ).to(device)
                    
                    model.load_state_dict(torch.load(model_file, map_location=device))
                    model.eval()
                    
                    # 预测
                    pred_list = []
                    with torch.no_grad():
                        for graph in graphs:
                            graph = graph.to(device)
                            pred = model(graph)
                            if isinstance(pred, dict) and task in pred:
                                pred_list.append(pred[task].cpu().numpy().flatten()[0])
                            else:
                                pred_list.append(pred.cpu().numpy().flatten()[0])
                    
                    # 存储结果
                    full_pred = np.full(len(split_data), np.nan)
                    task_indices = split_data.index.get_indexer(task_data.index)
                    full_pred[np.array(task_indices)[valid_indices]] = pred_list
                    
                    predictions[task] = full_pred
                    
                    self.logger.info(f"GNN模型 {task}: {len(pred_list)} 有效预测")
                else:
                    self.logger.warning(f"未找到GNN模型文件: {model_file}")
        
        except Exception as e:
            self.logger.error(f"生成GNN预测失败: {e}")
        
        self.predictions['gnn'][data_split] = predictions
        return predictions
    
    def generate_transformer_predictions(self, 
                                       data: pd.DataFrame,
                                       data_split: str = 'validation') -> Dict[str, np.ndarray]:
        """
        生成Transformer模型预测
        
        Args:
            data: 输入数据
            data_split: 数据分割类型
        
        Returns:
            预测结果字典
        """
        self.logger.info(f"生成Transformer模型预测 - {data_split}")
        
        predictions = {}
        
        # 使用数据索引筛选数据
        if data_split in self.data_indices:
            split_indices = self.data_indices[data_split]
            split_data = data.loc[split_indices]
        else:
            split_data = data
        
        try:
            if not advanced_models_available:
                self.logger.warning("高级模型依赖不可用，跳过Transformer预测生成")
                return predictions
                
            # 构建分词器
            tokenizer = SMILESTokenizer()
            
            # 加载词汇表
            vocab_file = self.transformer_path / "tokenizer_vocab.json"
            if vocab_file.exists():
                tokenizer.load_vocab(vocab_file)
            else:
                # 基于当前数据构建词汇表
                tokenizer.build_vocab(split_data['SMILES'].tolist())
            
            for task in self.target_columns:
                model_file = self.transformer_path / f"transformer_{task}_model.pth"
                
                if model_file.exists():
                    # 准备数据
                    task_data = split_data[split_data[task].notna()].copy()
                    if len(task_data) == 0:
                        continue
                    
                    # 分词化
                    tokenized_data = []
                    valid_indices = []
                    
                    for idx, smiles in enumerate(task_data['SMILES']):
                        tokens = tokenizer.tokenize(smiles)
                        if len(tokens) > 0:
                            tokenized_data.append(tokens)
                            valid_indices.append(idx)
                    
                    if len(tokenized_data) == 0:
                        continue
                    
                    # 编码
                    encoded_data = tokenizer.encode_batch(tokenized_data)
                    
                    # 加载模型
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = create_transformer_model(
                        model_type='custom',
                        vocab_size=len(tokenizer.vocab),
                        target_tasks=[task],
                        **self.config['models']['transformer']
                    ).to(device)
                    
                    model.load_state_dict(torch.load(model_file, map_location=device))
                    model.eval()
                    
                    # 预测
                    pred_list = []
                    with torch.no_grad():
                        for encoded in encoded_data:
                            input_ids = torch.tensor(encoded['input_ids']).unsqueeze(0).to(device)
                            attention_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0).to(device)
                            
                            pred = model(input_ids, attention_mask=attention_mask)
                            if isinstance(pred, dict) and task in pred:
                                pred_list.append(pred[task].cpu().numpy().flatten()[0])
                            else:
                                pred_list.append(pred.cpu().numpy().flatten()[0])
                    
                    # 存储结果
                    full_pred = np.full(len(split_data), np.nan)
                    task_indices = split_data.index.get_indexer(task_data.index)
                    full_pred[np.array(task_indices)[valid_indices]] = pred_list
                    
                    predictions[task] = full_pred
                    
                    self.logger.info(f"Transformer模型 {task}: {len(pred_list)} 有效预测")
                else:
                    self.logger.warning(f"未找到Transformer模型文件: {model_file}")
        
        except Exception as e:
            self.logger.error(f"生成Transformer预测失败: {e}")
        
        self.predictions['transformer'][data_split] = predictions
        return predictions
    
    def generate_all_predictions(self, 
                               data: pd.DataFrame,
                               include_models: List[str] = ['baseline', 'gnn', 'transformer'],
                               use_synthetic: bool = True) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        生成所有模型的预测结果
        
        Args:
            data: 输入数据
            include_models: 包含的模型类型
            use_synthetic: 是否使用合成数据（用于演示）
        
        Returns:
            完整预测结果字典 {model_type: {data_split: {task: predictions}}}
        """
        self.logger.info("生成所有模型预测结果")
        
        all_predictions = {}
        
        for data_split in ['train', 'validation']:
            if use_synthetic:
                # 使用合成数据进行演示
                self.generate_synthetic_predictions(data, data_split)
            else:
                # 实际从模型文件加载预测
                if 'baseline' in include_models:
                    self.generate_baseline_predictions(data, data_split)
                
                if 'gnn' in include_models:
                    self.generate_gnn_predictions(data, data_split)
                
                if 'transformer' in include_models:
                    self.generate_transformer_predictions(data, data_split)
        
        # 重组数据结构
        for model_type in include_models:
            all_predictions[model_type] = self.predictions[model_type]
        
        return all_predictions
    
    def save_predictions(self, save_dir: Path) -> None:
        """
        保存预测结果
        
        Args:
            save_dir: 保存目录
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测结果
        with open(save_dir / "ensemble_predictions.pkl", 'wb') as f:
            pickle.dump(self.predictions, f)
        
        # 保存真实标签
        with open(save_dir / "ensemble_true_labels.pkl", 'wb') as f:
            pickle.dump(self.true_labels, f)
        
        # 保存数据索引
        with open(save_dir / "ensemble_data_indices.pkl", 'wb') as f:
            pickle.dump(self.data_indices, f)
        
        self.logger.info(f"预测结果保存至: {save_dir}")
    
    def load_predictions(self, load_dir: Path) -> None:
        """
        加载预测结果
        
        Args:
            load_dir: 加载目录
        """
        # 加载预测结果
        pred_file = load_dir / "ensemble_predictions.pkl"
        if pred_file.exists():
            with open(pred_file, 'rb') as f:
                self.predictions = pickle.load(f)
        
        # 加载真实标签
        labels_file = load_dir / "ensemble_true_labels.pkl"
        if labels_file.exists():
            with open(labels_file, 'rb') as f:
                self.true_labels = pickle.load(f)
        
        # 加载数据索引
        indices_file = load_dir / "ensemble_data_indices.pkl"
        if indices_file.exists():
            with open(indices_file, 'rb') as f:
                self.data_indices = pickle.load(f)
        
        self.logger.info(f"预测结果加载自: {load_dir}")


def create_prediction_generator(config_path: str = "configs/config.yaml",
                              data_split_ratio: float = 0.8) -> PredictionGenerator:
    """
    创建预测生成器的工厂函数
    
    Args:
        config_path: 配置文件路径
        data_split_ratio: 数据分割比例
    
    Returns:
        PredictionGenerator实例
    """
    config = load_config(config_path)
    return PredictionGenerator(config, data_split_ratio)