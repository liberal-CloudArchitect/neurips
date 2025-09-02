"""
GNN模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import time
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .gnn import create_gnn_model, count_parameters
from ..data.graph_builder import MolecularGraphBuilder, collate_graphs
from ..utils.metrics import calculate_multi_target_metrics, print_metrics


class GNNTrainer:
    """GNN模型训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化GNN训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.gnn_config = config.get('models', {}).get('gnn', {})
        self.training_config = config.get('training', {})
        self.validation_config = config.get('data', {}).get('validation', {})
        self.output_config = config.get('output', {})
        
        # 设备配置
        self.device = torch.device(
            self.training_config.get('device', 'cuda') 
            if torch.cuda.is_available() else 'cpu'
        )
        
        self.logger = logging.getLogger(__name__)
        
        # 图构建器
        self.graph_builder = MolecularGraphBuilder()
        
        # 存储训练结果
        self.results = {}
        self.best_models = {}
        self.scalers = {}
        
        # 获取特征维度
        self.atom_feature_dim, self.edge_feature_dim, self.global_feature_dim = \
            self.graph_builder.get_feature_dimensions()
        
        self.logger.info(f"特征维度 - 原子: {self.atom_feature_dim}, "
                        f"边: {self.edge_feature_dim}, "
                        f"全局: {self.global_feature_dim}")
    
    def prepare_graph_data(self, 
                          smiles_list: List[str], 
                          targets: Optional[np.ndarray] = None,
                          target_names: List[str] = None) -> List[Data]:
        """
        准备图数据
        
        Args:
            smiles_list: SMILES字符串列表
            targets: 目标值数组 [n_samples, n_targets]
            target_names: 目标名称列表
            
        Returns:
            图数据列表
        """
        self.logger.info(f"开始构建 {len(smiles_list)} 个分子图...")
        
        # 批量转换为图数据
        graphs = self.graph_builder.batch_smiles_to_graphs(smiles_list)
        
        # 添加目标值
        if targets is not None and target_names is not None:
            for i, graph in enumerate(graphs):
                if i < len(targets):
                    # 为每个图添加目标值
                    for j, task in enumerate(target_names):
                        if j < targets.shape[1]:
                            setattr(graph, f'y_{task}', torch.tensor([targets[i, j]], dtype=torch.float))
        
        self.logger.info(f"成功构建 {len(graphs)} 个图 (失败: {len(smiles_list) - len(graphs)})")
        return graphs
    
    def create_data_loader(self, 
                          graphs: List[Data], 
                          batch_size: int = 32, 
                          shuffle: bool = True) -> PyGDataLoader:
        """
        创建图数据加载器
        
        Args:
            graphs: 图数据列表
            batch_size: 批次大小
            shuffle: 是否打乱
            
        Returns:
            数据加载器
        """
        return PyGDataLoader(
            graphs, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0  # 在Kaggle环境中避免多进程问题
        )
    
    def train_single_task_model(self,
                               train_graphs: List[Data],
                               val_graphs: List[Data],
                               task_name: str,
                               model_type: str = 'gat') -> Tuple[nn.Module, Dict]:
        """
        训练单任务GNN模型
        
        Args:
            train_graphs: 训练图数据
            val_graphs: 验证图数据  
            task_name: 任务名称
            model_type: 模型类型 ('gat', 'mpnn')
            
        Returns:
            (训练好的模型, 训练历史)
        """
        self.logger.info(f"开始训练 {task_name} 任务的 {model_type.upper()} 模型...")
        
        # 创建模型
        model_config = self.gnn_config.copy()
        model_config['task_names'] = [task_name]  # 单任务
        
        model = create_gnn_model(
            model_type=model_type,
            atom_feature_dim=self.atom_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            global_feature_dim=self.global_feature_dim,
            config=model_config
        ).to(self.device)
        
        self.logger.info(f"模型参数数量: {count_parameters(model):,}")
        
        # 准备数据加载器
        batch_size = self.training_config.get('batch_size', 32)
        train_loader = self.create_data_loader(train_graphs, batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_graphs, batch_size, shuffle=False)
        
        # 优化器和调度器
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_config.get('learning_rate', 1e-3),
            weight_decay=self.training_config.get('weight_decay', 1e-5)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=self.training_config.get('scheduler_patience', 5)
        )
        
        # 损失函数
        loss_type = self.training_config.get('loss', {}).get('type', 'smooth_l1')
        if loss_type == 'smooth_l1':
            criterion = nn.SmoothL1Loss()
        elif loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = nn.SmoothL1Loss()
        
        # 训练循环
        epochs = self.training_config.get('epochs', 100)
        patience = self.training_config.get('early_stopping', {}).get('patience', 10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_losses = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = batch.to(self.device)
                
                # 获取目标值
                y_true = getattr(batch, f'y_{task_name}')
                
                # 前向传播
                optimizer.zero_grad()
                predictions = model(batch)
                y_pred = predictions[task_name]
                
                # 计算损失
                loss = criterion(y_pred, y_true)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # 验证阶段
            model.eval()
            val_losses = []
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    y_true = getattr(batch, f'y_{task_name}')
                    
                    predictions = model(batch)
                    y_pred = predictions[task_name]
                    
                    loss = criterion(y_pred, y_true)
                    val_losses.append(loss.item())
                    
                    val_predictions.extend(y_pred.cpu().numpy())
                    val_targets.extend(y_true.cpu().numpy())
            
            avg_val_loss = np.mean(val_losses)
            val_mae = mean_absolute_error(val_targets, val_predictions)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            train_history['train_loss'].append(avg_train_loss)
            train_history['val_loss'].append(avg_val_loss)
            train_history['val_mae'].append(val_mae)
            train_history['learning_rate'].append(current_lr)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 日志输出
            if epoch % 10 == 0 or epoch == epochs - 1:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"Val MAE: {val_mae:.6f}, "
                    f"LR: {current_lr:.2e}"
                )
            
            # 早停
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        self.logger.info(f"{task_name} 任务训练完成，最佳验证MAE: {min(train_history['val_mae']):.6f}")
        
        return model, train_history
    
    def train_multi_task_models(self,
                               task_data: Dict[str, Tuple[List[str], np.ndarray]],
                               model_type: str = 'gat') -> Dict[str, Tuple[nn.Module, Dict]]:
        """
        训练多个单任务GNN模型
        
        Args:
            task_data: 任务数据字典 {task_name: (smiles_list, targets)}
            model_type: 模型类型
            
        Returns:
            训练结果字典
        """
        results = {}
        
        for task_name, (smiles_list, targets) in task_data.items():
            self.logger.info(f"=" * 50)
            self.logger.info(f"训练任务: {task_name}")
            self.logger.info(f"数据样本数: {len(smiles_list)}")
            
            # 准备图数据
            graphs = self.prepare_graph_data(
                smiles_list, 
                targets.reshape(-1, 1), 
                [task_name]
            )
            
            if len(graphs) == 0:
                self.logger.warning(f"任务 {task_name} 没有有效的图数据，跳过")
                continue
            
            # K折交叉验证
            n_splits = self.validation_config.get('n_splits', 5)
            random_state = self.validation_config.get('random_state', 42)
            
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(graphs)):
                self.logger.info(f"训练第 {fold+1}/{n_splits} 折...")
                
                # 分割数据
                train_graphs = [graphs[i] for i in train_idx]
                val_graphs = [graphs[i] for i in val_idx]
                
                # 训练模型
                model, history = self.train_single_task_model(
                    train_graphs, val_graphs, task_name, model_type
                )
                
                fold_results.append({
                    'model': model,
                    'history': history,
                    'best_val_mae': min(history['val_mae'])
                })
            
            # 选择最佳折
            best_fold = min(fold_results, key=lambda x: x['best_val_mae'])
            results[task_name] = (best_fold['model'], best_fold['history'])
            
            self.logger.info(f"任务 {task_name} 最佳MAE: {best_fold['best_val_mae']:.6f}")
        
        return results
    
    def predict(self, 
                model: nn.Module, 
                smiles_list: List[str], 
                task_name: str) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            model: 训练好的模型
            smiles_list: SMILES字符串列表
            task_name: 任务名称
            
        Returns:
            预测结果
        """
        # 准备图数据
        graphs = self.prepare_graph_data(smiles_list)
        
        if not graphs:
            return np.array([])
        
        # 创建数据加载器
        loader = self.create_data_loader(graphs, batch_size=64, shuffle=False)
        
        # 预测
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = model(batch)
                predictions.extend(pred[task_name].cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def save_models(self, models: Dict[str, nn.Module], save_dir: str):
        """保存训练好的模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for task_name, model in models.items():
            model_path = save_path / f"gnn_{task_name}_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': self.gnn_config,
                'atom_feature_dim': self.atom_feature_dim,
                'edge_feature_dim': self.edge_feature_dim,
                'global_feature_dim': self.global_feature_dim
            }, model_path)
            
            self.logger.info(f"模型已保存: {model_path}")
    
    def load_model(self, model_path: str, task_name: str, model_type: str = 'gat') -> nn.Module:
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        model_config = checkpoint['model_config']
        model_config['task_names'] = [task_name]
        
        model = create_gnn_model(
            model_type=model_type,
            atom_feature_dim=checkpoint['atom_feature_dim'],
            edge_feature_dim=checkpoint['edge_feature_dim'],
            global_feature_dim=checkpoint['global_feature_dim'],
            config=model_config
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model