"""
Transformer模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

from .transformer import create_transformer_model, count_transformer_parameters
from ..data.smiles_tokenizer import SMILESTokenizer, create_smiles_tokenizer
from ..utils.metrics import calculate_multi_target_metrics, print_metrics


class SMILESDataset(Dataset):
    """SMILES数据集类"""
    
    def __init__(self, 
                 smiles_list: List[str],
                 targets: Optional[np.ndarray] = None,
                 tokenizer: SMILESTokenizer = None,
                 target_names: List[str] = None):
        """
        初始化SMILES数据集
        
        Args:
            smiles_list: SMILES字符串列表
            targets: 目标值数组 [n_samples, n_targets]
            tokenizer: SMILES分词器
            target_names: 目标名称列表
        """
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.target_names = target_names or []
        
        if self.targets is not None:
            assert len(self.smiles_list) == len(self.targets), "SMILES和目标数量不匹配"
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        # 分词编码
        if self.tokenizer:
            encoded = self.tokenizer.encode(smiles, return_tensors='pt')
            item = {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'smiles': smiles
            }
        else:
            item = {'smiles': smiles}
        
        # 添加目标值
        if self.targets is not None:
            targets_dict = {}
            for i, task_name in enumerate(self.target_names):
                if i < self.targets.shape[1]:
                    targets_dict[task_name] = torch.tensor(self.targets[idx, i], dtype=torch.float)
            item['targets'] = targets_dict
        
        return item


def collate_fn(batch):
    """批处理整理函数"""
    batch_data = {}
    
    # 处理输入序列
    if 'input_ids' in batch[0]:
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        batch_data['input_ids'] = input_ids
        batch_data['attention_mask'] = attention_mask
    
    # 处理目标值
    if 'targets' in batch[0]:
        targets = {}
        for task_name in batch[0]['targets'].keys():
            targets[task_name] = torch.stack([item['targets'][task_name] for item in batch])
        batch_data['targets'] = targets
    
    # 其他字段
    batch_data['smiles'] = [item['smiles'] for item in batch]
    
    return batch_data


class TransformerTrainer:
    """Transformer模型训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化Transformer训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.transformer_config = config.get('models', {}).get('transformer', {})
        self.training_config = config.get('training', {})
        self.validation_config = config.get('data', {}).get('validation', {})
        self.output_config = config.get('output', {})
        
        # 设备配置
        self.device = torch.device(
            self.training_config.get('device', 'cuda') 
            if torch.cuda.is_available() else 'cpu'
        )
        
        self.logger = logging.getLogger(__name__)
        
        # 存储训练结果
        self.results = {}
        self.best_models = {}
        self.scalers = {}
        self.tokenizer = None
        
    def prepare_tokenizer(self, smiles_list: List[str]) -> SMILESTokenizer:
        """
        准备SMILES分词器
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            训练好的分词器
        """
        vocab_size = self.transformer_config.get('vocab_size', 1000)
        max_length = self.transformer_config.get('max_seq_length', 512)
        
        self.logger.info(f"训练SMILES分词器 - 词汇表大小: {vocab_size}, 最大长度: {max_length}")
        
        # 创建并训练分词器
        self.tokenizer = create_smiles_tokenizer(
            smiles_list=smiles_list,
            vocab_size=vocab_size,
            max_length=max_length
        )
        
        self.logger.info(f"分词器训练完成，实际词汇表大小: {self.tokenizer.get_vocab_size()}")
        return self.tokenizer
    
    def create_data_loader(self,
                          smiles_list: List[str],
                          targets: Optional[np.ndarray] = None,
                          target_names: List[str] = None,
                          batch_size: int = 32,
                          shuffle: bool = True) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            smiles_list: SMILES字符串列表
            targets: 目标值数组
            target_names: 目标名称列表
            batch_size: 批次大小
            shuffle: 是否打乱
            
        Returns:
            数据加载器
        """
        dataset = SMILESDataset(
            smiles_list=smiles_list,
            targets=targets,
            tokenizer=self.tokenizer,
            target_names=target_names
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0  # 避免多进程问题
        )
    
    def train_single_task_model(self,
                               train_smiles: List[str],
                               train_targets: np.ndarray,
                               val_smiles: List[str],
                               val_targets: np.ndarray,
                               task_name: str,
                               model_type: str = 'custom') -> Tuple[nn.Module, Dict]:
        """
        训练单任务Transformer模型
        
        Args:
            train_smiles: 训练SMILES列表
            train_targets: 训练目标值
            val_smiles: 验证SMILES列表
            val_targets: 验证目标值
            task_name: 任务名称
            model_type: 模型类型
            
        Returns:
            (训练好的模型, 训练历史)
        """
        self.logger.info(f"开始训练 {task_name} 任务的 {model_type} Transformer模型...")
        
        # 准备分词器（如果还没有）
        if self.tokenizer is None:
            all_smiles = list(set(train_smiles + val_smiles))
            self.prepare_tokenizer(all_smiles)
        
        # 创建模型
        model_config = self.transformer_config.copy()
        model_config['task_names'] = [task_name]  # 单任务
        
        model = create_transformer_model(
            model_type=model_type,
            vocab_size=self.tokenizer.get_vocab_size(),
            config=model_config
        ).to(self.device)
        
        self.logger.info(f"模型参数数量: {count_transformer_parameters(model):,}")
        
        # 准备数据加载器
        batch_size = self.training_config.get('batch_size', 32)
        train_loader = self.create_data_loader(
            train_smiles, train_targets.reshape(-1, 1), [task_name], batch_size, shuffle=True
        )
        val_loader = self.create_data_loader(
            val_smiles, val_targets.reshape(-1, 1), [task_name], batch_size, shuffle=False
        )
        
        # 优化器和调度器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.training_config.get('learning_rate', 1e-4),
            weight_decay=self.training_config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
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
        epochs = self.training_config.get('epochs', 50)
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
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(batch['input_ids'], batch['attention_mask'])
                predictions = outputs['predictions'][task_name]
                targets = batch['targets'][task_name]
                
                # 计算损失
                loss = criterion(predictions, targets.unsqueeze(1))
                
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
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    outputs = model(batch['input_ids'], batch['attention_mask'])
                    predictions = outputs['predictions'][task_name]
                    targets = batch['targets'][task_name]
                    
                    loss = criterion(predictions, targets.unsqueeze(1))
                    val_losses.append(loss.item())
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
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
            if epoch % 5 == 0 or epoch == epochs - 1:
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
                               model_type: str = 'custom') -> Dict[str, Tuple[nn.Module, Dict]]:
        """
        训练多个单任务Transformer模型
        
        Args:
            task_data: 任务数据字典 {task_name: (smiles_list, targets)}
            model_type: 模型类型
            
        Returns:
            训练结果字典
        """
        results = {}
        
        # 首先收集所有SMILES用于训练分词器
        all_smiles = []
        for smiles_list, _ in task_data.values():
            all_smiles.extend(smiles_list)
        all_smiles = list(set(all_smiles))  # 去重
        
        # 训练分词器
        self.prepare_tokenizer(all_smiles)
        
        for task_name, (smiles_list, targets) in task_data.items():
            self.logger.info(f"=" * 50)
            self.logger.info(f"训练任务: {task_name}")
            self.logger.info(f"数据样本数: {len(smiles_list)}")
            
            # K折交叉验证
            n_splits = self.validation_config.get('n_splits', 5)
            random_state = self.validation_config.get('random_state', 42)
            
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            fold_results = []
            
            smiles_array = np.array(smiles_list)
            targets_array = np.array(targets)
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(smiles_array)):
                self.logger.info(f"训练第 {fold+1}/{n_splits} 折...")
                
                # 分割数据
                train_smiles = smiles_array[train_idx].tolist()
                val_smiles = smiles_array[val_idx].tolist()
                train_targets = targets_array[train_idx]
                val_targets = targets_array[val_idx]
                
                # 训练模型
                model, history = self.train_single_task_model(
                    train_smiles, train_targets, val_smiles, val_targets, task_name, model_type
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
        if self.tokenizer is None:
            raise ValueError("分词器未初始化")
        
        # 创建数据加载器
        loader = self.create_data_loader(smiles_list, batch_size=64, shuffle=False)
        
        # 预测
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(batch['input_ids'], batch['attention_mask'])
                pred = outputs['predictions'][task_name]
                predictions.extend(pred.cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def save_models_and_tokenizer(self, models: Dict[str, nn.Module], save_dir: str):
        """保存训练好的模型和分词器"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存分词器
        if self.tokenizer:
            tokenizer_path = save_path / "tokenizer_vocab.json"
            self.tokenizer.save_vocab(str(tokenizer_path))
        
        # 保存模型
        for task_name, model in models.items():
            model_path = save_path / f"transformer_{task_name}_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': self.transformer_config,
                'tokenizer_config': {
                    'vocab_size': self.tokenizer.get_vocab_size() if self.tokenizer else None,
                    'max_length': self.tokenizer.max_length if self.tokenizer else None
                }
            }, model_path)
            
            self.logger.info(f"模型已保存: {model_path}")
    
    def load_model_and_tokenizer(self, model_path: str, tokenizer_path: str, 
                                task_name: str, model_type: str = 'custom') -> nn.Module:
        """加载训练好的模型和分词器"""
        # 加载分词器
        self.tokenizer = SMILESTokenizer()
        self.tokenizer.load_vocab(tokenizer_path)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_config = checkpoint['model_config']
        model_config['task_names'] = [task_name]
        
        model = create_transformer_model(
            model_type=model_type,
            vocab_size=self.tokenizer.get_vocab_size(),
            config=model_config
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model