"""
PyTorch数据集类
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union


class PolymerDataset(Dataset):
    """聚合物数据集类"""
    
    def __init__(self, 
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 smiles: Optional[List[str]] = None,
                 transform: Optional[callable] = None):
        """
        初始化数据集
        
        Args:
            X: 特征数据
            y: 目标数据（可选，用于训练）
            smiles: SMILES字符串列表（可选）
            transform: 数据变换函数（可选）
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
            
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.smiles = smiles
        self.transform = transform
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            特征张量，如果有目标则返回(特征, 目标)元组
        """
        x = self.X[idx]
        
        if self.transform:
            x = self.transform(x)
        
        if self.y is not None:
            y = self.y[idx]
            return x, y
        else:
            return x
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.X.shape[1]
    
    def get_target_dim(self) -> int:
        """获取目标维度"""
        if self.y is not None:
            return self.y.shape[1] if len(self.y.shape) > 1 else 1
        return 0
    
    def get_smiles(self, idx: int) -> Optional[str]:
        """获取指定索引的SMILES字符串"""
        if self.smiles and idx < len(self.smiles):
            return self.smiles[idx]
        return None


class GraphDataset(Dataset):
    """图数据集类（用于GNN模型）"""
    
    def __init__(self, 
                 graphs: List,
                 y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 smiles: Optional[List[str]] = None):
        """
        初始化图数据集
        
        Args:
            graphs: 图对象列表
            y: 目标数据（可选）
            smiles: SMILES字符串列表（可选）
        """
        self.graphs = graphs
        if isinstance(y, pd.DataFrame):
            y = y.values
        self.y = torch.FloatTensor(y) if y is not None else None
        self.smiles = smiles
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.graphs)
    
    def __getitem__(self, idx: int):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            图对象，如果有目标则返回(图, 目标)元组
        """
        graph = self.graphs[idx]
        
        if self.y is not None:
            y = self.y[idx]
            return graph, y
        else:
            return graph
    
    def get_target_dim(self) -> int:
        """获取目标维度"""
        if self.y is not None:
            return self.y.shape[1] if len(self.y.shape) > 1 else 1
        return 0
    
    def get_smiles(self, idx: int) -> Optional[str]:
        """获取指定索引的SMILES字符串"""
        if self.smiles and idx < len(self.smiles):
            return self.smiles[idx]
        return None


class SequenceDataset(Dataset):
    """序列数据集类（用于Transformer模型）"""
    
    def __init__(self, 
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 smiles: Optional[List[str]] = None):
        """
        初始化序列数据集
        
        Args:
            input_ids: 输入ID张量
            attention_mask: 注意力掩码张量
            y: 目标数据（可选）
            smiles: SMILES字符串列表（可选）
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        if isinstance(y, pd.DataFrame):
            y = y.values
        self.y = torch.FloatTensor(y) if y is not None else None
        self.smiles = smiles
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.input_ids)
    
    def __getitem__(self, idx: int):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            输入字典，如果有目标则包含目标
        """
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
        
        if self.y is not None:
            item['labels'] = self.y[idx]
        
        return item
    
    def get_target_dim(self) -> int:
        """获取目标维度"""
        if self.y is not None:
            return self.y.shape[1] if len(self.y.shape) > 1 else 1
        return 0
    
    def get_smiles(self, idx: int) -> Optional[str]:
        """获取指定索引的SMILES字符串"""
        if self.smiles and idx < len(self.smiles):
            return self.smiles[idx]
        return None