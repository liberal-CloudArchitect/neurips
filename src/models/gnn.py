"""
图神经网络模型 - GAT和MPNN架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple, Union
import math


class MultiTaskGNNPredictor(nn.Module):
    """多任务图神经网络预测器 - 基类"""
    
    def __init__(self, 
                 atom_feature_dim: int,
                 edge_feature_dim: int,
                 global_feature_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.2,
                 pool_type: str = 'attention',
                 task_names: List[str] = None):
        """
        初始化多任务GNN预测器
        
        Args:
            atom_feature_dim: 原子特征维度
            edge_feature_dim: 边特征维度  
            global_feature_dim: 全局特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout率
            pool_type: 池化类型 ('mean', 'max', 'add', 'attention')
            task_names: 任务名称列表
        """
        super().__init__()
        
        self.atom_feature_dim = atom_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_type = pool_type
        
        # 默认任务名称
        if task_names is None:
            task_names = ['Density', 'Tc', 'Tg', 'Rg', 'FFV']
        self.task_names = task_names
        
        # 原子特征编码器
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 图池化层
        if pool_type == 'attention':
            self.pool = GlobalAttentionPool(hidden_dim)
        elif pool_type == 'set2set':
            self.pool = Set2SetPool(hidden_dim)
        else:
            self.pool = None  # 使用PyG内置池化
        
        # 分子级表示的维度
        if pool_type == 'attention':
            graph_feature_dim = hidden_dim
        elif pool_type == 'set2set':
            graph_feature_dim = hidden_dim * 2
        else:
            graph_feature_dim = hidden_dim
        
        # 如果有全局特征，添加到分子表示中
        if global_feature_dim > 0:
            self.global_encoder = nn.Linear(global_feature_dim, hidden_dim // 4)
            max_total_feature_dim = graph_feature_dim + hidden_dim // 4
        else:
            self.global_encoder = None
            max_total_feature_dim = graph_feature_dim
        
        # 添加一个自适应层来处理不同的输入维度
        self.feature_adapter = nn.Linear(max_total_feature_dim, hidden_dim)
        
        # 多任务预测头 - 统一使用hidden_dim作为输入
        self.task_heads = nn.ModuleDict()
        for task in self.task_names:
            self.task_heads[task] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def _pool_nodes(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """节点池化到图级表示"""
        if self.pool_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            return global_max_pool(x, batch)
        elif self.pool_type == 'add':
            return global_add_pool(x, batch)
        elif self.pool_type == 'attention':
            return self.pool(x, batch)
        elif self.pool_type == 'set2set':
            return self.pool(x, batch)
        else:
            return global_mean_pool(x, batch)
    
    def forward(self, data: Batch) -> Dict[str, torch.Tensor]:
        """前向传播 - 需要在子类中实现具体的GNN层"""
        raise NotImplementedError


class GATPredictor(MultiTaskGNNPredictor):
    """基于Graph Attention Network的多任务预测器"""
    
    def __init__(self, 
                 atom_feature_dim: int,
                 edge_feature_dim: int, 
                 global_feature_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.2,
                 pool_type: str = 'attention',
                 task_names: List[str] = None):
        """
        初始化GAT预测器
        
        Args:
            num_heads: 注意力头数
            其他参数同基类
        """
        super().__init__(
            atom_feature_dim=atom_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pool_type=pool_type,
            task_names=task_names
        )
        
        self.num_heads = num_heads
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                # 第一层：从原子特征到隐藏维度
                gat = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    concat=True
                )
            else:
                # 中间层
                gat = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    concat=True
                )
            
            self.gat_layers.append(gat)
            self.batch_norms.append(BatchNorm(hidden_dim))
    
    def forward(self, data: Batch) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码原子和边特征
        x = self.atom_encoder(data.x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_encoder(data.edge_attr)  # [num_edges, hidden_dim]
        
        # GAT层处理
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            # GAT卷积
            x_new = gat(x, data.edge_index, edge_attr)
            
            # 批归一化和残差连接
            if i > 0:  # 第一层后开始残差连接
                x_new = bn(x_new) + x
            else:
                x_new = bn(x_new)
            
            x = F.relu(x_new)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级池化
        graph_repr = self._pool_nodes(x, data.batch)  # [batch_size, hidden_dim]
        
        # 如果有全局特征，添加进来
        if self.global_encoder is not None and hasattr(data, 'global_features'):
            # 确保全局特征与batch大小匹配
            batch_size = data.batch.max().item() + 1
            if data.global_features.size(0) == batch_size:
                global_feat = self.global_encoder(data.global_features)
                graph_repr = torch.cat([graph_repr, global_feat], dim=1)
            else:
                # 如果维度不匹配，用零填充
                zero_padding = torch.zeros(
                    graph_repr.size(0), 
                    self.global_encoder.out_features, 
                    device=graph_repr.device
                )
                graph_repr = torch.cat([graph_repr, zero_padding], dim=1)
        
        # 使用特征适配器统一维度
        adapted_features = self.feature_adapter(graph_repr)
        
        # 多任务预测
        predictions = {}
        for task in self.task_names:
            predictions[task] = self.task_heads[task](adapted_features)
        
        return predictions


class MPNNPredictor(MultiTaskGNNPredictor):
    """基于Message Passing Neural Network的多任务预测器"""
    
    def __init__(self,
                 atom_feature_dim: int,
                 edge_feature_dim: int,
                 global_feature_dim: int, 
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.2,
                 pool_type: str = 'attention',
                 task_names: List[str] = None):
        """初始化MPNN预测器"""
        super().__init__(
            atom_feature_dim=atom_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pool_type=pool_type,
            task_names=task_names
        )
        
        # MPNN层
        self.mpnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            mpnn = MPNNLayer(hidden_dim, hidden_dim, dropout)
            self.mpnn_layers.append(mpnn)
            self.batch_norms.append(BatchNorm(hidden_dim))
    
    def forward(self, data: Batch) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码原子和边特征
        x = self.atom_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # MPNN层处理
        for i, (mpnn, bn) in enumerate(zip(self.mpnn_layers, self.batch_norms)):
            # MPNN消息传递
            x_new = mpnn(x, data.edge_index, edge_attr)
            
            # 批归一化和残差连接
            if i > 0:
                x_new = bn(x_new) + x
            else:
                x_new = bn(x_new)
            
            x = F.relu(x_new)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级池化
        graph_repr = self._pool_nodes(x, data.batch)
        
        # 如果有全局特征，添加进来
        if self.global_encoder is not None and hasattr(data, 'global_features'):
            # 确保全局特征与batch大小匹配
            batch_size = data.batch.max().item() + 1
            if data.global_features.size(0) == batch_size:
                global_feat = self.global_encoder(data.global_features)
                graph_repr = torch.cat([graph_repr, global_feat], dim=1)
            else:
                # 如果维度不匹配，用零填充
                zero_padding = torch.zeros(
                    graph_repr.size(0), 
                    self.global_encoder.out_features, 
                    device=graph_repr.device
                )
                graph_repr = torch.cat([graph_repr, zero_padding], dim=1)
        
        # 使用特征适配器统一维度
        adapted_features = self.feature_adapter(graph_repr)
        
        # 多任务预测
        predictions = {}
        for task in self.task_names:
            predictions[task] = self.task_heads[task](adapted_features)
        
        return predictions


class MPNNLayer(MessagePassing):
    """MPNN消息传递层"""
    
    def __init__(self, node_features: int, edge_features: int, dropout: float = 0.2):
        super().__init__(aggr='add')  # 使用求和聚合
        
        self.node_features = node_features
        self.edge_features = edge_features
        
        # 消息函数
        self.message_net = nn.Sequential(
            nn.Linear(node_features * 2 + edge_features, node_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_features * 2, node_features)
        )
        
        # 更新函数
        self.update_net = nn.Sequential(
            nn.Linear(node_features * 2, node_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_features * 2, node_features)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """计算消息"""
        # 拼接发送节点、接收节点和边特征
        message_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(message_input)
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """更新节点表示"""
        # 拼接原节点特征和聚合消息
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_net(update_input)


class GlobalAttentionPool(nn.Module):
    """全局注意力池化"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 计算注意力权重
        attn_weights = self.attention(x)  # [num_nodes, 1]
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # 加权平均池化
        out = global_add_pool(x * attn_weights, batch)
        return out


class Set2SetPool(nn.Module):
    """Set2Set池化"""
    
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM用于Set2Set
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = batch.max().item() + 1
        
        # 初始化LSTM隐状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        
        # Set2Set处理（简化版本）
        # 这里为了简化，使用均值池化的两倍作为输出
        pooled = global_mean_pool(x, batch)
        return torch.cat([pooled, pooled], dim=1)


def create_gnn_model(model_type: str,
                    atom_feature_dim: int,
                    edge_feature_dim: int,
                    global_feature_dim: int,
                    config: Dict) -> MultiTaskGNNPredictor:
    """
    创建GNN模型的工厂函数
    
    Args:
        model_type: 模型类型 ('gat', 'mpnn')
        atom_feature_dim: 原子特征维度
        edge_feature_dim: 边特征维度
        global_feature_dim: 全局特征维度
        config: 模型配置
        
    Returns:
        GNN模型实例
    """
    common_params = {
        'atom_feature_dim': atom_feature_dim,
        'edge_feature_dim': edge_feature_dim,
        'global_feature_dim': global_feature_dim,
        'hidden_dim': config.get('hidden_dim', 256),
        'num_layers': config.get('num_layers', 4),
        'dropout': config.get('dropout', 0.2),
        'pool_type': config.get('pool_type', 'attention'),
        'task_names': config.get('task_names', ['Density', 'Tc', 'Tg', 'Rg', 'FFV'])
    }
    
    if model_type.lower() == 'gat':
        return GATPredictor(
            **common_params,
            num_heads=config.get('num_heads', 8)
        )
    elif model_type.lower() == 'mpnn':
        return MPNNPredictor(**common_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)