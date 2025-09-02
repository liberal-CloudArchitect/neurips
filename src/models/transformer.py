"""
Transformer模型 - 基于BERT架构的化学SMILES多任务预测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertModel, BertConfig, AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple, Union
import math
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiTaskTransformerPredictor(nn.Module):
    """多任务Transformer预测器 - 基类"""
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 max_seq_length: int = 512,
                 dropout: float = 0.1,
                 task_names: List[str] = None,
                 pooling_strategy: str = 'cls'):
        """
        初始化多任务Transformer预测器
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            max_seq_length: 最大序列长度
            dropout: Dropout率
            task_names: 任务名称列表
            pooling_strategy: 池化策略 ('cls', 'mean', 'max', 'attention')
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.pooling_strategy = pooling_strategy
        
        # 默认任务名称
        if task_names is None:
            task_names = ['Density', 'Tc', 'Tg', 'Rg', 'FFV']
        self.task_names = task_names
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 注意力池化（如果使用）
        if pooling_strategy == 'attention':
            self.attention_pool = AttentionPooling(d_model)
        
        # 多任务预测头
        self.task_heads = nn.ModuleDict()
        for task in self.task_names:
            self.task_heads[task] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_padding_mask(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """创建padding mask"""
        return (input_ids == pad_token_id)
    
    def _pool_sequence(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """序列池化"""
        if self.pooling_strategy == 'cls':
            # 使用第一个token ([CLS])
            return hidden_states[:, 0]
        elif self.pooling_strategy == 'mean':
            # 平均池化（忽略padding）
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            return sum_hidden / (sum_mask + 1e-9)
        elif self.pooling_strategy == 'max':
            # 最大池化
            hidden_states = hidden_states.masked_fill(
                attention_mask.unsqueeze(-1).expand_as(hidden_states) == 0, -1e9
            )
            return torch.max(hidden_states, dim=1)[0]
        elif self.pooling_strategy == 'attention':
            # 注意力池化
            return self.attention_pool(hidden_states, attention_mask)
        else:
            return hidden_states[:, 0]  # 默认使用CLS
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # 创建attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 创建padding mask for transformer
        src_key_padding_mask = self._create_padding_mask(input_ids)
        
        # 词嵌入和位置编码
        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        embeddings = self.pos_encoder(embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码
        hidden_states = self.transformer(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 层归一化
        hidden_states = self.layer_norm(hidden_states)
        
        # 序列池化
        pooled_output = self._pool_sequence(hidden_states, attention_mask)
        
        # 多任务预测
        predictions = {}
        for task in self.task_names:
            predictions[task] = self.task_heads[task](pooled_output)
        
        result = {'predictions': predictions}
        
        if return_hidden_states:
            result['hidden_states'] = hidden_states
            result['pooled_output'] = pooled_output
        
        return result


class BertBasedSMILESPredictor(nn.Module):
    """基于预训练BERT的SMILES预测器"""
    
    def __init__(self,
                 pretrained_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
                 task_names: List[str] = None,
                 dropout: float = 0.1,
                 freeze_encoder: bool = False,
                 num_layers_to_freeze: int = 0):
        """
        初始化基于BERT的SMILES预测器
        
        Args:
            pretrained_model_name: 预训练模型名称
            task_names: 任务名称列表
            dropout: Dropout率
            freeze_encoder: 是否冻结编码器
            num_layers_to_freeze: 冻结的层数
        """
        super().__init__()
        
        # 默认任务名称
        if task_names is None:
            task_names = ['Density', 'Tc', 'Tg', 'Rg', 'FFV']
        self.task_names = task_names
        
        # 加载预训练模型
        try:
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
            self.config = self.bert.config
            self.hidden_size = self.config.hidden_size
        except Exception as e:
            print(f"Warning: Failed to load {pretrained_model_name}, using default BERT: {e}")
            # 回退到默认BERT配置
            config = BertConfig(
                vocab_size=1000,
                hidden_size=512,
                num_hidden_layers=12,
                num_attention_heads=8,
                intermediate_size=2048,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout
            )
            self.bert = BertModel(config)
            self.config = config
            self.hidden_size = config.hidden_size
        
        # 冻结参数
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        elif num_layers_to_freeze > 0:
            # 冻结指定层数
            for layer_idx in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 多任务预测头
        self.task_heads = nn.ModuleDict()
        for task in self.task_names:
            self.task_heads[task] = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 4, 1)
            )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 获取[CLS] token的表示
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # 多任务预测
        predictions = {}
        for task in self.task_names:
            predictions[task] = self.task_heads[task](pooled_output)
        
        result = {'predictions': predictions}
        
        if return_hidden_states:
            result['hidden_states'] = outputs.last_hidden_state
            result['pooled_output'] = pooled_output
        
        return result


class AttentionPooling(nn.Module):
    """注意力池化层"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        注意力池化
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # 计算注意力权重
        attention_scores = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # 应用mask
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax
        attention_weights = self.softmax(attention_scores)  # [batch_size, seq_len]
        
        # 加权求和
        pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        
        return pooled_output


class TransformerWithPretraining(nn.Module):
    """支持预训练的Transformer模型"""
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 max_seq_length: int = 512,
                 dropout: float = 0.1,
                 task_names: List[str] = None):
        super().__init__()
        
        self.transformer_predictor = MultiTaskTransformerPredictor(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_seq_length=max_seq_length,
            dropout=dropout,
            task_names=task_names
        )
        
        # MLM头（用于预训练）
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
    
    def forward_mlm(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """MLM前向传播（预训练用）"""
        outputs = self.transformer_predictor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True
        )
        
        hidden_states = outputs['hidden_states']
        mlm_logits = self.mlm_head(hidden_states)
        
        return {'mlm_logits': mlm_logits, 'hidden_states': hidden_states}
    
    def forward_prediction(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """任务预测前向传播（微调用）"""
        return self.transformer_predictor(input_ids=input_ids, attention_mask=attention_mask)


def create_transformer_model(model_type: str,
                            vocab_size: int,
                            config: Dict) -> nn.Module:
    """
    创建Transformer模型的工厂函数
    
    Args:
        model_type: 模型类型 ('custom', 'bert_based', 'with_pretraining')
        vocab_size: 词汇表大小
        config: 模型配置
        
    Returns:
        Transformer模型实例
    """
    task_names = config.get('task_names', ['Density', 'Tc', 'Tg', 'Rg', 'FFV'])
    
    if model_type == 'custom':
        return MultiTaskTransformerPredictor(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 512),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            dim_feedforward=config.get('dim_feedforward', 2048),
            max_seq_length=config.get('max_seq_length', 512),
            dropout=config.get('dropout', 0.1),
            task_names=task_names,
            pooling_strategy=config.get('pooling_strategy', 'cls')
        )
    
    elif model_type == 'bert_based':
        return BertBasedSMILESPredictor(
            pretrained_model_name=config.get('pretrained_model_name', 'DeepChem/ChemBERTa-77M-MLM'),
            task_names=task_names,
            dropout=config.get('dropout', 0.1),
            freeze_encoder=config.get('freeze_encoder', False),
            num_layers_to_freeze=config.get('num_layers_to_freeze', 0)
        )
    
    elif model_type == 'with_pretraining':
        return TransformerWithPretraining(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 512),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            dim_feedforward=config.get('dim_feedforward', 2048),
            max_seq_length=config.get('max_seq_length', 512),
            dropout=config.get('dropout', 0.1),
            task_names=task_names
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_transformer_parameters(model: nn.Module) -> int:
    """计算Transformer模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)