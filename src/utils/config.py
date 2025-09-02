"""
配置文件加载工具
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 获取项目根目录
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "configs" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新配置字典
    
    Args:
        config: 原始配置字典
        updates: 需要更新的配置
        
    Returns:
        更新后的配置字典
    """
    config = config.copy()
    
    def _update_recursive(d: Dict[str, Any], u: Dict[str, Any]):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                _update_recursive(d[k], v)
            else:
                d[k] = v
    
    _update_recursive(config, updates)
    return config


def get_device(config: Dict[str, Any]) -> str:
    """
    根据配置和可用性获取设备
    
    Args:
        config: 配置字典
        
    Returns:
        设备字符串 ('cuda', 'mps', 'cpu')
    """
    import torch
    
    device = config.get('training', {}).get('device', 'cpu')
    
    if device == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    elif device == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'