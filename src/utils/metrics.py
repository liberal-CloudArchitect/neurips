"""
评估指标计算工具
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Union, Dict, List


def calculate_mae(y_true: Union[np.ndarray, pd.Series], 
                  y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    计算平均绝对误差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MAE值
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true: Union[np.ndarray, pd.Series], 
                   y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    计算均方根误差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true: Union[np.ndarray, pd.Series], 
                   y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    计算平均绝对百分比误差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MAPE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_multi_target_metrics(
    y_true: Union[np.ndarray, pd.DataFrame], 
    y_pred: Union[np.ndarray, pd.DataFrame],
    target_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    计算多目标预测的评估指标
    
    Args:
        y_true: 真实值矩阵 (n_samples, n_targets)
        y_pred: 预测值矩阵 (n_samples, n_targets)
        target_names: 目标名称列表
        
    Returns:
        包含各目标和总体指标的字典
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    
    n_targets = y_true.shape[1]
    
    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]
    
    metrics = {}
    
    # 计算每个目标的指标
    for i, target_name in enumerate(target_names):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        
        metrics[target_name] = {
            'mae': calculate_mae(y_true_i, y_pred_i),
            'rmse': calculate_rmse(y_true_i, y_pred_i),
            'mape': calculate_mape(y_true_i, y_pred_i)
        }
    
    # 计算总体平均指标
    all_mae = [metrics[name]['mae'] for name in target_names]
    all_rmse = [metrics[name]['rmse'] for name in target_names]
    all_mape = [metrics[name]['mape'] for name in target_names]
    
    metrics['overall'] = {
        'mean_mae': np.mean(all_mae),
        'mean_rmse': np.mean(all_rmse),
        'mean_mape': np.mean(all_mape),
        'total_mae': np.sum(all_mae)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]], 
                  title: str = "评估结果") -> None:
    """
    打印格式化的评估指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # 打印各目标指标
    for target_name, target_metrics in metrics.items():
        if target_name == 'overall':
            continue
            
        print(f"\n{target_name}:")
        for metric_name, value in target_metrics.items():
            print(f"  {metric_name.upper()}: {value:.6f}")
    
    # 打印总体指标
    if 'overall' in metrics:
        print(f"\n总体指标:")
        for metric_name, value in metrics['overall'].items():
            print(f"  {metric_name.upper()}: {value:.6f}")
    
    print("=" * 50)