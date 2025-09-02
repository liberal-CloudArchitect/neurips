"""
工具函数包
"""

from .config import load_config
from .logger import setup_logger
from .metrics import calculate_mae, calculate_rmse

# 第五阶段新增：预测生成器
prediction_generator_available = False
try:
    from .prediction_generator import PredictionGenerator, create_prediction_generator
    prediction_generator_available = True
except ImportError as e:
    print(f"Warning: Prediction generator not available due to missing dependencies: {e}")

__all__ = [
    'load_config',
    'setup_logger', 
    'calculate_mae',
    'calculate_rmse'
]

if prediction_generator_available:
    __all__.extend([
        'PredictionGenerator',
        'create_prediction_generator'
    ])