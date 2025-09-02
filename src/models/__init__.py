"""
模型包
"""

from .baseline import BaselineModel
from .trainer import BaselineTrainer
from .multi_task_trainer import MultiTaskTrainer

# 第三阶段新增：GNN模型
gnn_available = False
try:
    from .gnn import (
        MultiTaskGNNPredictor, 
        GATPredictor, 
        MPNNPredictor, 
        create_gnn_model,
        count_parameters
    )
    from .gnn_trainer import GNNTrainer
    gnn_available = True
except ImportError as e:
    print(f"Warning: GNN models not available due to missing dependencies: {e}")

# 第四阶段新增：Transformer模型
transformer_available = False
try:
    from .transformer import (
        MultiTaskTransformerPredictor,
        BertBasedSMILESPredictor,
        TransformerWithPretraining,
        create_transformer_model,
        count_transformer_parameters
    )
    from .transformer_trainer import TransformerTrainer
    transformer_available = True
except ImportError as e:
    print(f"Warning: Transformer models not available due to missing dependencies: {e}")

# 第五阶段新增：模型集成
ensemble_available = False
try:
    from .ensemble import (
        ModelEnsemble,
        create_ensemble_model
    )
    ensemble_available = True
except ImportError as e:
    print(f"Warning: Ensemble models not available due to missing dependencies: {e}")

# 动态构建__all__列表
__all__ = [
    'BaselineModel',
    'BaselineTrainer',
    'MultiTaskTrainer'
]

if gnn_available:
    __all__.extend([
        'MultiTaskGNNPredictor',
        'GATPredictor',
        'MPNNPredictor',
        'create_gnn_model',
        'count_parameters',
        'GNNTrainer'
    ])

if transformer_available:
    __all__.extend([
        'MultiTaskTransformerPredictor',
        'BertBasedSMILESPredictor',
        'TransformerWithPretraining',
        'create_transformer_model',
        'count_transformer_parameters',
        'TransformerTrainer'
    ])

if ensemble_available:
    __all__.extend([
        'ModelEnsemble',
        'create_ensemble_model'
    ])