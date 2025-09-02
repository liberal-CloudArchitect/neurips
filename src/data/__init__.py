"""
数据处理包
"""

from .features import MolecularFeatureExtractor
from .preprocessor import DataPreprocessor
from .multi_task_preprocessor import MultiTaskPreprocessor
from .dataset import PolymerDataset

# 第三阶段新增：图数据构建器
graph_builder_available = False
try:
    from .graph_builder import MolecularGraphBuilder, collate_graphs
    graph_builder_available = True
except ImportError as e:
    print(f"Warning: Graph builder not available due to missing dependencies: {e}")

# 第四阶段新增：SMILES分词器
smiles_tokenizer_available = False
try:
    from .smiles_tokenizer import SMILESTokenizer, create_smiles_tokenizer
    smiles_tokenizer_available = True
except ImportError as e:
    print(f"Warning: SMILES tokenizer not available due to missing dependencies: {e}")

# 动态构建__all__列表
__all__ = [
    'MolecularFeatureExtractor',
    'DataPreprocessor',
    'MultiTaskPreprocessor',
    'PolymerDataset'
]

if graph_builder_available:
    __all__.extend([
        'MolecularGraphBuilder',
        'collate_graphs'
    ])

if smiles_tokenizer_available:
    __all__.extend([
        'SMILESTokenizer',
        'create_smiles_tokenizer'
    ])