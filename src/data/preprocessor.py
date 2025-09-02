"""
数据预处理器
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Dict, Optional, Union
import logging

from .features import MolecularFeatureExtractor


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Dict):
        """
        初始化数据预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config['data']
        self.features_config = config['features']
        
        self.feature_extractor = MolecularFeatureExtractor(
            morgan_radius=self.features_config['morgan_fingerprint']['radius'],
            morgan_n_bits=self.features_config['morgan_fingerprint']['n_bits'],
            use_features=self.features_config['morgan_fingerprint']['use_features'],
            use_chirality=self.features_config['morgan_fingerprint']['use_chirality']
        )
        
        self.scaler = None
        self.target_scalers = {}
        self.feature_names = None
        
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, train_path: str = None, test_path: str = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        加载训练和测试数据
        
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径
            
        Returns:
            训练数据和测试数据
        """
        if train_path is None:
            train_path = self.data_config['train_path']
        if test_path is None:
            test_path = self.data_config['test_path']
        
        self.logger.info(f"加载训练数据: {train_path}")
        train_df = pd.read_csv(train_path)
        
        test_df = None
        if test_path and pd.io.common.file_exists(test_path):
            self.logger.info(f"加载测试数据: {test_path}")
            test_df = pd.read_csv(test_path)
        
        self.logger.info(f"训练数据形状: {train_df.shape}")
        if test_df is not None:
            self.logger.info(f"测试数据形状: {test_df.shape}")
        
        return train_df, test_df
    
    def validate_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        验证和清洗数据
        
        Args:
            df: 数据DataFrame
            is_train: 是否为训练数据
            
        Returns:
            清洗后的数据
        """
        self.logger.info("开始数据验证和清洗...")
        
        smiles_col = self.data_config['smiles_column']
        target_cols = self.data_config['target_columns']
        
        # 检查必要列是否存在
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES列 '{smiles_col}' 不存在")
        
        if is_train:
            missing_targets = [col for col in target_cols if col not in df.columns]
            if missing_targets:
                raise ValueError(f"目标列不存在: {missing_targets}")
        
        # 记录原始数据信息
        original_size = len(df)
        self.logger.info(f"原始数据大小: {original_size}")
        
        # 移除SMILES为空的行
        df = df.dropna(subset=[smiles_col])
        self.logger.info(f"移除空SMILES后: {len(df)} 行 (移除了 {original_size - len(df)} 行)")
        
        # 验证SMILES有效性
        validation_result = self.feature_extractor.validate_smiles(df[smiles_col].tolist())
        self.logger.info(f"SMILES验证结果: 有效 {validation_result['valid_count']}, "
                        f"无效 {validation_result['invalid_count']}, "
                        f"有效率 {validation_result['valid_ratio']:.4f}")
        
        # 移除无效的SMILES
        valid_mask = df[smiles_col].apply(lambda x: self.feature_extractor.smiles_to_mol(x) is not None)
        df = df[valid_mask].reset_index(drop=True)
        self.logger.info(f"移除无效SMILES后: {len(df)} 行")
        
        # 处理目标变量中的异常值（仅训练数据）
        if is_train:
            for target_col in target_cols:
                if target_col in df.columns:
                    # 移除无穷大和NaN值
                    valid_mask = ~(df[target_col].isna() | np.isinf(df[target_col]))
                    invalid_count = (~valid_mask).sum()
                    if invalid_count > 0:
                        self.logger.warning(f"{target_col} 列有 {invalid_count} 个无效值，将被移除")
                        df = df[valid_mask].reset_index(drop=True)
                    
                    # 检测和处理异常值（使用IQR方法）
                    Q1 = df[target_col].quantile(0.25)
                    Q3 = df[target_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        self.logger.info(f"{target_col} 列检测到 {outlier_count} 个异常值 "
                                       f"(范围: [{lower_bound:.4f}, {upper_bound:.4f}])")
        
        self.logger.info(f"数据清洗完成，最终数据大小: {len(df)}")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        提取分子特征
        
        Args:
            df: 包含SMILES的数据
            
        Returns:
            特征DataFrame和特征名称列表
        """
        self.logger.info("开始提取分子特征...")
        
        smiles_col = self.data_config['smiles_column']
        smiles_list = df[smiles_col].tolist()
        
        # 配置特征提取参数
        use_morgan = True
        use_descriptors = self.features_config['rdkit_descriptors']['use_2d']
        descriptor_names = self.features_config['rdkit_descriptors']['descriptors']
        
        # 提取特征
        features_df = self.feature_extractor.extract_features(
            smiles_list=smiles_list,
            use_morgan=use_morgan,
            use_descriptors=use_descriptors,
            descriptor_names=descriptor_names
        )
        
        # 获取特征名称
        feature_names = self.feature_extractor.get_feature_names(
            use_morgan=use_morgan,
            use_descriptors=use_descriptors,
            descriptor_names=descriptor_names
        )
        
        self.logger.info(f"特征提取完成，特征维度: {features_df.shape[1]}")
        self.logger.info(f"Morgan指纹维度: {self.features_config['morgan_fingerprint']['n_bits']}")
        self.logger.info(f"RDKit描述符数量: {len(descriptor_names) if use_descriptors else 0}")
        
        return features_df, feature_names
    
    def prepare_datasets(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        准备训练和测试数据集
        
        Args:
            train_df: 训练数据
            test_df: 测试数据
            
        Returns:
            包含特征和目标的数据字典
        """
        self.logger.info("开始准备数据集...")
        
        # 清洗训练数据
        train_df_clean = self.validate_data(train_df, is_train=True)
        
        # 提取训练特征
        X_train, feature_names = self.extract_features(train_df_clean)
        self.feature_names = feature_names
        
        # 提取目标变量
        target_cols = self.data_config['target_columns']
        y_train = train_df_clean[target_cols].values
        
        # 准备测试数据
        X_test = None
        if test_df is not None:
            test_df_clean = self.validate_data(test_df, is_train=False)
            X_test, _ = self.extract_features(test_df_clean)
        
        # 标准化特征
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        # 标准化目标变量（可选）
        y_train_scaled = y_train.copy()
        for i, target_col in enumerate(target_cols):
            scaler = RobustScaler()
            y_train_scaled[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).ravel()
            self.target_scalers[target_col] = scaler
        
        dataset = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'y_train_scaled': y_train_scaled,
            'X_test': X_test_scaled,
            'feature_names': feature_names,
            'target_names': target_cols,
            'train_indices': train_df_clean.index.tolist(),
            'test_indices': test_df.index.tolist() if test_df is not None else None
        }
        
        self.logger.info(f"数据集准备完成:")
        self.logger.info(f"  训练集: {X_train_scaled.shape}")
        self.logger.info(f"  目标变量: {y_train.shape}")
        if X_test_scaled is not None:
            self.logger.info(f"  测试集: {X_test_scaled.shape}")
        
        return dataset
    
    def create_cv_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        创建交叉验证分割
        
        Args:
            X: 特征数组
            y: 目标数组
            
        Returns:
            交叉验证分割列表
        """
        validation_config = self.data_config['validation']
        
        if validation_config['method'] == 'kfold':
            kf = KFold(
                n_splits=validation_config['n_splits'],
                shuffle=validation_config['shuffle'],
                random_state=validation_config['random_state']
            )
            splits = list(kf.split(X, y))
        else:
            raise ValueError(f"不支持的验证方法: {validation_config['method']}")
        
        self.logger.info(f"创建了 {len(splits)} 折交叉验证")
        return splits
    
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        反标准化目标变量
        
        Args:
            y_scaled: 标准化后的目标变量
            
        Returns:
            原始尺度的目标变量
        """
        if not self.target_scalers:
            return y_scaled
        
        y_original = y_scaled.copy()
        target_cols = self.data_config['target_columns']
        
        for i, target_col in enumerate(target_cols):
            if target_col in self.target_scalers:
                scaler = self.target_scalers[target_col]
                y_original[:, i] = scaler.inverse_transform(y_scaled[:, i].reshape(-1, 1)).ravel()
        
        return y_original