"""
多任务数据预处理器 - 专门处理NeurIPS 2025聚合物预测竞赛数据
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Dict, Optional, Union
import logging
from pathlib import Path

from .features import MolecularFeatureExtractor


class MultiTaskPreprocessor:
    """多任务数据预处理器"""
    
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
            use_chirality=self.features_config['morgan_fingerprint']['use_chirality'],
            use_maccs=self.features_config['maccs_keys']['enabled']
        )
        
        self.scaler = None
        self.target_scalers = {}
        self.feature_names = None
        
        # 任务特定的数据
        self.task_datasets = {}
        self.combined_data = None
        
        self.logger = logging.getLogger(__name__)
    
    def load_competition_data(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """
        加载完整的竞赛数据
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            包含所有数据集的字典
        """
        data_dir = Path(data_dir)
        
        self.logger.info("加载NeurIPS 2025竞赛数据...")
        
        datasets = {}
        
        # 1. 加载主训练集
        train_path = data_dir / "train.csv"
        if train_path.exists():
            datasets['main_train'] = pd.read_csv(train_path)
            self.logger.info(f"主训练集: {datasets['main_train'].shape}")
        
        # 2. 加载测试集
        test_path = data_dir / "test.csv"
        if test_path.exists():
            datasets['test'] = pd.read_csv(test_path)
            self.logger.info(f"测试集: {datasets['test'].shape}")
        
        # 3. 加载补充数据集
        supplement_dir = data_dir / "train_supplement"
        if supplement_dir.exists():
            for i in range(1, 5):
                dataset_path = supplement_dir / f"dataset{i}.csv"
                if dataset_path.exists():
                    datasets[f'supplement_{i}'] = pd.read_csv(dataset_path)
                    self.logger.info(f"补充数据集{i}: {datasets[f'supplement_{i}'].shape}")
        
        return datasets
    
    def analyze_data_coverage(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        分析数据覆盖情况
        
        Args:
            datasets: 数据集字典
            
        Returns:
            数据分析结果
        """
        self.logger.info("分析数据覆盖情况...")
        
        analysis = {}
        
        # 分析主训练集
        if 'main_train' in datasets:
            main_df = datasets['main_train']
            target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            
            coverage = {}
            for col in target_cols:
                non_null_count = main_df[col].notnull().sum()
                coverage[col] = {
                    'count': non_null_count,
                    'percentage': non_null_count / len(main_df) * 100
                }
            
            analysis['main_train_coverage'] = coverage
            
            self.logger.info("主训练集目标变量覆盖情况:")
            for col, info in coverage.items():
                self.logger.info(f"  {col}: {info['count']} 样本 ({info['percentage']:.2f}%)")
        
        # 分析补充数据集
        supplement_info = {}
        for key, df in datasets.items():
            if key.startswith('supplement_'):
                supplement_info[key] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'target_available': len([col for col in df.columns if col != 'SMILES'])
                }
        
        analysis['supplement_info'] = supplement_info
        
        return analysis
    
    def prepare_task_specific_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        为每个任务准备特定的数据集
        
        Args:
            datasets: 原始数据集字典
            
        Returns:
            任务特定的数据集字典
        """
        self.logger.info("准备任务特定的数据集...")
        
        task_datasets = {}
        
        # 1. FFV任务数据
        ffv_data = []
        
        # 从主训练集获取FFV数据
        if 'main_train' in datasets:
            main_ffv = datasets['main_train'][['SMILES', 'FFV']].dropna()
            if len(main_ffv) > 0:
                ffv_data.append(main_ffv)
                self.logger.info(f"主训练集FFV数据: {len(main_ffv)} 样本")
        
        # 从补充数据集4获取FFV数据
        if 'supplement_4' in datasets:
            supp_ffv = datasets['supplement_4'][['SMILES', 'FFV']]
            ffv_data.append(supp_ffv)
            self.logger.info(f"补充数据集FFV数据: {len(supp_ffv)} 样本")
        
        if ffv_data:
            task_datasets['FFV'] = pd.concat(ffv_data, ignore_index=True)
            self.logger.info(f"FFV任务总数据: {len(task_datasets['FFV'])} 样本")
        
        # 2. Tg任务数据
        tg_data = []
        
        # 从主训练集获取Tg数据
        if 'main_train' in datasets:
            main_tg = datasets['main_train'][['SMILES', 'Tg']].dropna()
            if len(main_tg) > 0:
                tg_data.append(main_tg)
                self.logger.info(f"主训练集Tg数据: {len(main_tg)} 样本")
        
        # 从补充数据集3获取Tg数据
        if 'supplement_3' in datasets:
            supp_tg = datasets['supplement_3'][['SMILES', 'Tg']]
            tg_data.append(supp_tg)
            self.logger.info(f"补充数据集Tg数据: {len(supp_tg)} 样本")
        
        if tg_data:
            task_datasets['Tg'] = pd.concat(tg_data, ignore_index=True)
            self.logger.info(f"Tg任务总数据: {len(task_datasets['Tg'])} 样本")
        
        # 3. Tc任务数据
        tc_data = []
        
        # 从主训练集获取Tc数据
        if 'main_train' in datasets:
            main_tc = datasets['main_train'][['SMILES', 'Tc']].dropna()
            if len(main_tc) > 0:
                tc_data.append(main_tc)
                self.logger.info(f"主训练集Tc数据: {len(main_tc)} 样本")
        
        # 从补充数据集1获取TC_mean数据（重命名为Tc）
        if 'supplement_1' in datasets:
            supp_tc = datasets['supplement_1'][['SMILES', 'TC_mean']].copy()
            supp_tc.rename(columns={'TC_mean': 'Tc'}, inplace=True)
            tc_data.append(supp_tc)
            self.logger.info(f"补充数据集Tc数据: {len(supp_tc)} 样本")
        
        if tc_data:
            task_datasets['Tc'] = pd.concat(tc_data, ignore_index=True)
            self.logger.info(f"Tc任务总数据: {len(task_datasets['Tc'])} 样本")
        
        # 4. Density任务数据
        if 'main_train' in datasets:
            density_data = datasets['main_train'][['SMILES', 'Density']].dropna()
            if len(density_data) > 0:
                task_datasets['Density'] = density_data
                self.logger.info(f"Density任务数据: {len(density_data)} 样本")
        
        # 5. Rg任务数据
        if 'main_train' in datasets:
            rg_data = datasets['main_train'][['SMILES', 'Rg']].dropna()
            if len(rg_data) > 0:
                task_datasets['Rg'] = rg_data
                self.logger.info(f"Rg任务数据: {len(rg_data)} 样本")
        
        self.task_datasets = task_datasets
        return task_datasets
    
    def extract_features_for_task(self, task_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        为特定任务提取特征
        
        Args:
            task_data: 任务数据DataFrame，包含SMILES列
            
        Returns:
            特征DataFrame和特征名称列表
        """
        smiles_list = task_data['SMILES'].tolist()
        
        # 验证SMILES有效性
        validation_result = self.feature_extractor.validate_smiles(smiles_list)
        self.logger.info(f"SMILES验证: 有效{validation_result['valid_count']}, "
                        f"无效{validation_result['invalid_count']}, "
                        f"有效率{validation_result['valid_ratio']:.4f}")
        
        # 提取特征
        use_morgan = self.features_config['morgan_fingerprint']['enabled']
        use_maccs = self.features_config['maccs_keys']['enabled']
        use_descriptors = self.features_config['rdkit_descriptors']['use_2d']
        descriptor_names = self.features_config['rdkit_descriptors']['descriptors']
        
        features_df = self.feature_extractor.extract_features(
            smiles_list=smiles_list,
            use_morgan=use_morgan,
            use_maccs=use_maccs,
            use_descriptors=use_descriptors,
            descriptor_names=descriptor_names
        )
        
        feature_names = self.feature_extractor.get_feature_names(
            use_morgan=use_morgan,
            use_maccs=use_maccs,
            use_descriptors=use_descriptors,
            descriptor_names=descriptor_names
        )
        
        return features_df, feature_names
    
    def prepare_single_task_dataset(self, task_name: str, 
                                   test_size: float = 0.2) -> Dict:
        """
        为单个任务准备训练和验证数据
        
        Args:
            task_name: 任务名称 (FFV, Tg, Tc, Density, Rg)
            test_size: 验证集比例
            
        Returns:
            包含训练和验证数据的字典
        """
        if task_name not in self.task_datasets:
            raise ValueError(f"任务 {task_name} 的数据不存在")
        
        self.logger.info(f"准备 {task_name} 任务数据...")
        
        task_data = self.task_datasets[task_name].copy()
        
        # 移除无效的SMILES
        valid_mask = task_data['SMILES'].apply(
            lambda x: self.feature_extractor.smiles_to_mol(x) is not None
        )
        task_data = task_data[valid_mask].reset_index(drop=True)
        
        self.logger.info(f"{task_name} 有效数据: {len(task_data)} 样本")
        
        # 提取特征
        X, feature_names = self.extract_features_for_task(task_data)
        y = task_data[task_name].values
        
        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # 标准化特征
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 标准化目标（可选）
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
        
        dataset = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_train_scaled': y_train_scaled,
            'y_val_scaled': y_val_scaled,
            'feature_names': feature_names,
            'feature_scaler': scaler,
            'target_scaler': target_scaler,
            'task_name': task_name
        }
        
        self.logger.info(f"{task_name} 数据准备完成:")
        self.logger.info(f"  训练集: {X_train_scaled.shape}")
        self.logger.info(f"  验证集: {X_val_scaled.shape}")
        
        return dataset
    
    def prepare_test_features(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        为测试数据提取特征
        
        Args:
            test_df: 测试数据DataFrame
            
        Returns:
            测试集特征数组
        """
        self.logger.info("准备测试数据特征...")
        
        # 提取特征
        X_test, _ = self.extract_features_for_task(test_df)
        
        # 使用训练时的scaler进行标准化
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            self.logger.warning("没有找到特征scaler，使用原始特征")
            X_test_scaled = X_test.values
        
        self.logger.info(f"测试集特征: {X_test_scaled.shape}")
        
        return X_test_scaled
    
    def create_submission_dataframe(self, test_df: pd.DataFrame, 
                                   predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        创建提交格式的DataFrame
        
        Args:
            test_df: 测试数据
            predictions: 各任务的预测结果字典
            
        Returns:
            提交格式的DataFrame
        """
        submission_df = pd.DataFrame()
        submission_df['id'] = test_df['id']
        
        # 按指定顺序添加预测列
        target_order = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        for target in target_order:
            if target in predictions:
                submission_df[target] = predictions[target]
            else:
                # 如果没有预测，填充0
                submission_df[target] = 0.0
                self.logger.warning(f"任务 {target} 没有预测结果，填充为0")
        
        return submission_df
    
    def get_task_summary(self) -> Dict:
        """
        获取任务数据摘要
        
        Returns:
            任务摘要字典
        """
        summary = {}
        
        for task_name, task_data in self.task_datasets.items():
            summary[task_name] = {
                'sample_count': len(task_data),
                'target_mean': task_data[task_name].mean(),
                'target_std': task_data[task_name].std(),
                'target_min': task_data[task_name].min(),
                'target_max': task_data[task_name].max()
            }
        
        return summary