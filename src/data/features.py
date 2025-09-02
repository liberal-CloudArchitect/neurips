"""
分子特征提取器
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class MolecularFeatureExtractor:
    """分子特征提取器"""
    
    def __init__(self, 
                 morgan_radius: int = 2,
                 morgan_n_bits: int = 2048,
                 use_features: bool = True,
                 use_chirality: bool = True,
                 use_maccs: bool = True):
        """
        初始化特征提取器
        
        Args:
            morgan_radius: Morgan指纹半径
            morgan_n_bits: Morgan指纹位数
            use_features: 是否使用特征信息
            use_chirality: 是否考虑手性
            use_maccs: 是否使用MACCS分子密钥
        """
        self.morgan_radius = morgan_radius
        self.morgan_n_bits = morgan_n_bits
        self.use_features = use_features
        self.use_chirality = use_chirality
        self.use_maccs = use_maccs
        
        # 预定义的RDKit描述符列表
        self.descriptor_functions = {
            'MolWt': Descriptors.MolWt,
            'TPSA': Descriptors.TPSA,
            'LogP': Descriptors.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'FractionCsp3': Descriptors.FractionCSP3,
            'NumHeteroatoms': Descriptors.NumHeteroatoms,
            'RingCount': Descriptors.RingCount,
            'BertzCT': Descriptors.BertzCT,
            'Chi0v': Descriptors.Chi0v,
            'Chi1v': Descriptors.Chi1v,
            'Chi2v': Descriptors.Chi2v,
            'HallKierAlpha': Descriptors.HallKierAlpha,
            'Kappa1': Descriptors.Kappa1,
            'Kappa2': Descriptors.Kappa2,
            'Kappa3': Descriptors.Kappa3
        }
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        将SMILES字符串转换为RDKit分子对象
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            RDKit分子对象，如果转换失败则返回None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
            return mol
        except:
            return None
    
    def get_morgan_fingerprint(self, smiles: str) -> np.ndarray:
        """
        提取Morgan指纹
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            Morgan指纹向量
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return np.zeros(self.morgan_n_bits)
        
        try:
            fingerprint = GetMorganFingerprintAsBitVect(
                mol, 
                radius=self.morgan_radius,
                nBits=self.morgan_n_bits,
                useFeatures=self.use_features,
                useChirality=self.use_chirality
            )
            return np.array(fingerprint)
        except:
            return np.zeros(self.morgan_n_bits)
    
    def get_maccs_keys(self, smiles: str) -> np.ndarray:
        """
        提取MACCS分子密钥
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            MACCS Keys向量 (167位)
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return np.zeros(167)  # MACCS Keys固定167位
        
        try:
            maccs_keys = MACCSkeys.GenMACCSKeys(mol)
            return np.array(maccs_keys)
        except:
            return np.zeros(167)
    
    def get_rdkit_descriptors(self, smiles: str, 
                            descriptor_names: List[str] = None) -> Dict[str, float]:
        """
        提取RDKit描述符
        
        Args:
            smiles: SMILES字符串
            descriptor_names: 需要计算的描述符名称列表
            
        Returns:
            描述符字典
        """
        if descriptor_names is None:
            descriptor_names = list(self.descriptor_functions.keys())
        
        mol = self.smiles_to_mol(smiles)
        descriptors = {}
        
        for name in descriptor_names:
            if name in self.descriptor_functions:
                try:
                    if mol is not None:
                        value = self.descriptor_functions[name](mol)
                        # 处理NaN和inf值
                        if np.isnan(value) or np.isinf(value):
                            value = 0.0
                        descriptors[name] = float(value)
                    else:
                        descriptors[name] = 0.0
                except:
                    descriptors[name] = 0.0
            else:
                descriptors[name] = 0.0
        
        return descriptors
    
    def extract_features(self, smiles_list: List[str],
                        use_morgan: bool = True,
                        use_maccs: bool = None,
                        use_descriptors: bool = True,
                        descriptor_names: List[str] = None) -> pd.DataFrame:
        """
        批量提取分子特征
        
        Args:
            smiles_list: SMILES字符串列表
            use_morgan: 是否使用Morgan指纹
            use_maccs: 是否使用MACCS Keys（None时使用初始化参数）
            use_descriptors: 是否使用RDKit描述符
            descriptor_names: 描述符名称列表
            
        Returns:
            特征DataFrame
        """
        # 如果没有指定use_maccs，使用初始化时的设置
        if use_maccs is None:
            use_maccs = self.use_maccs
            
        features_list = []
        
        for smiles in smiles_list:
            features = {}
            
            # Morgan指纹
            if use_morgan:
                morgan_fp = self.get_morgan_fingerprint(smiles)
                morgan_features = {f'morgan_{i}': int(bit) for i, bit in enumerate(morgan_fp)}
                features.update(morgan_features)
            
            # MACCS分子密钥
            if use_maccs:
                maccs_fp = self.get_maccs_keys(smiles)
                maccs_features = {f'maccs_{i}': int(bit) for i, bit in enumerate(maccs_fp)}
                features.update(maccs_features)
            
            # RDKit描述符
            if use_descriptors:
                descriptors = self.get_rdkit_descriptors(smiles, descriptor_names)
                features.update(descriptors)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def get_feature_names(self, 
                         use_morgan: bool = True,
                         use_maccs: bool = None,
                         use_descriptors: bool = True,
                         descriptor_names: List[str] = None) -> List[str]:
        """
        获取特征名称列表
        
        Args:
            use_morgan: 是否使用Morgan指纹
            use_maccs: 是否使用MACCS Keys（None时使用初始化参数）
            use_descriptors: 是否使用RDKit描述符
            descriptor_names: 描述符名称列表
            
        Returns:
            特征名称列表
        """
        # 如果没有指定use_maccs，使用初始化时的设置
        if use_maccs is None:
            use_maccs = self.use_maccs
            
        feature_names = []
        
        if use_morgan:
            morgan_names = [f'morgan_{i}' for i in range(self.morgan_n_bits)]
            feature_names.extend(morgan_names)
        
        if use_maccs:
            maccs_names = [f'maccs_{i}' for i in range(167)]
            feature_names.extend(maccs_names)
        
        if use_descriptors:
            if descriptor_names is None:
                descriptor_names = list(self.descriptor_functions.keys())
            feature_names.extend(descriptor_names)
        
        return feature_names
    
    def validate_smiles(self, smiles_list: List[str]) -> Dict[str, List]:
        """
        验证SMILES字符串的有效性
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            包含有效和无效SMILES的字典
        """
        valid_smiles = []
        invalid_smiles = []
        
        for smiles in smiles_list:
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
            else:
                invalid_smiles.append(smiles)
        
        return {
            'valid': valid_smiles,
            'invalid': invalid_smiles,
            'valid_count': len(valid_smiles),
            'invalid_count': len(invalid_smiles),
            'valid_ratio': len(valid_smiles) / len(smiles_list) if smiles_list else 0
        }