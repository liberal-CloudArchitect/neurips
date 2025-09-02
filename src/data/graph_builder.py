"""
图数据构建器 - 将SMILES转换为PyTorch Geometric图结构
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import torch_geometric
from torch_geometric.data import Data, Batch
import warnings
warnings.filterwarnings('ignore')


class MolecularGraphBuilder:
    """分子图构建器"""
    
    def __init__(self):
        """初始化图构建器"""
        # 原子特征维度
        self.atom_feature_dim = None
        self.bond_feature_dim = None
        
        # 原子类型映射 (扩展更多原子类型)
        self.atom_types = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 
            'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 
            'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]
        
        # 杂化类型
        self.hybridization_types = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ]
        
        # 键类型
        self.bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        
    def get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """
        提取原子特征
        
        Args:
            atom: RDKit原子对象
            
        Returns:
            原子特征向量
        """
        features = []
        
        # 1. 原子类型 (one-hot编码)
        atom_type = atom.GetSymbol()
        if atom_type not in self.atom_types:
            atom_type = 'Unknown'
        atom_type_features = [float(atom_type == x) for x in self.atom_types]
        features.extend(atom_type_features)
        
        # 2. 原子度 (连接的原子数量)
        degree = atom.GetDegree()
        degree_features = [float(degree == x) for x in range(6)]  # 0-5度
        features.extend(degree_features)
        
        # 3. 形式电荷
        formal_charge = atom.GetFormalCharge()
        charge_features = [float(formal_charge == x) for x in range(-2, 3)]  # -2到+2
        features.extend(charge_features)
        
        # 4. 杂化类型
        hybridization = atom.GetHybridization()
        hybrid_features = [float(hybridization == x) for x in self.hybridization_types]
        features.extend(hybrid_features)
        
        # 5. 是否为芳香原子
        features.append(float(atom.GetIsAromatic()))
        
        # 6. 氢原子数量
        num_hs = atom.GetTotalNumHs()
        h_features = [float(num_hs == x) for x in range(5)]  # 0-4个氢
        features.extend(h_features)
        
        # 7. 原子量 (标准化)
        atomic_mass = atom.GetMass()
        features.append(atomic_mass / 100.0)  # 简单标准化
        
        # 8. 原子价
        valence = atom.GetTotalValence()
        valence_features = [float(valence == x) for x in range(7)]  # 0-6价
        features.extend(valence_features)
        
        # 9. 是否在环中
        features.append(float(atom.IsInRing()))
        
        # 10. 环的大小 (如果在环中)
        ring_sizes = [3, 4, 5, 6, 7, 8]
        ring_features = []
        for size in ring_sizes:
            ring_features.append(float(atom.IsInRingSize(size)))
        features.extend(ring_features)
        
        return features
    
    def get_bond_features(self, bond: Chem.Bond) -> List[float]:
        """
        提取化学键特征
        
        Args:
            bond: RDKit化学键对象
            
        Returns:
            化学键特征向量
        """
        features = []
        
        # 1. 键类型 (one-hot编码)
        bond_type = bond.GetBondType()
        bond_type_features = [float(bond_type == x) for x in self.bond_types]
        features.extend(bond_type_features)
        
        # 2. 是否为芳香键
        features.append(float(bond.GetIsAromatic()))
        
        # 3. 是否在环中
        features.append(float(bond.IsInRing()))
        
        # 4. 是否为共轭键
        features.append(float(bond.GetIsConjugated()))
        
        # 5. 立体化学信息
        stereo = bond.GetStereo()
        stereo_features = [
            float(stereo == Chem.rdchem.BondStereo.STEREONONE),
            float(stereo == Chem.rdchem.BondStereo.STEREOANY),
            float(stereo == Chem.rdchem.BondStereo.STEREOZ),
            float(stereo == Chem.rdchem.BondStereo.STEREOE)
        ]
        features.extend(stereo_features)
        
        return features
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        将SMILES字符串转换为PyTorch Geometric图数据
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            PyTorch Geometric Data对象，如果转换失败返回None
        """
        try:
            # 解析SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 添加氢原子（可选）
            mol = Chem.AddHs(mol)
            
            # 提取原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                features = self.get_atom_features(atom)
                atom_features.append(features)
            
            if not atom_features:
                return None
                
            # 转换为tensor
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # 提取邻接信息和边特征
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # 获取边特征
                bond_feat = self.get_bond_features(bond)
                
                # 无向图：添加双向边
                edge_indices.extend([[i, j], [j, i]])
                edge_features.extend([bond_feat, bond_feat])
            
            # 转换为tensor
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                # 处理没有边的情况（单原子分子）
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(self.get_bond_features(mol.GetBonds()[0])) if mol.GetNumBonds() > 0 else 11), dtype=torch.float)
            
            # 计算全局分子特征（可选）
            global_features = self._get_global_features(mol)
            
            # 创建PyTorch Geometric数据对象
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=smiles,
                num_nodes=x.size(0),
                global_features=global_features
            )
            
            return data
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def _get_global_features(self, mol: Chem.Mol) -> torch.Tensor:
        """
        计算全局分子特征
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            全局特征tensor
        """
        features = []
        
        # 分子量
        features.append(Chem.Descriptors.MolWt(mol) / 1000.0)  # 标准化
        
        # 拓扑极性表面积
        features.append(Chem.Descriptors.TPSA(mol) / 200.0)  # 标准化
        
        # LogP
        features.append(Chem.Descriptors.MolLogP(mol) / 10.0)  # 标准化
        
        # 氢键供体和受体数量
        features.append(Chem.Descriptors.NumHDonors(mol) / 10.0)
        features.append(Chem.Descriptors.NumHAcceptors(mol) / 10.0)
        
        # 可旋转键数量
        features.append(Chem.Descriptors.NumRotatableBonds(mol) / 20.0)
        
        # 芳香环数量
        features.append(Chem.Descriptors.NumAromaticRings(mol) / 5.0)
        
        # 原子总数
        features.append(mol.GetNumAtoms() / 100.0)
        
        # 键总数
        features.append(mol.GetNumBonds() / 100.0)
        
        return torch.tensor(features, dtype=torch.float)
    
    def batch_smiles_to_graphs(self, smiles_list: List[str]) -> List[Data]:
        """
        批量将SMILES转换为图数据
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            图数据列表
        """
        graphs = []
        failed_count = 0
        
        for smiles in smiles_list:
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
            else:
                failed_count += 1
        
        if failed_count > 0:
            print(f"Warning: Failed to convert {failed_count}/{len(smiles_list)} SMILES to graphs")
        
        return graphs
    
    def get_feature_dimensions(self) -> Tuple[int, int, int]:
        """
        获取特征维度
        
        Returns:
            (原子特征维度, 边特征维度, 全局特征维度)
        """
        # 临时创建一个简单分子来计算特征维度
        test_mol = Chem.MolFromSmiles("C")
        if test_mol is None:
            return 0, 0, 0
            
        # 原子特征维度
        atom_features = self.get_atom_features(test_mol.GetAtomWithIdx(0))
        atom_feature_dim = len(atom_features)
        
        # 边特征维度（如果有边的话）
        test_mol_with_bond = Chem.MolFromSmiles("CC")
        if test_mol_with_bond and test_mol_with_bond.GetNumBonds() > 0:
            bond_features = self.get_bond_features(test_mol_with_bond.GetBondWithIdx(0))
            bond_feature_dim = len(bond_features)
        else:
            bond_feature_dim = 11  # 默认边特征维度
        
        # 全局特征维度
        global_features = self._get_global_features(test_mol)
        global_feature_dim = len(global_features)
        
        return atom_feature_dim, bond_feature_dim, global_feature_dim


def collate_graphs(batch: List[Data]) -> Batch:
    """
    批量图数据的collate函数
    
    Args:
        batch: 图数据列表
        
    Returns:
        批量图数据
    """
    return Batch.from_data_list(batch)