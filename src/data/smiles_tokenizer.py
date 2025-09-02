"""
SMILES Tokenizer - 化学SMILES字符串的专业分词器
"""

import re
import torch
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class SMILESTokenizer:
    """SMILES字符串的专业分词器"""
    
    def __init__(self, 
                 vocab_size: int = 1000,
                 max_length: int = 512,
                 pad_token: str = '<pad>',
                 unk_token: str = '<unk>',
                 bos_token: str = '<bos>',
                 eos_token: str = '<eos>',
                 mask_token: str = '<mask>'):
        """
        初始化SMILES分词器
        
        Args:
            vocab_size: 词汇表大小
            max_length: 最大序列长度
            pad_token: 填充token
            unk_token: 未知token
            bos_token: 开始token
            eos_token: 结束token
            mask_token: 掩码token (用于MLM)
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # 特殊token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        
        # 构建基础词汇表
        self._build_base_vocab()
        
        # 词汇表和逆映射
        self.vocab = {}
        self.id_to_token = {}
        self.is_trained = False
        
    def _build_base_vocab(self):
        """构建基础化学token集合"""
        # 特殊tokens
        self.special_tokens = [
            self.pad_token, self.unk_token, self.bos_token, 
            self.eos_token, self.mask_token
        ]
        
        # 原子符号 (最常见的化学元素)
        self.atom_tokens = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'H', 'B', 'Si', 'Li', 'Na', 'Mg', 'Al', 'K', 'Ca',
            'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'As', 'Se', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Pt',
            'Au', 'Hg', 'Pb', 'Bi'
        ]
        
        # 方括号内的原子 (带电荷、氢数等)
        self.bracketed_atoms = [
            '[C]', '[N]', '[O]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]',
            '[H]', '[B]', '[Si]', '[c]', '[n]', '[o]', '[s]', '[p]', '[nH]',
            '[NH]', '[NH2]', '[NH3+]', '[O-]', '[S-]', '[N+]', '[n+]',
            '[C+]', '[C-]', '[c+]', '[c-]', '[se]', '[Se]', '[te]', '[Te]'
        ]
        
        # 键类型
        self.bond_tokens = ['-', '=', '#', ':', '/', '\\', '.']
        
        # 环标记
        self.ring_tokens = [str(i) for i in range(10)]  # 0-9
        self.ring_tokens.extend(['%10', '%11', '%12', '%13', '%14', '%15'])
        
        # 分支符号
        self.branch_tokens = ['(', ')']
        
        # 芳香性原子
        self.aromatic_tokens = ['c', 'n', 'o', 's', 'p', 'b', 'se', 'te']
        
        # 立体化学
        self.stereo_tokens = ['@', '@@', '@TH1', '@TH2', '@AL1', '@AL2', '@SP1', '@SP2', '@SP3']
        
        # 常见的多字符token
        self.multi_char_tokens = [
            'Cl', 'Br', 'Si', 'Li', 'Na', 'Mg', 'Al', 'Ca', 'Ti', 'Cr',
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Ag', 'Cd',
            'In', 'Sn', 'Sb', 'Te', 'Pt', 'Au', 'Hg', 'Pb', 'Bi'
        ]
    
    def _smiles_tokenize(self, smiles: str) -> List[str]:
        """
        将SMILES字符串分解为token
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            token列表
        """
        tokens = []
        i = 0
        
        while i < len(smiles):
            # 跳过空格
            if smiles[i].isspace():
                i += 1
                continue
            
            # 处理方括号内的原子
            if smiles[i] == '[':
                j = i + 1
                while j < len(smiles) and smiles[j] != ']':
                    j += 1
                if j < len(smiles):
                    tokens.append(smiles[i:j+1])
                    i = j + 1
                else:
                    tokens.append(smiles[i])
                    i += 1
                continue
            
            # 处理环标记 (%10, %11等)
            if smiles[i] == '%' and i + 2 < len(smiles) and smiles[i+1:i+3].isdigit():
                tokens.append(smiles[i:i+3])
                i += 3
                continue
            
            # 处理立体化学标记
            if smiles[i] == '@':
                # 查找完整的立体化学标记
                for stereo in self.stereo_tokens:
                    if smiles[i:].startswith(stereo):
                        tokens.append(stereo)
                        i += len(stereo)
                        break
                else:
                    tokens.append('@')
                    i += 1
                continue
            
            # 处理多字符原子 (Cl, Br等)
            found_multi_char = False
            for multi_token in self.multi_char_tokens:
                if smiles[i:].startswith(multi_token):
                    tokens.append(multi_token)
                    i += len(multi_token)
                    found_multi_char = True
                    break
            
            if found_multi_char:
                continue
            
            # 处理单字符token
            tokens.append(smiles[i])
            i += 1
        
        return tokens
    
    def train_from_corpus(self, smiles_list: List[str]):
        """
        从SMILES语料库训练分词器
        
        Args:
            smiles_list: SMILES字符串列表
        """
        # 收集所有token
        all_tokens = []
        for smiles in smiles_list:
            tokens = self._smiles_tokenize(smiles)
            all_tokens.extend(tokens)
        
        # 统计频率
        token_counts = Counter(all_tokens)
        
        # 构建词汇表：特殊token + 高频token
        vocab_tokens = self.special_tokens.copy()
        
        # 添加基础化学token (确保重要的化学符号被包含)
        base_chemical_tokens = (
            self.atom_tokens + self.aromatic_tokens + self.bond_tokens + 
            self.ring_tokens + self.branch_tokens + self.multi_char_tokens
        )
        
        for token in base_chemical_tokens:
            if token not in vocab_tokens:
                vocab_tokens.append(token)
        
        # 添加高频token直到达到vocab_size
        remaining_slots = self.vocab_size - len(vocab_tokens)
        most_common = token_counts.most_common()
        
        for token, count in most_common:
            if token not in vocab_tokens and len(vocab_tokens) < self.vocab_size:
                vocab_tokens.append(token)
            if len(vocab_tokens) >= self.vocab_size:
                break
        
        # 构建映射
        self.vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        self.is_trained = True
        print(f"训练完成，词汇表大小: {len(self.vocab)}")
        print(f"最常见的20个token: {[token for token, _ in token_counts.most_common(20)]}")
    
    def encode(self, 
               smiles: str, 
               add_special_tokens: bool = True,
               padding: bool = True,
               truncation: bool = True,
               return_tensors: Optional[str] = None) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        编码SMILES字符串
        
        Args:
            smiles: SMILES字符串
            add_special_tokens: 是否添加特殊token
            padding: 是否填充到max_length
            truncation: 是否截断到max_length
            return_tensors: 返回张量类型 ('pt'为PyTorch)
            
        Returns:
            编码结果字典
        """
        if not self.is_trained:
            raise ValueError("分词器未训练，请先调用train_from_corpus")
        
        # 分词
        tokens = self._smiles_tokenize(smiles)
        
        # 添加特殊token
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # 转换为ID
        input_ids = []
        for token in tokens:
            if token in self.vocab:
                input_ids.append(self.vocab[token])
            else:
                input_ids.append(self.vocab[self.unk_token])
        
        # 截断
        if truncation and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            if add_special_tokens:
                input_ids[-1] = self.vocab[self.eos_token]  # 确保最后是EOS token
        
        # 生成attention mask
        attention_mask = [1] * len(input_ids)
        
        # 填充
        if padding and len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([self.vocab[self.pad_token]] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens[:self.max_length] if truncation else tokens
        }
        
        # 转换为张量
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'])
            result['attention_mask'] = torch.tensor(result['attention_mask'])
        
        return result
    
    def decode(self, input_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        解码ID序列为SMILES字符串
        
        Args:
            input_ids: ID序列
            skip_special_tokens: 是否跳过特殊token
            
        Returns:
            SMILES字符串
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        tokens = []
        for id in input_ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return ''.join(tokens)
    
    def batch_encode(self, 
                    smiles_list: List[str],
                    add_special_tokens: bool = True,
                    padding: bool = True,
                    truncation: bool = True,
                    return_tensors: Optional[str] = None) -> Dict[str, Union[List, torch.Tensor]]:
        """
        批量编码SMILES字符串
        
        Args:
            smiles_list: SMILES字符串列表
            其他参数同encode方法
            
        Returns:
            批量编码结果
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_tokens = []
        
        for smiles in smiles_list:
            encoded = self.encode(
                smiles, 
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                return_tensors=None
            )
            batch_input_ids.append(encoded['input_ids'])
            batch_attention_mask.append(encoded['attention_mask'])
            batch_tokens.append(encoded['tokens'])
        
        result = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'tokens': batch_tokens
        }
        
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'])
            result['attention_mask'] = torch.tensor(result['attention_mask'])
        
        return result
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocab) if self.is_trained else 0
    
    def get_special_tokens_dict(self) -> Dict[str, int]:
        """获取特殊token的ID映射"""
        if not self.is_trained:
            return {}
        
        return {
            'pad_token_id': self.vocab.get(self.pad_token, 0),
            'unk_token_id': self.vocab.get(self.unk_token, 1),
            'bos_token_id': self.vocab.get(self.bos_token, 2),
            'eos_token_id': self.vocab.get(self.eos_token, 3),
            'mask_token_id': self.vocab.get(self.mask_token, 4)
        }
    
    def save_vocab(self, vocab_path: str):
        """保存词汇表"""
        import json
        if not self.is_trained:
            raise ValueError("分词器未训练，无法保存")
        
        vocab_data = {
            'vocab': self.vocab,
            'config': {
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token,
                'mask_token': self.mask_token
            }
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"词汇表已保存到: {vocab_path}")
    
    def load_vocab(self, vocab_path: str):
        """加载词汇表"""
        import json
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.id_to_token = {int(idx): token for token, idx in self.vocab.items()}
        
        # 更新配置
        config = vocab_data['config']
        self.vocab_size = config['vocab_size']
        self.max_length = config['max_length']
        self.pad_token = config['pad_token']
        self.unk_token = config['unk_token']
        self.bos_token = config['bos_token']
        self.eos_token = config['eos_token']
        self.mask_token = config['mask_token']
        
        self.is_trained = True
        print(f"词汇表已从 {vocab_path} 加载，大小: {len(self.vocab)}")


def create_smiles_tokenizer(smiles_list: List[str],
                           vocab_size: int = 1000,
                           max_length: int = 512) -> SMILESTokenizer:
    """
    创建并训练SMILES分词器的便捷函数
    
    Args:
        smiles_list: 训练用的SMILES字符串列表
        vocab_size: 词汇表大小
        max_length: 最大序列长度
        
    Returns:
        训练好的分词器
    """
    tokenizer = SMILESTokenizer(vocab_size=vocab_size, max_length=max_length)
    tokenizer.train_from_corpus(smiles_list)
    return tokenizer


# 示例用法和测试函数
def test_smiles_tokenizer():
    """测试SMILES分词器"""
    # 示例SMILES
    test_smiles = [
        "CCO",  # 乙醇
        "CC(C)O",  # 异丙醇
        "c1ccccc1",  # 苯
        "CC(=O)NC1=CC=C(C=C1)O",  # 对乙酰氨基酚
        "CN1CCC[C@H]1c2cccnc2",  # 尼古丁
        "CC(C)(C)c1ccc(cc1)O",  # 对叔丁基苯酚
    ]
    
    # 创建和训练分词器
    tokenizer = create_smiles_tokenizer(test_smiles, vocab_size=200, max_length=128)
    
    # 测试编码和解码
    for smiles in test_smiles[:3]:
        encoded = tokenizer.encode(smiles, return_tensors='pt')
        decoded = tokenizer.decode(encoded['input_ids'])
        
        print(f"原始: {smiles}")
        print(f"编码: {encoded['input_ids'][:20]}...")  # 只显示前20个token
        print(f"解码: {decoded}")
        print(f"Token数: {len(encoded['tokens'])}")
        print("-" * 50)
    
    return tokenizer


if __name__ == "__main__":
    test_smiles_tokenizer()