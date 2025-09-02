#!/usr/bin/env python
"""
Transformer模块设置测试脚本

测试第四阶段Transformer开发的关键组件：
1. SMILES分词器功能
2. Transformer模型创建和前向传播
3. 配置加载
4. 基础训练流程

使用方法:
    python test_transformer_setup.py
"""

import sys
import warnings
from pathlib import Path
import torch
import numpy as np
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

# 测试颜色输出
def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_warning(message):
    print(f"⚠️ {message}")

def print_info(message):
    print(f"🔍 {message}")


def test_library_imports():
    """测试库导入"""
    print_info("测试库导入...")
    
    try:
        # 核心库
        import torch
        print_success(f"PyTorch: {torch.__version__}")
        
        # Transformers库（如果有的话）
        try:
            import transformers
            print_success(f"Transformers: {transformers.__version__}")
        except ImportError:
            print_warning("Transformers库未安装，将使用自定义实现")
        
        # 化学库
        try:
            import rdkit
            print_success("RDKit: 导入成功")
        except ImportError:
            print_error("RDKit未安装，这是必需的库")
            return False
        
        # 数据处理库
        import numpy
        import pandas
        print_success("NumPy和Pandas: 导入成功")
        
        return True
        
    except Exception as e:
        print_error(f"库导入失败: {e}")
        return False


def test_smiles_tokenizer():
    """测试SMILES分词器"""
    print_info("测试SMILES分词器...")
    
    try:
        from src.data.smiles_tokenizer import SMILESTokenizer, create_smiles_tokenizer
        
        # 测试SMILES
        test_smiles = [
            "CCO",  # 乙醇
            "CC(C)O",  # 异丙醇
            "c1ccccc1",  # 苯
            "CC(=O)NC1=CC=C(C=C1)O",  # 对乙酰氨基酚
            "CN1CCC[C@H]1c2cccnc2",  # 尼古丁
            "Invalid_SMILES"  # 无效SMILES
        ]
        
        print(f"  测试SMILES: {test_smiles[:4]}...")
        
        # 创建分词器
        tokenizer = create_smiles_tokenizer(test_smiles, vocab_size=200, max_length=128)
        
        print(f"  词汇表大小: {tokenizer.get_vocab_size()}")
        print(f"  最大序列长度: {tokenizer.max_length}")
        
        # 测试编码和解码
        test_smiles_sample = "CCO"
        encoded = tokenizer.encode(test_smiles_sample, return_tensors='pt')
        decoded = tokenizer.decode(encoded['input_ids'])
        
        print(f"  测试编码/解码:")
        print(f"    原始: {test_smiles_sample}")
        print(f"    解码: {decoded}")
        print(f"    Token数: {len(encoded['tokens'])}")
        print(f"    输入维度: {encoded['input_ids'].shape}")
        print(f"    注意力掩码维度: {encoded['attention_mask'].shape}")
        
        # 测试批量处理
        batch_encoded = tokenizer.batch_encode(test_smiles[:3], return_tensors='pt')
        print(f"  批量编码维度: {batch_encoded['input_ids'].shape}")
        
        # 测试特殊token
        special_tokens = tokenizer.get_special_tokens_dict()
        print(f"  特殊token: {special_tokens}")
        
        print_success("SMILES分词器测试通过")
        return tokenizer
        
    except Exception as e:
        print_error(f"SMILES分词器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_transformer_models():
    """测试Transformer模型"""
    print_info("测试Transformer模型...")
    
    try:
        from src.models.transformer import (
            create_transformer_model, 
            count_transformer_parameters,
            MultiTaskTransformerPredictor,
            BertBasedSMILESPredictor
        )
        
        # 模型配置
        vocab_size = 200
        config = {
            'd_model': 128,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'max_seq_length': 64,
            'dropout': 0.1,
            'pooling_strategy': 'cls',
            'task_names': ['Density', 'Tc']
        }
        
        # 测试自定义Transformer
        print("  测试自定义Transformer模型...")
        custom_model = create_transformer_model('custom', vocab_size, config)
        param_count = count_transformer_parameters(custom_model)
        print(f"    自定义模型参数数量: {param_count:,}")
        
        # 创建测试数据
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # 前向传播测试
        custom_model.eval()
        with torch.no_grad():
            outputs = custom_model(input_ids, attention_mask)
            print(f"    自定义模型输出形状: {list(outputs['predictions'].keys())}")
            for task, pred in outputs['predictions'].items():
                print(f"      {task}: {pred.shape}")
        
        # 测试BERT-based模型（可能会失败，因为没有预训练模型）
        print("  测试BERT-based模型...")
        try:
            bert_config = config.copy()
            bert_config['pretrained_model_name'] = 'DeepChem/ChemBERTa-77M-MLM'
            bert_model = create_transformer_model('bert_based', vocab_size, bert_config)
            print(f"    BERT模型参数数量: {count_transformer_parameters(bert_model):,}")
            print_success("BERT-based模型创建成功")
        except Exception as e:
            print_warning(f"BERT-based模型创建失败（预期的）: {e}")
        
        print_success("Transformer模型测试通过")
        return custom_model
        
    except Exception as e:
        print_error(f"Transformer模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_config_loading():
    """测试配置加载"""
    print_info("测试配置加载...")
    
    try:
        from src.utils.config import load_config
        
        config_path = "configs/config.yaml"
        config = load_config(config_path)
        
        # 检查Transformer配置
        if 'transformer' in config.get('models', {}):
            transformer_config = config['models']['transformer']
            print(f"  Transformer配置加载成功:")
            print(f"    模型维度: {transformer_config.get('d_model', 'N/A')}")
            print(f"    注意力头数: {transformer_config.get('nhead', 'N/A')}")
            print(f"    层数: {transformer_config.get('num_layers', 'N/A')}")
            print(f"    词汇表大小: {transformer_config.get('vocab_size', 'N/A')}")
            print(f"    最大序列长度: {transformer_config.get('max_seq_length', 'N/A')}")
            print(f"    预训练模型: {transformer_config.get('pretrained_model_name', 'N/A')}")
        else:
            print_warning("配置文件中未找到Transformer配置")
        
        print_success("配置测试通过")
        return config
        
    except Exception as e:
        print_error(f"配置测试失败: {e}")
        return None


def test_training_integration():
    """测试训练集成"""
    print_info("测试训练集成...")
    
    try:
        from src.models.transformer_trainer import TransformerTrainer, SMILESDataset, collate_fn
        from torch.utils.data import DataLoader
        
        # 模拟配置
        config = {
            'models': {
                'transformer': {
                    'd_model': 64,
                    'nhead': 4,
                    'num_layers': 2,
                    'vocab_size': 100,
                    'max_seq_length': 32,
                    'task_names': ['Density']
                }
            },
            'training': {
                'device': 'cpu',
                'batch_size': 2,
                'epochs': 2,
                'learning_rate': 1e-3
            },
            'data': {
                'validation': {
                    'n_splits': 2,
                    'random_state': 42
                }
            }
        }
        
        # 创建训练器
        trainer = TransformerTrainer(config)
        
        # 测试数据准备
        test_smiles = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O"]
        test_targets = np.array([1.0, 2.0, 3.0, 4.0])
        
        # 准备分词器
        tokenizer = trainer.prepare_tokenizer(test_smiles)
        print(f"    分词器词汇表大小: {tokenizer.get_vocab_size()}")
        
        # 测试数据集
        dataset = SMILESDataset(
            smiles_list=test_smiles,
            targets=test_targets.reshape(-1, 1),
            tokenizer=tokenizer,
            target_names=['Density']
        )
        
        # 测试数据加载器
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        # 测试一个批次
        for batch in dataloader:
            print(f"    批次键: {list(batch.keys())}")
            print(f"    输入维度: {batch['input_ids'].shape}")
            print(f"    目标维度: {batch['targets']['Density'].shape}")
            break
        
        print_success("训练集成测试通过")
        return trainer
        
    except Exception as e:
        print_error(f"训练集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_end_to_end_mini_training():
    """测试端到端的迷你训练"""
    print_info("测试端到端迷你训练...")
    
    try:
        from src.models.transformer_trainer import TransformerTrainer
        
        # 简化配置
        config = {
            'models': {
                'transformer': {
                    'd_model': 64,
                    'nhead': 4,
                    'num_layers': 2,
                    'vocab_size': 100,
                    'max_seq_length': 32,
                    'task_names': ['test_task']
                }
            },
            'training': {
                'device': 'cpu',
                'batch_size': 2,
                'epochs': 2,
                'learning_rate': 1e-3,
                'weight_decay': 1e-5,
                'scheduler_patience': 1,
                'early_stopping': {'patience': 1},
                'loss': {'type': 'mse'}
            },
            'data': {
                'validation': {
                    'n_splits': 2,
                    'random_state': 42
                }
            }
        }
        
        # 创建模拟数据
        np.random.seed(42)
        train_smiles = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O"] * 2  # 8个样本
        train_targets = np.random.normal(0, 1, len(train_smiles))
        val_smiles = ["CCN", "CC(C)N"]
        val_targets = np.random.normal(0, 1, len(val_smiles))
        
        # 创建训练器
        trainer = TransformerTrainer(config)
        
        # 准备分词器
        all_smiles = train_smiles + val_smiles
        trainer.prepare_tokenizer(all_smiles)
        
        # 尝试单个epoch的训练
        print("    开始迷你训练...")
        model, history = trainer.train_single_task_model(
            train_smiles=train_smiles,
            train_targets=train_targets,
            val_smiles=val_smiles,
            val_targets=val_targets,
            task_name='test_task',
            model_type='custom'
        )
        
        print(f"    训练完成，历史长度: {len(history['train_loss'])}")
        print(f"    最终训练损失: {history['train_loss'][-1]:.6f}")
        print(f"    最终验证MAE: {history['val_mae'][-1]:.6f}")
        
        # 测试预测
        test_smiles = ["CCO", "CCN"]
        predictions = trainer.predict(model, test_smiles, 'test_task')
        print(f"    测试预测: {predictions}")
        
        print_success("端到端迷你训练测试通过")
        return True
        
    except Exception as e:
        print_error(f"端到端训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始Transformer模块测试")
    print("=" * 50)
    
    # 测试结果统计
    test_results = {
        '库导入': False,
        'SMILES分词器': False,
        'Transformer模型': False,
        '配置加载': False,
        '训练集成': False,
        '端到端训练': False
    }
    
    # 1. 测试库导入
    test_results['库导入'] = test_library_imports()
    print()
    
    if not test_results['库导入']:
        print_error("基础库导入失败，跳过后续测试")
        return
    
    # 2. 测试SMILES分词器
    tokenizer = test_smiles_tokenizer()
    test_results['SMILES分词器'] = tokenizer is not None
    print()
    
    # 3. 测试Transformer模型
    model = test_transformer_models()
    test_results['Transformer模型'] = model is not None
    print()
    
    # 4. 测试配置加载
    config = test_config_loading()
    test_results['配置加载'] = config is not None
    print()
    
    # 5. 测试训练集成
    trainer = test_training_integration()
    test_results['训练集成'] = trainer is not None
    print()
    
    # 6. 测试端到端训练（如果前面都通过）
    if all([test_results['SMILES分词器'], test_results['Transformer模型']]):
        test_results['端到端训练'] = test_end_to_end_mini_training()
    else:
        print_warning("跳过端到端训练测试（依赖组件测试失败）")
    print()
    
    # 测试总结
    print("=" * 50)
    print("🏁 测试总结")
    print("=" * 50)
    
    for test_name, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    passed_count = sum(test_results.values())
    total_count = len(test_results)
    
    print(f"\n总体结果: {passed_count}/{total_count} 测试通过")
    
    if passed_count == total_count:
        print("🎉 所有测试通过！Transformer模块准备就绪")
        print("\n📋 下一步：")
        print("1. 运行快速测试: python src/experiments/transformer_experiment.py --test_tokenizer")
        print("2. 运行小规模训练: python src/experiments/transformer_experiment.py --max_samples 50")
        print("3. 运行完整训练: python src/experiments/transformer_experiment.py")
        print("4. 模型对比: python src/experiments/transformer_experiment.py --compare")
    else:
        print("⚠️ 存在失败的测试，请检查并解决问题")
        print("\n🔧 常见问题解决:")
        print("1. 如果Transformers库导入失败:")
        print("   pip install transformers")
        print("2. 如果RDKit导入失败:")
        print("   pip install rdkit-pypi")
        print("3. 如果模型创建失败:")
        print("   检查PyTorch版本兼容性")
        print("4. 如果内存不足:")
        print("   减小模型尺寸或批次大小")


if __name__ == "__main__":
    main()