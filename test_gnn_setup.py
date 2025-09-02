"""
测试GNN模块设置的脚本
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_imports():
    """测试必要的库是否能正常导入"""
    print("🔍 测试库导入...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import torch_geometric
        print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
        
        from rdkit import Chem
        print("✅ RDKit: 导入成功")
        
        import numpy as np
        import pandas as pd
        print("✅ NumPy和Pandas: 导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_graph_builder():
    """测试图构建器"""
    print("\n🔧 测试图构建器...")
    
    try:
        from src.data.graph_builder import MolecularGraphBuilder
        
        builder = MolecularGraphBuilder()
        
        # 测试简单分子
        test_smiles = [
            "CCO",  # 乙醇
            "CC(C)O",  # 异丙醇  
            "c1ccccc1",  # 苯
            "Invalid"  # 无效SMILES
        ]
        
        print(f"  测试SMILES: {test_smiles}")
        
        graphs = builder.batch_smiles_to_graphs(test_smiles)
        print(f"  成功构建图数量: {len(graphs)}/{len(test_smiles)}")
        
        if graphs:
            # 检查第一个图的特征
            graph = graphs[0]
            print(f"  第一个图特征:")
            print(f"    节点数: {graph.num_nodes}")
            print(f"    边数: {graph.edge_index.size(1)}")
            print(f"    节点特征维度: {graph.x.size(1)}")
            print(f"    边特征维度: {graph.edge_attr.size(1)}")
        
        # 测试特征维度
        atom_dim, edge_dim, global_dim = builder.get_feature_dimensions()
        print(f"  特征维度 - 原子: {atom_dim}, 边: {edge_dim}, 全局: {global_dim}")
        
        print("✅ 图构建器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 图构建器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gnn_models():
    """测试GNN模型"""
    print("\n🧠 测试GNN模型...")
    
    try:
        import torch
        from src.models.gnn import create_gnn_model, count_parameters
        from src.data.graph_builder import MolecularGraphBuilder
        from torch_geometric.data import Batch
        
        # 创建测试数据
        builder = MolecularGraphBuilder()
        test_smiles = ["CCO", "CC(C)O"]
        graphs = builder.batch_smiles_to_graphs(test_smiles)
        
        if not graphs:
            print("❌ 无法创建测试图数据")
            return False
        
        # 创建批次
        batch = Batch.from_data_list(graphs)
        
        # 测试GAT模型
        print("  测试GAT模型...")
        atom_dim, edge_dim, global_dim = builder.get_feature_dimensions()
        
        gat_config = {
            'hidden_dim': 64,  # 减小维度用于测试
            'num_layers': 2,
            'dropout': 0.1,
            'pool_type': 'mean',
            'num_heads': 4,
            'task_names': ['test_task']
        }
        
        gat_model = create_gnn_model(
            model_type='gat',
            atom_feature_dim=atom_dim,
            edge_feature_dim=edge_dim, 
            global_feature_dim=global_dim,
            config=gat_config
        )
        
        print(f"    GAT参数数量: {count_parameters(gat_model):,}")
        
        # 前向传播测试
        gat_model.eval()
        with torch.no_grad():
            gat_output = gat_model(batch)
            print(f"    GAT输出形状: {gat_output['test_task'].shape}")
        
        # 测试MPNN模型
        print("  测试MPNN模型...")
        mpnn_config = gat_config.copy()
        
        mpnn_model = create_gnn_model(
            model_type='mpnn',
            atom_feature_dim=atom_dim,
            edge_feature_dim=edge_dim,
            global_feature_dim=global_dim,
            config=mpnn_config
        )
        
        print(f"    MPNN参数数量: {count_parameters(mpnn_model):,}")
        
        mpnn_model.eval()
        with torch.no_grad():
            mpnn_output = mpnn_model(batch)
            print(f"    MPNN输出形状: {mpnn_output['test_task'].shape}")
        
        print("✅ GNN模型测试通过")
        return True
        
    except Exception as e:
        print(f"❌ GNN模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置加载"""
    print("\n⚙️ 测试配置加载...")
    
    try:
        from src.utils.config import load_config
        
        config = load_config("configs/config.yaml")
        
        # 检查GNN配置
        if 'gnn' in config.get('models', {}):
            gnn_config = config['models']['gnn']
            print(f"  GNN配置加载成功:")
            print(f"    隐藏维度: {gnn_config.get('hidden_dim')}")
            print(f"    层数: {gnn_config.get('num_layers')}")
            print(f"    池化类型: {gnn_config.get('pool_type')}")
            print("✅ 配置测试通过")
            return True
        else:
            print("❌ 未找到GNN配置")
            return False
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始GNN模块测试")
    print("=" * 50)
    
    tests = [
        ("库导入", test_imports),
        ("图构建器", test_graph_builder), 
        ("GNN模型", test_gnn_models),
        ("配置加载", test_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("🏁 测试总结")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！GNN模块准备就绪")
        print("\n📋 下一步：")
        print("1. 运行快速测试: python src/experiments/gnn_experiment.py --max_samples 100")
        print("2. 运行完整训练: python src/experiments/gnn_experiment.py")
        print("3. 模型对比: python src/experiments/gnn_experiment.py --compare")
    else:
        print("⚠️ 存在失败的测试，请检查并解决问题")
        
        # 提供一些常见问题的解决建议
        print("\n🔧 常见问题解决:")
        print("1. 如果PyTorch Geometric导入失败:")
        print("   pip install torch-geometric torch-scatter torch-sparse torch-cluster")
        print("2. 如果RDKit导入失败:")
        print("   pip install rdkit-pypi")
        print("3. 如果CUDA相关错误:")
        print("   检查CUDA版本兼容性，或使用CPU版本")


if __name__ == "__main__":
    main()