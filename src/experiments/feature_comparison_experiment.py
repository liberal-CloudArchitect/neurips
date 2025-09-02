"""
特征对比实验脚本 - 测试MACCS Keys的性能提升
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Optional, List
import time
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config, update_config
from src.utils.logger import setup_logger
from src.data.multi_task_preprocessor import MultiTaskPreprocessor
from src.models.multi_task_trainer import MultiTaskTrainer


def run_feature_comparison_experiment(config_path: str = None, 
                                    data_dir: str = "data",
                                    test_task: str = "FFV"):
    """
    运行特征对比实验
    
    Args:
        config_path: 配置文件路径
        data_dir: 数据目录路径
        test_task: 用于测试的任务名称（选择数据量大的任务）
    """
    # 加载配置
    base_config = load_config(config_path)
    
    # 设置日志
    logger = setup_logger(
        name="feature_comparison",
        level=base_config['experiment']['logging']['level'],
        log_dir=base_config['experiment']['logging']['log_dir'] if base_config['experiment']['logging']['save_logs'] else None,
        log_to_file=base_config['experiment']['logging']['save_logs']
    )
    
    logger.info("=" * 80)
    logger.info("开始特征对比实验 - 测试MACCS Keys性能提升")
    logger.info("=" * 80)
    
    # 定义特征组合实验配置
    feature_configs = [
        {
            "name": "Morgan_Only",
            "morgan_enabled": True,
            "maccs_enabled": False,
            "descriptors_enabled": False
        },
        {
            "name": "Morgan_Descriptors",
            "morgan_enabled": True,
            "maccs_enabled": False,
            "descriptors_enabled": True
        },
        {
            "name": "Morgan_MACCS",
            "morgan_enabled": True,
            "maccs_enabled": True,
            "descriptors_enabled": False
        },
        {
            "name": "All_Features",
            "morgan_enabled": True,
            "maccs_enabled": True,
            "descriptors_enabled": True
        },
        {
            "name": "MACCS_Only",
            "morgan_enabled": False,
            "maccs_enabled": True,
            "descriptors_enabled": False
        }
    ]
    
    results = []
    
    try:
        for i, feature_config in enumerate(feature_configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"实验 {i+1}/{len(feature_configs)}: {feature_config['name']}")
            logger.info(f"{'='*60}")
            
            # 创建特定的配置
            config = update_config(base_config, {
                'features': {
                    'morgan_fingerprint': {
                        'enabled': feature_config['morgan_enabled']
                    },
                    'maccs_keys': {
                        'enabled': feature_config['maccs_enabled']
                    },
                    'rdkit_descriptors': {
                        'use_2d': feature_config['descriptors_enabled']
                    }
                }
            })
            
            # 运行单次实验
            result = run_single_feature_experiment(
                config, data_dir, test_task, feature_config['name'], logger
            )
            
            if result:
                results.append(result)
                logger.info(f"实验完成 - {feature_config['name']}: MAE = {result['mae']:.6f}")
            else:
                logger.error(f"实验失败 - {feature_config['name']}")
        
        # 生成对比报告
        if results:
            generate_comparison_report(results, logger)
        
        logger.info("=" * 80)
        logger.info("特征对比实验完成!")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"实验过程中发生错误: {str(e)}")
        raise e


def run_single_feature_experiment(config: Dict, 
                                 data_dir: str, 
                                 test_task: str,
                                 experiment_name: str,
                                 logger) -> Optional[Dict]:
    """
    运行单个特征配置的实验
    
    Args:
        config: 实验配置
        data_dir: 数据目录
        test_task: 测试任务
        experiment_name: 实验名称
        logger: 日志器
        
    Returns:
        实验结果字典
    """
    try:
        start_time = time.time()
        
        # 1. 数据预处理
        logger.info(f"  步骤 1: 数据预处理 - {experiment_name}")
        preprocessor = MultiTaskPreprocessor(config)
        
        # 加载数据
        datasets = preprocessor.load_competition_data(data_dir)
        
        # 准备任务特定数据集
        task_datasets_raw = preprocessor.prepare_task_specific_datasets(datasets)
        
        # 仅处理测试任务
        if test_task not in task_datasets_raw:
            logger.error(f"测试任务 {test_task} 不存在")
            return None
        
        # 准备单个任务的数据
        task_dataset = preprocessor.prepare_single_task_dataset(test_task, test_size=0.2)
        
        # 记录特征维度
        feature_dim = task_dataset['X_train'].shape[1]
        logger.info(f"  特征维度: {feature_dim}")
        
        # 2. 模型训练
        logger.info(f"  步骤 2: 模型训练 - {experiment_name}")
        trainer = MultiTaskTrainer(config)
        
        # 只训练测试任务
        task_datasets = {test_task: task_dataset}
        
        # 训练模型（只用XGBoost加快速度）
        all_results = trainer.train_all_tasks(
            task_datasets=task_datasets,
            model_types=['xgboost']  # 只使用XGBoost加快实验速度
        )
        
        # 3. 提取结果
        if test_task in trainer.task_models:
            model_info = trainer.task_models[test_task]
            metrics = model_info['metrics']
            
            training_time = time.time() - start_time
            
            result = {
                'experiment_name': experiment_name,
                'task': test_task,
                'feature_dim': feature_dim,
                'mae': metrics['mae'],
                'mae_std': metrics['mae_std'],
                'rmse': metrics['rmse'],
                'rmse_std': metrics['rmse_std'],
                'training_time': training_time,
                'samples': len(task_datasets_raw[test_task])
            }
            
            return result
        else:
            logger.error(f"训练失败 - {experiment_name}")
            return None
            
    except Exception as e:
        logger.error(f"单个实验失败 - {experiment_name}: {str(e)}")
        return None


def generate_comparison_report(results: List[Dict], logger):
    """
    生成特征对比报告
    
    Args:
        results: 实验结果列表
        logger: 日志器
    """
    logger.info("\n" + "=" * 100)
    logger.info("特征对比实验报告")
    logger.info("=" * 100)
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('mae')
    
    # 找出最佳配置
    best_result = df.iloc[0]
    baseline_result = df[df['experiment_name'] == 'Morgan_Only'].iloc[0] if len(df[df['experiment_name'] == 'Morgan_Only']) > 0 else None
    
    logger.info(f"\n任务: {best_result['task']}")
    logger.info(f"样本数: {best_result['samples']}")
    
    logger.info(f"\n特征配置性能排名:")
    for i, (_, row) in enumerate(df.iterrows()):
        logger.info(f"  {i+1}. {row['experiment_name']:<20} "
                   f"MAE: {row['mae']:.6f} ± {row['mae_std']:.6f} "
                   f"特征维度: {row['feature_dim']:>4d} "
                   f"时间: {row['training_time']:.1f}s")
    
    # 性能提升分析
    if baseline_result is not None:
        logger.info(f"\n性能提升分析 (vs Morgan_Only基线):")
        baseline_mae = baseline_result['mae']
        
        for _, row in df.iterrows():
            if row['experiment_name'] != 'Morgan_Only':
                improvement = (baseline_mae - row['mae']) / baseline_mae * 100
                logger.info(f"  {row['experiment_name']:<20} "
                           f"提升: {improvement:+.2f}% "
                           f"(MAE: {baseline_mae:.6f} → {row['mae']:.6f})")
    
    # MACCS Keys价值分析
    morgan_only = df[df['experiment_name'] == 'Morgan_Only']
    morgan_maccs = df[df['experiment_name'] == 'Morgan_MACCS']
    
    if len(morgan_only) > 0 and len(morgan_maccs) > 0:
        mae_morgan = morgan_only.iloc[0]['mae']
        mae_morgan_maccs = morgan_maccs.iloc[0]['mae']
        maccs_improvement = (mae_morgan - mae_morgan_maccs) / mae_morgan * 100
        
        logger.info(f"\n🎯 MACCS Keys价值分析:")
        logger.info(f"  Morgan Only:      MAE = {mae_morgan:.6f}")
        logger.info(f"  Morgan + MACCS:   MAE = {mae_morgan_maccs:.6f}")
        logger.info(f"  MACCS Keys提升:   {maccs_improvement:+.2f}%")
        
        if maccs_improvement > 0:
            logger.info(f"  ✅ MACCS Keys有效！建议启用")
        else:
            logger.info(f"  ❌ MACCS Keys无明显提升")
    
    # 保存结果
    results_dir = Path("results/feature_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(results_dir / "feature_comparison_results.csv", index=False)
    logger.info(f"\n详细结果已保存: {results_dir / 'feature_comparison_results.csv'}")
    logger.info("=" * 100)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行特征对比实验')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录路径')
    parser.add_argument('--task', type=str, default='FFV', help='测试任务名称')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_feature_comparison_experiment(
        config_path=args.config,
        data_dir=args.data_dir,
        test_task=args.task
    )
    
    return results


if __name__ == "__main__":
    main()