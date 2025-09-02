"""
基线模型实验脚本
执行第一阶段：基础设施与基线模型训练
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import print_metrics
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import BaselineTrainer


def run_baseline_experiment(config_path: str = None, 
                          train_data_path: str = None,
                          test_data_path: str = None):
    """
    运行基线实验
    
    Args:
        config_path: 配置文件路径
        train_data_path: 训练数据路径（可选，覆盖配置文件中的路径）
        test_data_path: 测试数据路径（可选，覆盖配置文件中的路径）
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置日志
    logger = setup_logger(
        name="baseline_experiment",
        level=config['experiment']['logging']['level'],
        log_dir=config['experiment']['logging']['log_dir'] if config['experiment']['logging']['save_logs'] else None,
        log_to_file=config['experiment']['logging']['save_logs']
    )
    
    logger.info("=" * 60)
    logger.info("开始基线模型实验")
    logger.info("=" * 60)
    
    try:
        # 1. 数据预处理
        logger.info("步骤 1: 数据预处理")
        preprocessor = DataPreprocessor(config)
        
        # 加载数据
        train_df, test_df = preprocessor.load_data(train_data_path, test_data_path)
        
        # 准备数据集
        dataset = preprocessor.prepare_datasets(train_df, test_df)
        
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        target_names = dataset['target_names']
        feature_names = dataset['feature_names']
        
        logger.info(f"训练数据准备完成:")
        logger.info(f"  特征维度: {X_train.shape}")
        logger.info(f"  目标维度: {y_train.shape}")
        logger.info(f"  目标名称: {target_names}")
        
        # 2. 创建交叉验证分割
        logger.info("步骤 2: 创建交叉验证分割")
        cv_splits = preprocessor.create_cv_splits(X_train, y_train)
        
        # 3. 训练基线模型
        logger.info("步骤 3: 训练基线模型")
        trainer = BaselineTrainer(config)
        
        # 训练所有基线模型
        all_results = trainer.train_all_baseline_models(
            X_train=X_train,
            y_train=y_train,
            target_names=target_names,
            cv_splits=cv_splits
        )
        
        # 4. 保存结果
        logger.info("步骤 4: 保存训练结果")
        results_dir = Path(config['output']['results_dir']) / "baseline_models"
        trainer.save_results(results_dir)
        
        # 5. 分析特征重要性
        logger.info("步骤 5: 分析特征重要性")
        analyze_feature_importance(trainer, feature_names, target_names, results_dir)
        
        # 6. 生成预测（如果有测试数据）
        if dataset['X_test'] is not None:
            logger.info("步骤 6: 生成测试集预测")
            generate_predictions(trainer, dataset, results_dir)
        
        # 7. 生成总结报告
        logger.info("步骤 7: 生成实验总结")
        generate_summary_report(all_results, results_dir, logger)
        
        logger.info("=" * 60)
        logger.info("基线模型实验完成!")
        logger.info("=" * 60)
        
        return all_results
        
    except Exception as e:
        logger.error(f"实验过程中发生错误: {str(e)}")
        raise e


def analyze_feature_importance(trainer: BaselineTrainer, 
                             feature_names: list,
                             target_names: list,
                             results_dir: Path):
    """
    分析特征重要性
    
    Args:
        trainer: 训练器实例
        feature_names: 特征名称列表
        target_names: 目标名称列表
        results_dir: 结果保存目录
    """
    logger = logging.getLogger(__name__)
    logger.info("分析特征重要性...")
    
    importance_results = {}
    
    # 分析每个模型的特征重要性
    for model_type in trainer.results.keys():
        try:
            # 获取第一折的模型作为代表
            model = trainer.results[model_type]['models'][0]
            importance_dict = model.get_feature_importance()
            
            if importance_dict:
                importance_results[model_type] = {}
                
                for target_name, importances in importance_dict.items():
                    # 创建特征重要性DataFrame
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    importance_results[model_type][target_name] = importance_df
                    
                    # 保存特征重要性
                    save_path = results_dir / f"{model_type}_{target_name}_feature_importance.csv"
                    importance_df.to_csv(save_path, index=False)
                    
                    # 打印前10个重要特征
                    logger.info(f"\n{model_type} - {target_name} 前10个重要特征:")
                    for idx, row in importance_df.head(10).iterrows():
                        logger.info(f"  {row['feature']}: {row['importance']:.6f}")
                        
        except Exception as e:
            logger.warning(f"分析 {model_type} 特征重要性时出错: {str(e)}")
    
    return importance_results


def generate_predictions(trainer: BaselineTrainer, dataset: dict, results_dir: Path):
    """
    生成测试集预测
    
    Args:
        trainer: 训练器实例
        dataset: 数据集字典
        results_dir: 结果保存目录
    """
    logger = logging.getLogger(__name__)
    logger.info("生成测试集预测...")
    
    X_test = dataset['X_test']
    target_names = dataset['target_names']
    test_indices = dataset['test_indices']
    
    # 为每个模型生成预测
    for model_type in trainer.results.keys():
        try:
            # 获取最佳模型（第一折作为代表）
            model = trainer.results[model_type]['models'][0]
            
            # 生成预测
            predictions = model.predict(X_test)
            
            # 创建预测DataFrame
            pred_df = pd.DataFrame(predictions, columns=target_names)
            if test_indices is not None:
                pred_df.index = test_indices
            
            # 保存预测结果
            save_path = results_dir / f"{model_type}_predictions.csv"
            pred_df.to_csv(save_path, index=True)
            
            logger.info(f"{model_type} 预测已保存至: {save_path}")
            
        except Exception as e:
            logger.error(f"生成 {model_type} 预测时出错: {str(e)}")


def generate_summary_report(all_results: dict, results_dir: Path, logger):
    """
    生成实验总结报告
    
    Args:
        all_results: 所有模型结果
        results_dir: 结果保存目录
        logger: 日志器
    """
    logger.info("生成实验总结报告...")
    
    summary_data = []
    
    # 收集所有模型的性能指标
    for model_type, results in all_results.items():
        avg_metrics = results['average_metrics']
        
        # 总体指标
        overall_metrics = avg_metrics['overall']
        summary_row = {
            'model_type': model_type,
            'mean_mae': overall_metrics['mean_mae'],
            'mean_mae_std': overall_metrics['mean_mae_std'],
            'mean_rmse': overall_metrics['mean_rmse'],
            'mean_rmse_std': overall_metrics['mean_rmse_std'],
            'total_mae': overall_metrics['total_mae'],
            'total_mae_std': overall_metrics['total_mae_std']
        }
        
        # 各目标的MAE
        for target_name in avg_metrics.keys():
            if target_name != 'overall':
                summary_row[f'{target_name}_mae'] = avg_metrics[target_name]['mae']
                summary_row[f'{target_name}_mae_std'] = avg_metrics[target_name]['mae_std']
        
        summary_data.append(summary_row)
    
    # 创建总结DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('mean_mae')
    
    # 保存总结
    summary_path = results_dir / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("实验总结报告")
    logger.info("=" * 80)
    
    logger.info("\n模型性能排名 (按平均MAE排序):")
    for idx, row in summary_df.iterrows():
        logger.info(f"{idx+1}. {row['model_type']}: "
                   f"平均MAE = {row['mean_mae']:.6f} ± {row['mean_mae_std']:.6f}")
    
    # 找出最佳模型
    best_model = summary_df.iloc[0]
    logger.info(f"\n最佳模型: {best_model['model_type']}")
    logger.info(f"  平均MAE: {best_model['mean_mae']:.6f} ± {best_model['mean_mae_std']:.6f}")
    logger.info(f"  平均RMSE: {best_model['mean_rmse']:.6f} ± {best_model['mean_rmse_std']:.6f}")
    
    logger.info(f"\n详细结果已保存至: {summary_path}")
    logger.info("=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行基线模型实验')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--train_data', type=str, help='训练数据路径')
    parser.add_argument('--test_data', type=str, help='测试数据路径')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_baseline_experiment(
        config_path=args.config,
        train_data_path=args.train_data,
        test_data_path=args.test_data
    )
    
    return results


if __name__ == "__main__":
    main()