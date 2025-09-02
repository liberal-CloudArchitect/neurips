"""
多任务实验脚本 - 使用真实NeurIPS 2025竞赛数据
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Optional
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.multi_task_preprocessor import MultiTaskPreprocessor
from src.models.multi_task_trainer import MultiTaskTrainer


def run_multi_task_experiment(config_path: str = None, 
                             data_dir: str = "data"):
    """
    运行多任务实验
    
    Args:
        config_path: 配置文件路径
        data_dir: 数据目录路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置日志
    logger = setup_logger(
        name="multi_task_experiment",
        level=config['experiment']['logging']['level'],
        log_dir=config['experiment']['logging']['log_dir'] if config['experiment']['logging']['save_logs'] else None,
        log_to_file=config['experiment']['logging']['save_logs']
    )
    
    logger.info("=" * 80)
    logger.info("开始NeurIPS 2025多任务聚合物预测实验")
    logger.info("=" * 80)
    
    try:
        # 1. 数据加载和分析
        logger.info("步骤 1: 加载和分析竞赛数据")
        preprocessor = MultiTaskPreprocessor(config)
        
        # 加载所有数据
        datasets = preprocessor.load_competition_data(data_dir)
        
        # 分析数据覆盖情况
        analysis = preprocessor.analyze_data_coverage(datasets)
        logger.info("数据分析完成")
        
        # 2. 准备任务特定数据集
        logger.info("步骤 2: 准备任务特定数据集")
        task_datasets_raw = preprocessor.prepare_task_specific_datasets(datasets)
        
        # 打印任务摘要
        task_summary = {}
        for task_name, task_data in task_datasets_raw.items():
            task_summary[task_name] = len(task_data)
            logger.info(f"  {task_name}: {len(task_data)} 样本")
        
        # 3. 为每个任务准备训练数据
        logger.info("步骤 3: 为每个任务准备训练和验证数据")
        task_datasets = {}
        
        for task_name in task_datasets_raw.keys():
            try:
                dataset = preprocessor.prepare_single_task_dataset(task_name, test_size=0.2)
                task_datasets[task_name] = dataset
                logger.info(f"  {task_name} 数据准备完成")
            except Exception as e:
                logger.error(f"准备 {task_name} 数据时出错: {str(e)}")
                continue
        
        if not task_datasets:
            logger.error("没有成功准备任何任务数据，实验终止")
            return None
        
        # 4. 训练多任务模型
        logger.info("步骤 4: 训练多任务模型")
        trainer = MultiTaskTrainer(config)
        
        # 训练所有任务
        all_results = trainer.train_all_tasks(
            task_datasets=task_datasets,
            model_types=['xgboost', 'lightgbm']
        )
        
        # 5. 保存训练结果
        logger.info("步骤 5: 保存训练结果")
        results_dir = Path(config['output']['results_dir']) / "multi_task_models"
        trainer.save_models(results_dir)
        
        # 6. 生成测试集预测
        if 'test' in datasets:
            logger.info("步骤 6: 生成测试集预测")
            test_df = datasets['test']
            
            # 为测试数据提取特征（使用第一个任务的特征提取器作为统一标准）
            first_task = list(task_datasets.keys())[0]
            first_dataset = task_datasets[first_task]
            
            # 重新用测试数据提取特征
            X_test, _ = preprocessor.extract_features_for_task(test_df)
            
            # 使用第一个任务的scaler标准化测试特征
            feature_scaler = first_dataset['feature_scaler']
            X_test_scaled = feature_scaler.transform(X_test)
            
            # 创建各任务的scaler字典
            feature_scalers = {
                task_name: dataset['feature_scaler'] 
                for task_name, dataset in task_datasets.items()
            }
            
            # 生成预测
            predictions = trainer.predict_all_tasks(X_test_scaled, feature_scalers)
            
            # 创建提交文件
            submission_df = preprocessor.create_submission_dataframe(test_df, predictions)
            
            # 保存提交文件
            submission_path = results_dir / "submission.csv"
            submission_df.to_csv(submission_path, index=False)
            logger.info(f"提交文件已保存: {submission_path}")
            
            # 显示预测摘要
            logger.info("\n预测结果摘要:")
            for task_name, pred_values in predictions.items():
                logger.info(f"  {task_name}: 均值={np.mean(pred_values):.6f}, "
                           f"标准差={np.std(pred_values):.6f}")
        
        # 7. 生成实验报告
        logger.info("步骤 7: 生成实验报告")
        generate_experiment_report(trainer, task_summary, results_dir, logger)
        
        logger.info("=" * 80)
        logger.info("多任务实验完成!")
        logger.info("=" * 80)
        
        return {
            'trainer': trainer,
            'results': all_results,
            'task_summary': task_summary,
            'submission_df': submission_df if 'test' in datasets else None
        }
        
    except Exception as e:
        logger.error(f"实验过程中发生错误: {str(e)}")
        raise e


def generate_experiment_report(trainer: MultiTaskTrainer, 
                             task_summary: Dict,
                             results_dir: Path,
                             logger):
    """
    生成实验报告
    
    Args:
        trainer: 训练器实例
        task_summary: 任务摘要
        results_dir: 结果目录
        logger: 日志器
    """
    logger.info("生成实验报告...")
    
    # 获取训练摘要
    training_summary = trainer.get_training_summary()
    
    # 创建报告DataFrame
    report_data = []
    
    for task_name in trainer.task_models.keys():
        model_info = trainer.task_models[task_name]
        
        report_row = {
            'task': task_name,
            'training_samples': task_summary.get(task_name, 0),
            'best_model': model_info['model_type'],
            'cv_mae': model_info['metrics']['mae'],
            'cv_mae_std': model_info['metrics']['mae_std'],
            'cv_rmse': model_info['metrics']['rmse'],
            'cv_rmse_std': model_info['metrics']['rmse_std']
        }
        
        report_data.append(report_row)
    
    # 创建报告DataFrame
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('cv_mae')
    else:
        # 如果没有训练成功的模型，创建空的DataFrame
        report_df = pd.DataFrame(columns=['task', 'training_samples', 'best_model', 'cv_mae', 'cv_mae_std', 'cv_rmse', 'cv_rmse_std'])
        logger.warning("没有成功训练的模型，无法生成完整报告")
    
    # 保存报告
    report_path = results_dir / "experiment_report.csv"
    report_df.to_csv(report_path, index=False)
    
    # 打印报告
    logger.info("\n" + "=" * 100)
    logger.info("实验报告 - 多任务学习结果")
    logger.info("=" * 100)
    
    logger.info(f"\n数据摘要:")
    total_samples = sum(task_summary.values())
    logger.info(f"  总训练样本: {total_samples}")
    logger.info(f"  任务数量: {len(task_summary)}")
    
    logger.info(f"\n各任务训练样本数:")
    for task, count in task_summary.items():
        logger.info(f"  {task}: {count} 样本")
    
    if len(report_df) > 0 and 'cv_mae' in report_df.columns:
        logger.info(f"\n模型性能 (按MAE排序):")
        for idx, row in report_df.iterrows():
            logger.info(f"  {idx+1}. {row['task']} ({row['best_model']}): "
                       f"MAE = {row['cv_mae']:.6f} ± {row['cv_mae_std']:.6f}")
        
        # 总体统计
        overall_mae = report_df['cv_mae'].mean()
        best_task = report_df.iloc[0]['task']
        worst_task = report_df.iloc[-1]['task']
        
        logger.info(f"\n总体统计:")
        logger.info(f"  平均MAE: {overall_mae:.6f}")
        logger.info(f"  最佳任务: {best_task} (MAE: {report_df.iloc[0]['cv_mae']:.6f})")
        logger.info(f"  最差任务: {worst_task} (MAE: {report_df.iloc[-1]['cv_mae']:.6f})")
    else:
        logger.warning("\n没有成功训练的模型，无法显示性能统计")
    
    logger.info(f"\n详细报告已保存: {report_path}")
    logger.info("=" * 100)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行多任务聚合物预测实验')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录路径')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_multi_task_experiment(
        config_path=args.config,
        data_dir=args.data_dir
    )
    
    return results


if __name__ == "__main__":
    main()