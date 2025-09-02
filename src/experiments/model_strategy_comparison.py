"""
模型策略对比实验 - 独立任务模型 vs 单一多输出模型
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Optional, List, Tuple
import time
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_multi_target_metrics, print_metrics
from src.data.multi_task_preprocessor import MultiTaskPreprocessor
from src.models.multi_task_trainer import MultiTaskTrainer
from src.models.baseline import BaselineModel
from sklearn.model_selection import KFold


def run_model_strategy_comparison(config_path: str = None, 
                                 data_dir: str = "data"):
    """
    运行模型策略对比实验
    
    Args:
        config_path: 配置文件路径
        data_dir: 数据目录路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置日志
    logger = setup_logger(
        name="model_strategy_comparison",
        level=config['experiment']['logging']['level'],
        log_dir=config['experiment']['logging']['log_dir'] if config['experiment']['logging']['save_logs'] else None,
        log_to_file=config['experiment']['logging']['save_logs']
    )
    
    logger.info("=" * 80)
    logger.info("开始模型策略对比实验 - 独立任务模型 vs 单一多输出模型")
    logger.info("=" * 80)
    
    try:
        # 1. 数据预处理
        logger.info("步骤 1: 数据预处理和准备")
        preprocessor = MultiTaskPreprocessor(config)
        
        # 加载数据
        datasets = preprocessor.load_competition_data(data_dir)
        
        # 准备多任务数据（只使用有完整标签的数据）
        main_train = datasets['main_train']
        
        # 获取有完整目标值的样本
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        complete_mask = main_train[target_cols].notnull().all(axis=1)
        complete_data = main_train[complete_mask].reset_index(drop=True)
        
        logger.info(f"完整标签数据: {len(complete_data)} 样本 (原始: {len(main_train)} 样本)")
        
        if len(complete_data) < 50:
            logger.warning("完整标签数据太少，使用所有有效数据进行对比")
            # 如果完整数据太少，我们使用FFV数据（最多）进行对比
            task_datasets_raw = preprocessor.prepare_task_specific_datasets(datasets)
            ffv_data = task_datasets_raw['FFV']
            
            # 为其他任务生成虚拟标签（用于演示）
            logger.warning("使用FFV数据生成多输出演示数据")
            demo_data = ffv_data.copy()
            np.random.seed(42)
            demo_data['Tg'] = demo_data['FFV'] * 1000 + np.random.normal(0, 10, len(demo_data))
            demo_data['Tc'] = demo_data['FFV'] * 0.5 + np.random.normal(0, 0.05, len(demo_data))
            demo_data['Density'] = 1.0 + demo_data['FFV'] * 0.3 + np.random.normal(0, 0.1, len(demo_data))
            demo_data['Rg'] = 20 + demo_data['FFV'] * 10 + np.random.normal(0, 2, len(demo_data))
            complete_data = demo_data.sample(n=min(1000, len(demo_data)), random_state=42).reset_index(drop=True)
            logger.info(f"生成演示数据: {len(complete_data)} 样本")
        
        # 2. 提取特征
        logger.info("步骤 2: 特征提取")
        X_df, feature_names = preprocessor.extract_features_for_task(complete_data)
        X = X_df.values  # 转换为numpy数组
        y = complete_data[target_cols].values
        
        logger.info(f"特征维度: {X.shape}")
        logger.info(f"目标维度: {y.shape}")
        
        # 3. 对比实验
        logger.info("步骤 3: 模型策略对比")
        
        # 策略1: 独立任务模型
        logger.info("\n=== 策略1: 独立任务模型 ===")
        independent_results = run_independent_task_strategy(X, y, target_cols, config, logger)
        
        # 策略2: 单一多输出模型
        logger.info("\n=== 策略2: 单一多输出模型 ===")
        multioutput_results = run_multioutput_strategy(X, y, target_cols, config, logger)
        
        # 4. 生成对比报告
        logger.info("步骤 4: 生成对比报告")
        generate_strategy_comparison_report(
            independent_results, multioutput_results, target_cols, logger
        )
        
        logger.info("=" * 80)
        logger.info("模型策略对比实验完成!")
        logger.info("=" * 80)
        
        return {
            'independent': independent_results,
            'multioutput': multioutput_results
        }
        
    except Exception as e:
        logger.error(f"实验过程中发生错误: {str(e)}")
        raise e


def run_independent_task_strategy(X: np.ndarray, 
                                 y: np.ndarray, 
                                 target_cols: List[str],
                                 config: Dict,
                                 logger) -> Dict:
    """
    运行独立任务模型策略
    
    Args:
        X: 特征数据
        y: 目标数据
        target_cols: 目标列名
        config: 配置
        logger: 日志器
        
    Returns:
        实验结果
    """
    start_time = time.time()
    
    # 使用5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(kf.split(X, y))
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"独立任务策略 - 第 {fold + 1} 折")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 为每个任务训练独立模型
        task_predictions = {}
        task_maes = {}
        
        for i, target_name in enumerate(target_cols):
            # 训练单任务模型
            model = BaselineModel(
                model_type='xgboost',
                model_params=config['models']['baseline']['xgboost'],
                use_multioutput=False
            )
            
            model.fit(X_train_fold, y_train_fold[:, i])
            pred = model.predict(X_val_fold)
            
            if len(pred.shape) > 1:
                pred = pred.ravel()
            
            task_predictions[target_name] = pred
            task_maes[target_name] = np.mean(np.abs(y_val_fold[:, i] - pred))
        
        # 组合预测结果
        y_pred_fold = np.column_stack([task_predictions[col] for col in target_cols])
        
        # 计算指标
        metrics = calculate_multi_target_metrics(y_val_fold, y_pred_fold, target_cols)
        fold_results.append(metrics)
        
        # 打印本折结果
        logger.info(f"  第 {fold + 1} 折平均MAE: {metrics['overall']['mean_mae']:.6f}")
    
    # 计算平均结果
    avg_metrics = average_cv_results(fold_results, target_cols)
    
    training_time = time.time() - start_time
    
    result = {
        'strategy': 'independent_tasks',
        'fold_results': fold_results,
        'average_metrics': avg_metrics,
        'training_time': training_time,
        'model_count': len(target_cols)
    }
    
    logger.info(f"独立任务策略完成 - 平均MAE: {avg_metrics['overall']['mean_mae']:.6f} ± {avg_metrics['overall']['mean_mae_std']:.6f}")
    
    return result


def run_multioutput_strategy(X: np.ndarray, 
                           y: np.ndarray, 
                           target_cols: List[str],
                           config: Dict,
                           logger) -> Dict:
    """
    运行单一多输出模型策略
    
    Args:
        X: 特征数据
        y: 目标数据
        target_cols: 目标列名
        config: 配置
        logger: 日志器
        
    Returns:
        实验结果
    """
    start_time = time.time()
    
    # 使用5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(kf.split(X, y))
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"多输出策略 - 第 {fold + 1} 折")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 训练单一多输出模型
        model = BaselineModel(
            model_type='xgboost',
            model_params=config['models']['baseline']['xgboost'],
            use_multioutput=True
        )
        
        model.fit(X_train_fold, y_train_fold, target_names=target_cols)
        y_pred_fold = model.predict(X_val_fold)
        
        # 计算指标
        metrics = calculate_multi_target_metrics(y_val_fold, y_pred_fold, target_cols)
        fold_results.append(metrics)
        
        # 打印本折结果
        logger.info(f"  第 {fold + 1} 折平均MAE: {metrics['overall']['mean_mae']:.6f}")
    
    # 计算平均结果
    avg_metrics = average_cv_results(fold_results, target_cols)
    
    training_time = time.time() - start_time
    
    result = {
        'strategy': 'multioutput',
        'fold_results': fold_results,
        'average_metrics': avg_metrics,
        'training_time': training_time,
        'model_count': 1
    }
    
    logger.info(f"多输出策略完成 - 平均MAE: {avg_metrics['overall']['mean_mae']:.6f} ± {avg_metrics['overall']['mean_mae_std']:.6f}")
    
    return result


def average_cv_results(fold_results: List[Dict], target_cols: List[str]) -> Dict:
    """
    计算交叉验证的平均结果
    
    Args:
        fold_results: 每折的结果列表
        target_cols: 目标列名
        
    Returns:
        平均结果字典
    """
    avg_metrics = {}
    
    # 为每个目标计算平均指标
    for target_name in target_cols:
        target_metrics = {}
        for metric_name in ['mae', 'rmse']:
            values = [fold[target_name][metric_name] for fold in fold_results]
            target_metrics[metric_name] = np.mean(values)
            target_metrics[f'{metric_name}_std'] = np.std(values)
        avg_metrics[target_name] = target_metrics
    
    # 计算总体平均指标
    overall_metrics = {}
    for metric_name in ['mean_mae', 'mean_rmse', 'total_mae']:
        values = [fold['overall'][metric_name] for fold in fold_results]
        overall_metrics[metric_name] = np.mean(values)
        overall_metrics[f'{metric_name}_std'] = np.std(values)
    avg_metrics['overall'] = overall_metrics
    
    return avg_metrics


def generate_strategy_comparison_report(independent_results: Dict,
                                      multioutput_results: Dict,
                                      target_cols: List[str],
                                      logger):
    """
    生成策略对比报告
    
    Args:
        independent_results: 独立任务结果
        multioutput_results: 多输出结果
        target_cols: 目标列名
        logger: 日志器
    """
    logger.info("\n" + "=" * 100)
    logger.info("模型策略对比报告")
    logger.info("=" * 100)
    
    # 准备对比数据
    comparison_data = []
    
    # 独立任务策略
    independent_avg = independent_results['average_metrics']
    comparison_data.append({
        'strategy': '独立任务模型',
        'mean_mae': independent_avg['overall']['mean_mae'],
        'mean_mae_std': independent_avg['overall']['mean_mae_std'],
        'mean_rmse': independent_avg['overall']['mean_rmse'],
        'mean_rmse_std': independent_avg['overall']['mean_rmse_std'],
        'training_time': independent_results['training_time'],
        'model_count': independent_results['model_count']
    })
    
    # 多输出策略
    multioutput_avg = multioutput_results['average_metrics']
    comparison_data.append({
        'strategy': '单一多输出模型',
        'mean_mae': multioutput_avg['overall']['mean_mae'],
        'mean_mae_std': multioutput_avg['overall']['mean_mae_std'],
        'mean_rmse': multioutput_avg['overall']['mean_rmse'],
        'mean_rmse_std': multioutput_avg['overall']['mean_rmse_std'],
        'training_time': multioutput_results['training_time'],
        'model_count': multioutput_results['model_count']
    })
    
    # 创建对比DataFrame
    df = pd.DataFrame(comparison_data)
    
    # 打印总体对比
    logger.info(f"\n总体性能对比:")
    for _, row in df.iterrows():
        logger.info(f"  {row['strategy']:<15} "
                   f"平均MAE: {row['mean_mae']:.6f} ± {row['mean_mae_std']:.6f} "
                   f"训练时间: {row['training_time']:.1f}s "
                   f"模型数量: {row['model_count']}")
    
    # 详细任务对比
    logger.info(f"\n各任务详细对比:")
    logger.info(f"{'任务':<10} {'独立模型MAE':<15} {'多输出MAE':<15} {'性能差异':<12}")
    logger.info(f"{'-'*60}")
    
    for target_name in target_cols:
        independent_mae = independent_avg[target_name]['mae']
        multioutput_mae = multioutput_avg[target_name]['mae']
        difference = ((independent_mae - multioutput_mae) / independent_mae) * 100
        
        logger.info(f"{target_name:<10} "
                   f"{independent_mae:<15.6f} "
                   f"{multioutput_mae:<15.6f} "
                   f"{difference:>+6.2f}%")
    
    # 分析和建议
    better_strategy = "独立任务模型" if independent_avg['overall']['mean_mae'] < multioutput_avg['overall']['mean_mae'] else "单一多输出模型"
    mae_improvement = abs(independent_avg['overall']['mean_mae'] - multioutput_avg['overall']['mean_mae'])
    improvement_pct = (mae_improvement / max(independent_avg['overall']['mean_mae'], multioutput_avg['overall']['mean_mae'])) * 100
    
    logger.info(f"\n🎯 策略分析:")
    logger.info(f"  最佳策略: {better_strategy}")
    logger.info(f"  性能差异: {improvement_pct:.2f}%")
    
    if improvement_pct < 1.0:
        logger.info(f"  📊 结论: 两种策略性能相近，建议考虑以下因素:")
        logger.info(f"    - 独立模型: 更灵活，可针对性优化，但复杂度高")
        logger.info(f"    - 多输出模型: 更简洁，训练快速，但调优受限")
    else:
        if better_strategy == "独立任务模型":
            logger.info(f"  📊 结论: 独立任务模型表现更好，适合当前多任务场景")
        else:
            logger.info(f"  📊 结论: 单一多输出模型表现更好，建议采用")
    
    # 保存结果
    results_dir = Path("results/model_strategy_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(results_dir / "strategy_comparison_summary.csv", index=False)
    logger.info(f"\n详细结果已保存: {results_dir / 'strategy_comparison_summary.csv'}")
    logger.info("=" * 100)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行模型策略对比实验')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录路径')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_model_strategy_comparison(
        config_path=args.config,
        data_dir=args.data_dir
    )
    
    return results


if __name__ == "__main__":
    main()