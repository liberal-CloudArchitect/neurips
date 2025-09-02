"""
第五阶段：模型集成实验脚本
整合基线模型、GNN模型和Transformer模型的预测结果
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, List, Tuple, Optional
import torch
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_multi_target_metrics


class ModelEnsemble:
    """模型集成器"""
    
    def __init__(self, config: Dict):
        """
        初始化模型集成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型路径配置
        self.baseline_results_path = Path("results/multi_task_models")
        self.gnn_results_path = Path("results/gnn_models") 
        self.transformer_results_path = Path("results/transformer_models")
        
        # 存储各模型的预测结果
        self.baseline_predictions = {}
        self.gnn_predictions = {}
        self.transformer_predictions = {}
        
        # 集成模型
        self.ensemble_models = {}
        
    def load_baseline_predictions(self) -> Dict[str, np.ndarray]:
        """加载基线模型预测结果"""
        self.logger.info("加载基线模型预测结果...")
        
        predictions = {}
        
        # 从基线模型结果中加载预测
        # 这里需要根据实际的基线模型保存格式来实现
        # 暂时返回空字典，实际实现时需要加载真实的预测结果
        
        return predictions
    
    def load_gnn_predictions(self) -> Dict[str, np.ndarray]:
        """加载GNN模型预测结果"""
        self.logger.info("加载GNN模型预测结果...")
        
        predictions = {}
        
        # 从GNN模型结果中加载预测
        # 需要实现GNN模型的预测加载
        
        return predictions
    
    def load_transformer_predictions(self) -> Dict[str, np.ndarray]:
        """加载Transformer模型预测结果"""
        self.logger.info("加载Transformer模型预测结果...")
        
        predictions = {}
        
        # 从Transformer模型结果中加载预测
        # 需要实现Transformer模型的预测加载
        
        return predictions
    
    def simple_average_ensemble(self, task: str) -> np.ndarray:
        """简单平均集成"""
        predictions = []
        
        if task in self.baseline_predictions:
            predictions.append(self.baseline_predictions[task])
        if task in self.gnn_predictions:
            predictions.append(self.gnn_predictions[task])
        if task in self.transformer_predictions:
            predictions.append(self.transformer_predictions[task])
        
        if not predictions:
            raise ValueError(f"没有找到任务 {task} 的预测结果")
        
        # 简单平均
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def weighted_average_ensemble(self, task: str, weights: List[float]) -> np.ndarray:
        """加权平均集成"""
        predictions = []
        
        if task in self.baseline_predictions:
            predictions.append(self.baseline_predictions[task])
        if task in self.gnn_predictions:
            predictions.append(self.gnn_predictions[task])
        if task in self.transformer_predictions:
            predictions.append(self.transformer_predictions[task])
        
        if not predictions:
            raise ValueError(f"没有找到任务 {task} 的预测结果")
        
        if len(weights) != len(predictions):
            raise ValueError("权重数量与模型数量不匹配")
        
        # 加权平均
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def stacking_ensemble(self, 
                         train_predictions: Dict[str, List[np.ndarray]], 
                         train_targets: Dict[str, np.ndarray],
                         meta_model_type: str = 'ridge') -> Dict[str, object]:
        """Stacking集成"""
        self.logger.info(f"训练Stacking集成模型，元模型类型: {meta_model_type}")
        
        stacking_models = {}
        
        for task in train_targets.keys():
            if task not in train_predictions:
                continue
            
            # 准备元特征（各基础模型的预测结果）
            X_meta = np.column_stack(train_predictions[task])
            y_meta = train_targets[task]
            
            # 选择元模型
            if meta_model_type == 'ridge':
                meta_model = Ridge(alpha=1.0)
            elif meta_model_type == 'linear':
                meta_model = LinearRegression()
            elif meta_model_type == 'rf':
                meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"不支持的元模型类型: {meta_model_type}")
            
            # 训练元模型
            meta_model.fit(X_meta, y_meta)
            stacking_models[task] = meta_model
            
            # 交叉验证评估
            cv_scores = cross_val_score(meta_model, X_meta, y_meta, 
                                      cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            self.logger.info(f"任务 {task} Stacking MAE: {cv_mae:.6f} (±{cv_scores.std():.6f})")
        
        return stacking_models
    
    def evaluate_ensemble_strategy(self, 
                                 strategy: str,
                                 test_predictions: Dict[str, List[np.ndarray]],
                                 test_targets: Dict[str, np.ndarray],
                                 **kwargs) -> Dict[str, float]:
        """评估集成策略"""
        self.logger.info(f"评估集成策略: {strategy}")
        
        results = {}
        
        for task in test_targets.keys():
            if task not in test_predictions:
                continue
            
            if strategy == 'simple_average':
                pred = np.mean(test_predictions[task], axis=0)
            elif strategy == 'weighted_average':
                weights = kwargs.get('weights', [1.0] * len(test_predictions[task]))
                pred = np.average(test_predictions[task], axis=0, weights=weights)
            elif strategy == 'stacking':
                stacking_model = kwargs.get('stacking_models', {}).get(task)
                if stacking_model is None:
                    continue
                X_meta = np.column_stack(test_predictions[task])
                pred = stacking_model.predict(X_meta)
            else:
                raise ValueError(f"不支持的集成策略: {strategy}")
            
            # 计算MAE
            mae = mean_absolute_error(test_targets[task], pred)
            results[task] = mae
            
            self.logger.info(f"任务 {task} {strategy} MAE: {mae:.6f}")
        
        # 计算平均MAE
        avg_mae = np.mean(list(results.values()))
        results['average'] = avg_mae
        
        self.logger.info(f"{strategy} 平均MAE: {avg_mae:.6f}")
        
        return results


def run_ensemble_experiment(config_path: str = None,
                           data_dir: str = "data",
                           strategy: str = "all",
                           max_samples: Optional[int] = None):
    """
    运行模型集成实验
    
    Args:
        config_path: 配置文件路径
        data_dir: 数据目录路径
        strategy: 集成策略 ('simple', 'weighted', 'stacking', 'all')
        max_samples: 最大样本数（用于快速测试）
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置日志
    logger = setup_logger(
        name="ensemble_experiment",
        level=config['experiment']['logging']['level'],
        log_dir=config['experiment']['logging']['log_dir'] if config['experiment']['logging']['save_logs'] else None,
        log_to_file=config['experiment']['logging']['save_logs']
    )
    
    logger.info("=" * 80)
    logger.info("开始第五阶段模型集成实验")
    logger.info("=" * 80)
    
    try:
        # 1. 导入预测生成器
        from src.utils.prediction_generator import PredictionGenerator
        from src.models.ensemble import ModelEnsemble
        
        # 2. 初始化预测生成器
        logger.info("步骤 1: 初始化预测生成器")
        pred_generator = PredictionGenerator(config)
        
        # 3. 加载并分割数据
        logger.info("步骤 2: 加载并分割训练数据")
        data, true_labels = pred_generator.load_and_split_data(max_samples)
        
        # 4. 生成各模型预测结果
        logger.info("步骤 3: 生成各模型预测结果")
        all_predictions = pred_generator.generate_all_predictions(data, use_synthetic=True)
        
        # 5. 初始化集成器
        logger.info("步骤 4: 初始化模型集成器")
        ensemble = ModelEnsemble(
            target_columns=config['data']['target_columns'],
            ensemble_strategy=strategy if strategy != 'all' else 'stacking',
            meta_model_type='ridge'
        )
        
        # 6. 准备集成数据
        logger.info("步骤 5: 准备集成训练数据")
        
        # 加载验证集预测结果
        for model_type in ['baseline', 'gnn', 'transformer']:
            for task in config['data']['target_columns']:
                if (model_type in all_predictions and 
                    'validation' in all_predictions[model_type] and
                    task in all_predictions[model_type]['validation']):
                    
                    if model_type not in ensemble.model_predictions:
                        ensemble.model_predictions[model_type] = {}
                    
                    pred_data = all_predictions[model_type]['validation'][task]
                    # 过滤NaN值
                    valid_mask = ~np.isnan(pred_data)
                    ensemble.model_predictions[model_type][task] = pred_data[valid_mask]
        
        # 加载真实标签
        validation_labels = {}
        for task in config['data']['target_columns']:
            if task in true_labels:
                val_labels = true_labels[task]['validation']
                # 匹配预测数据的有效索引
                if (task in ensemble.model_predictions.get('transformer', {}) or 
                    task in ensemble.model_predictions.get('gnn', {}) or
                    task in ensemble.model_predictions.get('baseline', {})):
                    
                    # 使用第一个可用模型的有效索引
                    for model_type in ['transformer', 'gnn', 'baseline']:
                        if task in ensemble.model_predictions.get(model_type, {}):
                            pred_data = all_predictions[model_type]['validation'][task]
                            valid_mask = ~np.isnan(pred_data)
                            validation_labels[task] = val_labels[valid_mask]
                            break
        
        ensemble.load_true_labels(validation_labels)
        
        # 7. 计算各模型性能
        logger.info("步骤 6: 计算各模型性能")
        model_performances = ensemble.calculate_model_performances()
        
        # 8. 评估不同集成策略
        logger.info("步骤 7: 评估集成策略")
        
        strategies_to_test = []
        if strategy in ['simple', 'all']:
            strategies_to_test.append('simple')
        if strategy in ['weighted', 'all']:
            strategies_to_test.append('weighted')
        if strategy in ['stacking', 'all']:
            strategies_to_test.append('stacking')
        
        ensemble_results = {}
        
        for strat in strategies_to_test:
            logger.info(f"\n测试集成策略: {strat}")
            
            # 设置集成策略
            ensemble.ensemble_strategy = strat
            
            if strat == 'simple':
                ensemble_pred = ensemble.simple_average_ensemble()
            elif strat == 'weighted':
                ensemble_pred = ensemble.weighted_average_ensemble()
            elif strat == 'stacking':
                ensemble_pred = ensemble.stacking_ensemble()
            
            # 评估集成结果
            performance = ensemble.evaluate_ensemble(ensemble_pred)
            ensemble_results[strat] = performance
            
            logger.info(f"{strat} 集成策略完成")
        
        # 9. 保存结果
        logger.info("步骤 8: 保存集成实验结果")
        
        results_dir = Path(config['output']['results_dir']) / 'ensemble_models'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测数据
        pred_generator.save_predictions(results_dir)
        
        # 保存集成模型
        ensemble.save_ensemble_model(results_dir)
        
        # 保存实验报告
        report_data = []
        for strategy_name, performance in ensemble_results.items():
            for task, metrics in performance.items():
                if task != 'average' and isinstance(metrics, dict):
                    report_data.append({
                        'Strategy': strategy_name,
                        'Task': task,
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse'],
                        'Samples': metrics['samples']
                    })
        
        if report_data:
            results_df = pd.DataFrame(report_data)
            results_df.to_csv(results_dir / "ensemble_experiment_report.csv", index=False)
            
            # 打印结果摘要
            logger.info("\n" + "="*60)
            logger.info("集成实验结果摘要")
            logger.info("="*60)
            
            for strategy_name, performance in ensemble_results.items():
                if 'average' in performance:
                    avg_mae = performance['average']['mae']
                    avg_rmse = performance['average']['rmse']
                    logger.info(f"{strategy_name:12} | 平均MAE: {avg_mae:.6f} | 平均RMSE: {avg_rmse:.6f}")
            
            logger.info("="*60)
        
        logger.info("模型集成实验完成")
        
        return ensemble_results
        
    except Exception as e:
        logger.error(f"集成实验执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型集成实验脚本")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--strategy", type=str, default="all", 
                       choices=['simple', 'weighted', 'stacking', 'all'], 
                       help="集成策略")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（测试用）")
    
    args = parser.parse_args()
    
    run_ensemble_experiment(
        config_path=args.config,
        data_dir=args.data_dir,
        strategy=args.strategy,
        max_samples=args.max_samples
    )