"""
多任务训练器 - 为每个目标变量独立训练模型
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import time
import pickle

from .baseline import BaselineModel
from ..utils.metrics import calculate_mae, calculate_rmse, print_metrics


class MultiTaskTrainer:
    """多任务训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.models_config = config['models']['baseline']
        self.output_config = config['output']
        
        self.logger = logging.getLogger(__name__)
        
        # 存储每个任务的结果
        self.task_results = {}
        self.task_models = {}
        
        # 支持的任务列表
        self.supported_tasks = ['FFV', 'Tg', 'Tc', 'Density', 'Rg']
    
    def train_single_task_model(self, 
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: Optional[np.ndarray] = None,
                               y_val: Optional[np.ndarray] = None,
                               task_name: str = 'unknown',
                               model_type: str = 'xgboost') -> BaselineModel:
        """
        为单个任务训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            task_name: 任务名称
            model_type: 模型类型
            
        Returns:
            训练好的模型
        """
        self.logger.info(f"开始训练 {task_name} 任务的 {model_type} 模型...")
        
        # 获取模型参数
        model_params = self.models_config[model_type].copy()
        
        # 创建单输出模型
        model = BaselineModel(
            model_type=model_type,
            model_params=model_params,
            use_multioutput=False  # 单任务使用单输出
        )
        
        start_time = time.time()
        
        # 训练模型（单目标，不需要target_names）
        model.fit(X_train, y_train, X_val, y_val)
        
        training_time = time.time() - start_time
        
        self.logger.info(f"{task_name}-{model_type} 模型训练完成，耗时 {training_time:.2f} 秒")
        
        return model
    
    def evaluate_single_task_model(self, 
                                  model: BaselineModel,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  task_name: str) -> Dict[str, float]:
        """
        评估单任务模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            task_name: 任务名称
            
        Returns:
            评估指标字典
        """
        # 预测
        y_pred = model.predict(X_test)
        
        # 对于单任务，预测结果可能是二维的，需要展平
        if len(y_pred.shape) > 1:
            y_pred = y_pred.ravel()
        
        # 计算指标
        metrics = {
            'mae': calculate_mae(y_test, y_pred),
            'rmse': calculate_rmse(y_test, y_pred),
            'task_name': task_name
        }
        
        return metrics
    
    def cross_validate_task(self,
                           task_dataset: Dict,
                           model_type: str = 'xgboost',
                           cv_folds: int = 5) -> Dict:
        """
        对单个任务进行交叉验证
        
        Args:
            task_dataset: 任务数据集
            model_type: 模型类型
            cv_folds: 交叉验证折数
            
        Returns:
            交叉验证结果
        """
        task_name = task_dataset['task_name']
        self.logger.info(f"开始 {task_name} 任务的 {cv_folds} 折交叉验证...")
        
        # 合并训练和验证数据进行交叉验证
        X = np.vstack([task_dataset['X_train'], task_dataset['X_val']])
        y = np.concatenate([task_dataset['y_train'], task_dataset['y_val']])
        
        # 创建交叉验证分割
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_splits = list(kf.split(X, y))
        
        # 存储每折的结果
        fold_results = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            self.logger.info(f"训练 {task_name} 第 {fold + 1} 折...")
            
            # 分割数据
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # 训练模型
            model = self.train_single_task_model(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                task_name=task_name,
                model_type=model_type
            )
            
            # 评估模型
            metrics = self.evaluate_single_task_model(
                model, X_val_fold, y_val_fold, task_name
            )
            
            fold_results.append(metrics)
            fold_models.append(model)
            
            # 打印本折结果
            self.logger.info(f"第 {fold + 1} 折 - MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}")
        
        # 计算平均结果
        avg_mae = np.mean([result['mae'] for result in fold_results])
        avg_rmse = np.mean([result['rmse'] for result in fold_results])
        std_mae = np.std([result['mae'] for result in fold_results])
        std_rmse = np.std([result['rmse'] for result in fold_results])
        
        avg_metrics = {
            'mae': avg_mae,
            'mae_std': std_mae,
            'rmse': avg_rmse,
            'rmse_std': std_rmse
        }
        
        self.logger.info(f"{task_name} {model_type} 交叉验证结果:")
        self.logger.info(f"  MAE: {avg_mae:.6f} ± {std_mae:.6f}")
        self.logger.info(f"  RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}")
        
        cv_results = {
            'task_name': task_name,
            'model_type': model_type,
            'fold_results': fold_results,
            'average_metrics': avg_metrics,
            'models': fold_models,
            'cv_splits': cv_splits
        }
        
        return cv_results
    
    def train_all_tasks(self,
                       task_datasets: Dict[str, Dict],
                       model_types: List[str] = None) -> Dict:
        """
        训练所有任务的模型
        
        Args:
            task_datasets: 任务数据集字典
            model_types: 要训练的模型类型列表
            
        Returns:
            所有任务的训练结果
        """
        if model_types is None:
            model_types = ['xgboost', 'lightgbm']
        
        self.logger.info(f"开始训练所有任务: {list(task_datasets.keys())}")
        
        all_results = {}
        
        for task_name, task_dataset in task_datasets.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"训练任务: {task_name}")
            self.logger.info(f"{'='*50}")
            
            task_results = {}
            
            for model_type in model_types:
                if model_type in self.models_config:
                    try:
                        cv_results = self.cross_validate_task(
                            task_dataset, model_type=model_type
                        )
                        task_results[model_type] = cv_results
                        
                    except Exception as e:
                        self.logger.error(f"训练 {task_name}-{model_type} 时出错: {str(e)}")
                        continue
            
            if task_results:
                all_results[task_name] = task_results
                self.task_results[task_name] = task_results
                
                # 找出该任务的最佳模型
                best_model_type = self._find_best_model_for_task(task_results)
                if best_model_type:
                    # 保存最佳模型（使用第一折作为代表）
                    best_model = task_results[best_model_type]['models'][0]
                    self.task_models[task_name] = {
                        'model': best_model,
                        'model_type': best_model_type,
                        'metrics': task_results[best_model_type]['average_metrics']
                    }
        
        # 打印总体结果摘要
        self._print_overall_summary()
        
        return all_results
    
    def _find_best_model_for_task(self, task_results: Dict) -> Optional[str]:
        """
        为单个任务找出最佳模型
        
        Args:
            task_results: 任务结果字典
            
        Returns:
            最佳模型类型
        """
        best_model_type = None
        best_mae = float('inf')
        
        for model_type, results in task_results.items():
            mae = results['average_metrics']['mae']
            if mae < best_mae:
                best_mae = mae
                best_model_type = model_type
        
        return best_model_type
    
    def _print_overall_summary(self):
        """打印总体结果摘要"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("所有任务训练完成 - 结果摘要")
        self.logger.info(f"{'='*60}")
        
        for task_name, model_info in self.task_models.items():
            metrics = model_info['metrics']
            model_type = model_info['model_type']
            
            self.logger.info(f"\n{task_name} 最佳模型: {model_type}")
            self.logger.info(f"  MAE: {metrics['mae']:.6f} ± {metrics['mae_std']:.6f}")
            self.logger.info(f"  RMSE: {metrics['rmse']:.6f} ± {metrics['rmse_std']:.6f}")
        
        # 计算总体平均MAE
        if self.task_models:
            overall_mae = np.mean([info['metrics']['mae'] for info in self.task_models.values()])
            self.logger.info(f"\n总体平均MAE: {overall_mae:.6f}")
        
        self.logger.info(f"{'='*60}")
    
    def predict_all_tasks(self, X_test: np.ndarray, 
                         feature_scalers: Dict[str, object] = None) -> Dict[str, np.ndarray]:
        """
        使用所有任务的最佳模型进行预测
        
        Args:
            X_test: 测试特征
            feature_scalers: 特征标准化器字典（可选）
            
        Returns:
            各任务的预测结果字典
        """
        predictions = {}
        
        for task_name, model_info in self.task_models.items():
            try:
                model = model_info['model']
                
                # 如果提供了对应任务的scaler，使用它来标准化特征
                X_test_scaled = X_test
                if feature_scalers and task_name in feature_scalers:
                    X_test_scaled = feature_scalers[task_name].transform(X_test)
                
                # 预测
                y_pred = model.predict(X_test_scaled)
                
                # 确保预测结果是一维的
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.ravel()
                
                predictions[task_name] = y_pred
                
                self.logger.info(f"{task_name} 预测完成: {len(y_pred)} 个样本")
                
            except Exception as e:
                self.logger.error(f"预测 {task_name} 时出错: {str(e)}")
                # 如果预测失败，使用0填充
                predictions[task_name] = np.zeros(len(X_test))
        
        return predictions
    
    def save_models(self, save_dir: Union[str, Path]):
        """
        保存所有训练好的模型
        
        Args:
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存每个任务的最佳模型
        for task_name, model_info in self.task_models.items():
            model_path = save_dir / f"{task_name}_best_model.pkl"
            model_info['model'].save_model(model_path)
            
            # 保存模型信息
            info_path = save_dir / f"{task_name}_model_info.pkl"
            with open(info_path, 'wb') as f:
                info_to_save = {
                    'model_type': model_info['model_type'],
                    'metrics': model_info['metrics'],
                    'task_name': task_name
                }
                pickle.dump(info_to_save, f)
        
        # 保存训练结果摘要
        summary_path = save_dir / "training_summary.pkl"
        with open(summary_path, 'wb') as f:
            pickle.dump(self.task_results, f)
        
        self.logger.info(f"所有模型已保存至: {save_dir}")
    
    def load_models(self, load_dir: Union[str, Path]):
        """
        加载所有训练好的模型
        
        Args:
            load_dir: 模型目录
        """
        load_dir = Path(load_dir)
        
        self.task_models = {}
        
        for task_name in self.supported_tasks:
            model_path = load_dir / f"{task_name}_best_model.pkl"
            info_path = load_dir / f"{task_name}_model_info.pkl"
            
            if model_path.exists() and info_path.exists():
                try:
                    # 加载模型
                    model = BaselineModel.load_model(model_path)
                    
                    # 加载模型信息
                    with open(info_path, 'rb') as f:
                        model_info = pickle.load(f)
                    
                    self.task_models[task_name] = {
                        'model': model,
                        'model_type': model_info['model_type'],
                        'metrics': model_info['metrics']
                    }
                    
                    self.logger.info(f"成功加载 {task_name} 模型")
                    
                except Exception as e:
                    self.logger.error(f"加载 {task_name} 模型失败: {str(e)}")
        
        self.logger.info(f"共加载了 {len(self.task_models)} 个任务模型")
    
    def get_training_summary(self) -> Dict:
        """
        获取训练摘要
        
        Returns:
            训练摘要字典
        """
        summary = {
            'tasks_trained': list(self.task_models.keys()),
            'task_count': len(self.task_models),
            'best_models': {}
        }
        
        for task_name, model_info in self.task_models.items():
            summary['best_models'][task_name] = {
                'model_type': model_info['model_type'],
                'mae': model_info['metrics']['mae'],
                'rmse': model_info['metrics']['rmse']
            }
        
        if self.task_models:
            summary['overall_mae'] = np.mean([
                info['metrics']['mae'] for info in self.task_models.values()
            ])
        
        return summary