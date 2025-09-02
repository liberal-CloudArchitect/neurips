"""
基线模型训练器
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import time

from .baseline import BaselineModel
from ..utils.metrics import calculate_multi_target_metrics, print_metrics


class BaselineTrainer:
    """基线模型训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.models_config = config['models']['baseline']
        self.validation_config = config['data']['validation']
        self.output_config = config['output']
        
        self.logger = logging.getLogger(__name__)
        
        # 存储训练结果
        self.results = {}
        self.best_models = {}
        
    def train_single_model(self, 
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None,
                          model_type: str = 'xgboost',
                          target_names: Optional[List[str]] = None) -> BaselineModel:
        """
        训练单个模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            model_type: 模型类型
            target_names: 目标名称列表
            
        Returns:
            训练好的模型
        """
        self.logger.info(f"开始训练 {model_type} 模型...")
        
        # 获取模型参数
        model_params = self.models_config[model_type].copy()
        
        # 创建并训练模型
        model = BaselineModel(
            model_type=model_type,
            model_params=model_params,
            use_multioutput=True
        )
        
        start_time = time.time()
        model.fit(X_train, y_train, X_val, y_val, target_names=target_names)
        training_time = time.time() - start_time
        
        self.logger.info(f"{model_type} 模型训练完成，耗时 {training_time:.2f} 秒")
        
        return model
    
    def evaluate_model(self, 
                      model: BaselineModel,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      target_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            target_names: 目标名称列表
            
        Returns:
            评估指标字典
        """
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = calculate_multi_target_metrics(y_test, y_pred, target_names)
        
        return metrics
    
    def cross_validate_model(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_type: str = 'xgboost',
                           target_names: Optional[List[str]] = None,
                           cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> Dict:
        """
        交叉验证模型
        
        Args:
            X: 特征数据
            y: 目标数据
            model_type: 模型类型
            target_names: 目标名称列表
            cv_splits: 预定义的交叉验证分割
            
        Returns:
            交叉验证结果
        """
        self.logger.info(f"开始 {model_type} 模型的交叉验证...")
        
        if target_names is None:
            target_names = [f'target_{i}' for i in range(y.shape[1])]
        
        # 创建交叉验证分割
        if cv_splits is None:
            kf = KFold(
                n_splits=self.validation_config['n_splits'],
                shuffle=self.validation_config['shuffle'],
                random_state=self.validation_config['random_state']
            )
            cv_splits = list(kf.split(X, y))
        
        # 存储每折的结果
        fold_results = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            self.logger.info(f"训练第 {fold + 1} 折...")
            
            # 分割数据
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # 训练模型
            model = self.train_single_model(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                model_type=model_type,
                target_names=target_names
            )
            
            # 评估模型
            metrics = self.evaluate_model(model, X_val_fold, y_val_fold, target_names)
            
            fold_results.append(metrics)
            fold_models.append(model)
            
            # 打印本折结果
            print_metrics(metrics, f"第 {fold + 1} 折验证结果")
        
        # 计算平均结果
        avg_metrics = self._average_cv_results(fold_results, target_names)
        
        # 打印平均结果
        print_metrics(avg_metrics, f"{model_type} 交叉验证平均结果")
        
        cv_results = {
            'model_type': model_type,
            'fold_results': fold_results,
            'average_metrics': avg_metrics,
            'models': fold_models,
            'cv_splits': cv_splits
        }
        
        return cv_results
    
    def _average_cv_results(self, 
                           fold_results: List[Dict],
                           target_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        计算交叉验证的平均结果
        
        Args:
            fold_results: 每折的结果列表
            target_names: 目标名称列表
            
        Returns:
            平均结果字典
        """
        avg_metrics = {}
        
        # 为每个目标计算平均指标
        for target_name in target_names:
            target_metrics = {}
            for metric_name in ['mae', 'rmse', 'mape']:
                values = [fold[target_name][metric_name] for fold in fold_results]
                target_metrics[metric_name] = np.mean(values)
                target_metrics[f'{metric_name}_std'] = np.std(values)
            avg_metrics[target_name] = target_metrics
        
        # 计算总体平均指标
        overall_metrics = {}
        for metric_name in ['mean_mae', 'mean_rmse', 'mean_mape', 'total_mae']:
            if metric_name in fold_results[0]['overall']:
                values = [fold['overall'][metric_name] for fold in fold_results]
                overall_metrics[metric_name] = np.mean(values)
                overall_metrics[f'{metric_name}_std'] = np.std(values)
        avg_metrics['overall'] = overall_metrics
        
        return avg_metrics
    
    def train_all_baseline_models(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                target_names: List[str],
                                cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> Dict:
        """
        训练所有基线模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            target_names: 目标名称列表
            cv_splits: 预定义的交叉验证分割
            
        Returns:
            所有模型的结果
        """
        self.logger.info("开始训练所有基线模型...")
        
        all_results = {}
        
        # 训练XGBoost
        if 'xgboost' in self.models_config:
            xgb_results = self.cross_validate_model(
                X_train, y_train, 
                model_type='xgboost',
                target_names=target_names,
                cv_splits=cv_splits
            )
            all_results['xgboost'] = xgb_results
            self.results['xgboost'] = xgb_results
        
        # 训练LightGBM
        if 'lightgbm' in self.models_config:
            lgb_results = self.cross_validate_model(
                X_train, y_train,
                model_type='lightgbm', 
                target_names=target_names,
                cv_splits=cv_splits
            )
            all_results['lightgbm'] = lgb_results
            self.results['lightgbm'] = lgb_results
        
        # 找出最佳模型
        self._find_best_models(all_results)
        
        return all_results
    
    def _find_best_models(self, all_results: Dict):
        """
        找出每个目标的最佳模型
        
        Args:
            all_results: 所有模型的结果
        """
        self.logger.info("寻找最佳模型...")
        
        # 比较总体MAE来确定最佳模型
        best_overall_model = None
        best_overall_mae = float('inf')
        
        for model_type, results in all_results.items():
            overall_mae = results['average_metrics']['overall']['mean_mae']
            self.logger.info(f"{model_type} 总体平均MAE: {overall_mae:.6f}")
            
            if overall_mae < best_overall_mae:
                best_overall_mae = overall_mae
                best_overall_model = model_type
        
        self.best_models['overall'] = {
            'model_type': best_overall_model,
            'mae': best_overall_mae
        }
        
        self.logger.info(f"最佳整体模型: {best_overall_model} (MAE: {best_overall_mae:.6f})")
    
    def save_results(self, save_dir: Union[str, Path]):
        """
        保存训练结果
        
        Args:
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存结果摘要
        summary = {
            'best_models': self.best_models,
            'training_config': self.models_config,
            'validation_config': self.validation_config
        }
        
        # 保存详细结果和模型
        for model_type, results in self.results.items():
            # 保存模型（选择第一折作为代表）
            if results['models']:
                model_path = save_dir / f"{model_type}_model.pkl"
                results['models'][0].save_model(model_path)
            
            # 保存结果
            results_path = save_dir / f"{model_type}_results.pkl"
            import pickle
            with open(results_path, 'wb') as f:
                # 不保存模型对象，避免文件过大
                results_to_save = results.copy()
                results_to_save['models'] = None
                pickle.dump(results_to_save, f)
        
        # 保存摘要
        summary_path = save_dir / "training_summary.pkl"
        import pickle
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        
        self.logger.info(f"训练结果已保存至: {save_dir}")
    
    def get_best_model(self, model_type: str = None) -> BaselineModel:
        """
        获取最佳模型
        
        Args:
            model_type: 指定模型类型，如果为None则返回总体最佳模型
            
        Returns:
            最佳模型实例
        """
        if model_type is None:
            model_type = self.best_models['overall']['model_type']
        
        if model_type not in self.results:
            raise ValueError(f"模型类型 {model_type} 不存在")
        
        # 返回第一折的模型作为代表
        return self.results[model_type]['models'][0]