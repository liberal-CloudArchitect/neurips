"""
第五阶段：模型集成模块
整合基线模型、GNN模型和Transformer模型的预测能力
实现多种集成策略：简单平均、加权平均、Stacking等

Author: World-class ML Engineer & Kaggle Specialist
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import torch
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
warnings.filterwarnings('ignore')


class ModelEnsemble:
    """
    高级模型集成器
    
    支持多种集成策略：
    1. 简单平均 (Simple Average)
    2. 加权平均 (Weighted Average) 
    3. Stacking集成 (Stacking Ensemble)
    4. 动态加权 (Dynamic Weighting)
    """
    
    def __init__(self, 
                 target_columns: List[str],
                 ensemble_strategy: str = 'stacking',
                 meta_model_type: str = 'ridge',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        初始化模型集成器
        
        Args:
            target_columns: 目标属性列表
            ensemble_strategy: 集成策略 ('simple', 'weighted', 'stacking', 'dynamic')
            meta_model_type: 元模型类型 ('ridge', 'linear', 'rf', 'gbm')
            cv_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.target_columns = target_columns
        self.ensemble_strategy = ensemble_strategy
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.logger = logging.getLogger(__name__)
        
        # 存储各模型预测结果
        self.model_predictions = {
            'baseline': {},
            'gnn': {}, 
            'transformer': {}
        }
        
        # 存储真实标签
        self.true_labels = {}
        
        # 存储集成模型
        self.meta_models = {}
        self.ensemble_weights = {}
        
        # 存储性能指标
        self.model_performances = {}
        self.ensemble_performance = {}
        
        self.logger.info(f"初始化模型集成器: {ensemble_strategy} + {meta_model_type}")
    
    def load_model_predictions(self, 
                             baseline_path: Optional[Path] = None,
                             gnn_path: Optional[Path] = None,
                             transformer_path: Optional[Path] = None,
                             data_split: str = 'validation') -> None:
        """
        加载各模型的预测结果
        
        Args:
            baseline_path: 基线模型结果路径
            gnn_path: GNN模型结果路径  
            transformer_path: Transformer模型结果路径
            data_split: 数据分割类型 ('validation', 'test')
        """
        self.logger.info(f"加载模型预测结果 - {data_split}")
        
        # 默认路径
        if baseline_path is None:
            baseline_path = Path("results/multi_task_models")
        if gnn_path is None:
            gnn_path = Path("results/gnn_models")
        if transformer_path is None:
            transformer_path = Path("results/transformer_models")
        
        # 加载基线模型预测
        self._load_baseline_predictions(baseline_path, data_split)
        
        # 加载GNN模型预测
        self._load_gnn_predictions(gnn_path, data_split)
        
        # 加载Transformer模型预测
        self._load_transformer_predictions(transformer_path, data_split)
        
        self.logger.info("模型预测结果加载完成")
    
    def _load_baseline_predictions(self, path: Path, data_split: str) -> None:
        """加载基线模型预测结果"""
        try:
            # 尝试加载pickle格式的预测结果
            for task in self.target_columns:
                pred_file = path / f"{task}_predictions_{data_split}.pkl"
                if pred_file.exists():
                    with open(pred_file, 'rb') as f:
                        predictions = pickle.load(f)
                    self.model_predictions['baseline'][task] = predictions
                    self.logger.info(f"加载基线模型 {task} 预测: {len(predictions)} 样本")
                else:
                    self.logger.warning(f"未找到基线模型 {task} 预测文件: {pred_file}")
        except Exception as e:
            self.logger.error(f"加载基线模型预测失败: {e}")
    
    def _load_gnn_predictions(self, path: Path, data_split: str) -> None:
        """加载GNN模型预测结果"""
        try:
            # 尝试加载GNN模型的预测结果
            for task in self.target_columns:
                pred_file = path / f"gnn_{task}_predictions_{data_split}.pkl"
                if pred_file.exists():
                    with open(pred_file, 'rb') as f:
                        predictions = pickle.load(f)
                    self.model_predictions['gnn'][task] = predictions
                    self.logger.info(f"加载GNN模型 {task} 预测: {len(predictions)} 样本")
                else:
                    self.logger.warning(f"未找到GNN模型 {task} 预测文件: {pred_file}")
        except Exception as e:
            self.logger.error(f"加载GNN模型预测失败: {e}")
    
    def _load_transformer_predictions(self, path: Path, data_split: str) -> None:
        """加载Transformer模型预测结果"""
        try:
            # 尝试加载Transformer模型的预测结果
            for task in self.target_columns:
                pred_file = path / f"transformer_{task}_predictions_{data_split}.pkl"
                if pred_file.exists():
                    with open(pred_file, 'rb') as f:
                        predictions = pickle.load(f)
                    self.model_predictions['transformer'][task] = predictions
                    self.logger.info(f"加载Transformer模型 {task} 预测: {len(predictions)} 样本")
                else:
                    self.logger.warning(f"未找到Transformer模型 {task} 预测文件: {pred_file}")
        except Exception as e:
            self.logger.error(f"加载Transformer模型预测失败: {e}")
    
    def load_true_labels(self, labels: Dict[str, np.ndarray]) -> None:
        """
        加载真实标签
        
        Args:
            labels: 真实标签字典 {task: labels_array}
        """
        self.true_labels = labels.copy()
        self.logger.info(f"加载真实标签，任务数: {len(labels)}")
        for task, labels_array in labels.items():
            self.logger.info(f"任务 {task}: {len(labels_array)} 样本")
    
    def calculate_model_performances(self) -> Dict[str, Dict[str, float]]:
        """
        计算各模型的性能指标
        
        Returns:
            性能指标字典
        """
        self.logger.info("计算各模型性能指标")
        
        performances = {
            'baseline': {},
            'gnn': {},
            'transformer': {}
        }
        
        for model_type in ['baseline', 'gnn', 'transformer']:
            for task in self.target_columns:
                if (task in self.model_predictions[model_type] and 
                    task in self.true_labels):
                    
                    pred = self.model_predictions[model_type][task]
                    true = self.true_labels[task]
                    
                    # 确保长度一致
                    min_len = min(len(pred), len(true))
                    pred = pred[:min_len]
                    true = true[:min_len]
                    
                    mae = mean_absolute_error(true, pred)
                    rmse = np.sqrt(mean_squared_error(true, pred))
                    
                    performances[model_type][task] = {
                        'mae': mae,
                        'rmse': rmse,
                        'samples': min_len
                    }
                    
                    self.logger.info(f"{model_type} {task}: MAE={mae:.6f}, RMSE={rmse:.6f}")
        
        self.model_performances = performances
        return performances
    
    def simple_average_ensemble(self) -> Dict[str, np.ndarray]:
        """
        简单平均集成
        
        Returns:
            集成预测结果
        """
        self.logger.info("执行简单平均集成")
        
        ensemble_predictions = {}
        
        for task in self.target_columns:
            predictions = []
            model_names = []
            
            # 收集所有可用的预测
            for model_type in ['baseline', 'gnn', 'transformer']:
                if task in self.model_predictions[model_type]:
                    predictions.append(self.model_predictions[model_type][task])
                    model_names.append(model_type)
            
            if len(predictions) == 0:
                self.logger.warning(f"任务 {task} 没有可用的预测结果")
                continue
            
            # 确保所有预测长度一致
            min_len = min(len(pred) for pred in predictions)
            predictions = [pred[:min_len] for pred in predictions]
            
            # 简单平均
            ensemble_pred = np.mean(predictions, axis=0)
            ensemble_predictions[task] = ensemble_pred
            
            self.logger.info(f"任务 {task} 简单平均集成: {len(model_names)} 个模型")
        
        return ensemble_predictions
    
    def weighted_average_ensemble(self, 
                                weights: Optional[Dict[str, List[float]]] = None) -> Dict[str, np.ndarray]:
        """
        加权平均集成
        
        Args:
            weights: 权重字典 {task: [baseline_weight, gnn_weight, transformer_weight]}
        
        Returns:
            集成预测结果
        """
        self.logger.info("执行加权平均集成")
        
        if weights is None:
            weights = self._optimize_ensemble_weights()
        
        ensemble_predictions = {}
        
        for task in self.target_columns:
            predictions = []
            model_names = []
            task_weights = []
            
            # 收集所有可用的预测和对应权重
            weight_idx = 0
            for model_type in ['baseline', 'gnn', 'transformer']:
                if task in self.model_predictions[model_type]:
                    predictions.append(self.model_predictions[model_type][task])
                    model_names.append(model_type)
                    
                    if task in weights:
                        task_weights.append(weights[task][weight_idx])
                    else:
                        task_weights.append(1.0 / 3.0)  # 默认均等权重
                
                weight_idx += 1
            
            if len(predictions) == 0:
                self.logger.warning(f"任务 {task} 没有可用的预测结果")
                continue
            
            # 确保所有预测长度一致
            min_len = min(len(pred) for pred in predictions)
            predictions = [pred[:min_len] for pred in predictions]
            
            # 归一化权重
            task_weights = np.array(task_weights)
            task_weights = task_weights / task_weights.sum()
            
            # 加权平均
            ensemble_pred = np.average(predictions, axis=0, weights=task_weights)
            ensemble_predictions[task] = ensemble_pred
            
            self.logger.info(f"任务 {task} 加权平均集成: 权重 {task_weights}")
        
        return ensemble_predictions
    
    def stacking_ensemble(self, 
                         cv_folds: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Stacking集成学习
        
        Args:
            cv_folds: 交叉验证折数
        
        Returns:
            集成预测结果
        """
        self.logger.info("执行Stacking集成")
        
        if cv_folds is None:
            cv_folds = self.cv_folds
        
        ensemble_predictions = {}
        self.meta_models = {}
        
        for task in self.target_columns:
            # 收集基础模型预测
            base_predictions = []
            model_names = []
            
            for model_type in ['baseline', 'gnn', 'transformer']:
                if task in self.model_predictions[model_type]:
                    base_predictions.append(self.model_predictions[model_type][task])
                    model_names.append(model_type)
            
            if len(base_predictions) == 0 or task not in self.true_labels:
                self.logger.warning(f"任务 {task} 数据不足，跳过Stacking")
                continue
            
            # 确保所有预测和标签长度一致
            min_len = min(len(pred) for pred in base_predictions + [self.true_labels[task]])
            base_predictions = [pred[:min_len] for pred in base_predictions]
            true_labels = self.true_labels[task][:min_len]
            
            # 构建元特征矩阵
            X_meta = np.column_stack(base_predictions)
            y_meta = true_labels
            
            # 创建并训练元模型
            meta_model = self._create_meta_model()
            
            # 交叉验证训练元模型（检查样本数量）
            n_samples = len(X_meta)
            actual_cv_folds = min(cv_folds, n_samples)
            
            if actual_cv_folds < 2:
                self.logger.warning(f"任务 {task} 样本数太少 ({n_samples})，跳过交叉验证")
                cv_mae = 0.0
                cv_std = 0.0
            else:
                cv_scores = cross_val_score(
                    meta_model, X_meta, y_meta,
                    cv=actual_cv_folds, scoring='neg_mean_absolute_error'
                )
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
            
            # 在全部数据上训练最终元模型
            meta_model.fit(X_meta, y_meta)
            self.meta_models[task] = meta_model
            
            # 生成集成预测
            ensemble_pred = meta_model.predict(X_meta)
            ensemble_predictions[task] = ensemble_pred
            
            self.logger.info(f"任务 {task} Stacking: CV MAE={cv_mae:.6f}±{cv_std:.6f}")
        
        return ensemble_predictions
    
    def _create_meta_model(self):
        """创建元模型"""
        if self.meta_model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=self.random_state)
        elif self.meta_model_type == 'linear':
            return LinearRegression()
        elif self.meta_model_type == 'lasso':
            return Lasso(alpha=0.1, random_state=self.random_state)
        elif self.meta_model_type == 'elastic':
            return ElasticNet(alpha=0.1, random_state=self.random_state)
        elif self.meta_model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=100, 
                max_depth=5,
                random_state=self.random_state
            )
        elif self.meta_model_type == 'gbm':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"不支持的元模型类型: {self.meta_model_type}")
    
    def _optimize_ensemble_weights(self) -> Dict[str, List[float]]:
        """
        使用Optuna优化集成权重
        
        Returns:
            优化后的权重字典
        """
        self.logger.info("使用Optuna优化集成权重")
        
        optimized_weights = {}
        
        for task in self.target_columns:
            if (task not in self.true_labels or 
                not any(task in self.model_predictions[model] 
                       for model in ['baseline', 'gnn', 'transformer'])):
                continue
            
            # 创建优化目标函数
            def objective(trial):
                # 定义权重参数
                w_baseline = trial.suggest_float('w_baseline', 0.0, 1.0)
                w_gnn = trial.suggest_float('w_gnn', 0.0, 1.0)
                w_transformer = trial.suggest_float('w_transformer', 0.0, 1.0)
                
                weights = [w_baseline, w_gnn, w_transformer]
                
                # 收集预测
                predictions = []
                available_weights = []
                
                for i, model_type in enumerate(['baseline', 'gnn', 'transformer']):
                    if task in self.model_predictions[model_type]:
                        predictions.append(self.model_predictions[model_type][task])
                        available_weights.append(weights[i])
                
                if len(predictions) == 0:
                    return float('inf')
                
                # 确保长度一致
                min_len = min(len(pred) for pred in predictions + [self.true_labels[task]])
                predictions = [pred[:min_len] for pred in predictions]
                true_labels = self.true_labels[task][:min_len]
                
                # 归一化权重
                available_weights = np.array(available_weights)
                if available_weights.sum() == 0:
                    return float('inf')
                available_weights = available_weights / available_weights.sum()
                
                # 加权平均预测
                ensemble_pred = np.average(predictions, axis=0, weights=available_weights)
                
                # 计算MAE
                mae = mean_absolute_error(true_labels, ensemble_pred)
                return mae
            
            # 运行优化
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100, show_progress_bar=False)
            
            # 提取最优权重
            best_params = study.best_params
            optimized_weights[task] = [
                best_params.get('w_baseline', 0.0),
                best_params.get('w_gnn', 0.0),
                best_params.get('w_transformer', 0.0)
            ]
            
            self.logger.info(f"任务 {task} 优化权重: {optimized_weights[task]}, "
                           f"MAE: {study.best_value:.6f}")
        
        self.ensemble_weights = optimized_weights
        return optimized_weights
    
    def predict_ensemble(self, 
                        new_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        对新数据进行集成预测
        
        Args:
            new_predictions: 新的预测结果 {model_type: {task: predictions}}
        
        Returns:
            集成预测结果
        """
        self.logger.info("对新数据进行集成预测")
        
        ensemble_predictions = {}
        
        for task in self.target_columns:
            if self.ensemble_strategy == 'simple':
                predictions = []
                for model_type in ['baseline', 'gnn', 'transformer']:
                    if (model_type in new_predictions and 
                        task in new_predictions[model_type]):
                        predictions.append(new_predictions[model_type][task])
                
                if predictions:
                    min_len = min(len(pred) for pred in predictions)
                    predictions = [pred[:min_len] for pred in predictions]
                    ensemble_predictions[task] = np.mean(predictions, axis=0)
            
            elif self.ensemble_strategy == 'weighted':
                if task in self.ensemble_weights:
                    predictions = []
                    weights = []
                    
                    for i, model_type in enumerate(['baseline', 'gnn', 'transformer']):
                        if (model_type in new_predictions and 
                            task in new_predictions[model_type]):
                            predictions.append(new_predictions[model_type][task])
                            weights.append(self.ensemble_weights[task][i])
                    
                    if predictions:
                        min_len = min(len(pred) for pred in predictions)
                        predictions = [pred[:min_len] for pred in predictions]
                        weights = np.array(weights)
                        weights = weights / weights.sum()
                        ensemble_predictions[task] = np.average(predictions, axis=0, weights=weights)
            
            elif self.ensemble_strategy == 'stacking':
                if task in self.meta_models:
                    base_predictions = []
                    for model_type in ['baseline', 'gnn', 'transformer']:
                        if (model_type in new_predictions and 
                            task in new_predictions[model_type]):
                            base_predictions.append(new_predictions[model_type][task])
                    
                    if base_predictions:
                        min_len = min(len(pred) for pred in base_predictions)
                        base_predictions = [pred[:min_len] for pred in base_predictions]
                        X_meta = np.column_stack(base_predictions)
                        ensemble_predictions[task] = self.meta_models[task].predict(X_meta)
        
        return ensemble_predictions
    
    def evaluate_ensemble(self, 
                         ensemble_predictions: Dict[str, np.ndarray],
                         true_labels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        评估集成模型性能
        
        Args:
            ensemble_predictions: 集成预测结果
            true_labels: 真实标签（可选，默认使用已加载的标签）
        
        Returns:
            性能指标字典
        """
        if true_labels is None:
            true_labels = self.true_labels
        
        performance = {}
        
        for task in self.target_columns:
            if task in ensemble_predictions and task in true_labels:
                pred = ensemble_predictions[task]
                true = true_labels[task]
                
                min_len = min(len(pred), len(true))
                pred = pred[:min_len]
                true = true[:min_len]
                
                mae = mean_absolute_error(true, pred)
                rmse = np.sqrt(mean_squared_error(true, pred))
                
                performance[task] = {
                    'mae': mae,
                    'rmse': rmse,
                    'samples': min_len
                }
                
                self.logger.info(f"集成 {task}: MAE={mae:.6f}, RMSE={rmse:.6f}")
        
        # 计算平均指标
        if performance:
            avg_mae = np.mean([p['mae'] for p in performance.values()])
            avg_rmse = np.mean([p['rmse'] for p in performance.values()])
            performance['average'] = {
                'mae': avg_mae,
                'rmse': avg_rmse
            }
            self.logger.info(f"集成平均: MAE={avg_mae:.6f}, RMSE={avg_rmse:.6f}")
        
        self.ensemble_performance = performance
        return performance
    
    def save_ensemble_model(self, save_path: Path) -> None:
        """
        保存集成模型
        
        Args:
            save_path: 保存路径
        """
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存元模型
        if self.meta_models:
            with open(save_path / "meta_models.pkl", 'wb') as f:
                pickle.dump(self.meta_models, f)
        
        # 保存权重
        if self.ensemble_weights:
            with open(save_path / "ensemble_weights.pkl", 'wb') as f:
                pickle.dump(self.ensemble_weights, f)
        
        # 保存性能指标
        performance_data = {
            'model_performances': self.model_performances,
            'ensemble_performance': self.ensemble_performance,
            'ensemble_strategy': self.ensemble_strategy,
            'meta_model_type': self.meta_model_type
        }
        
        with open(save_path / "ensemble_performance.pkl", 'wb') as f:
            pickle.dump(performance_data, f)
        
        self.logger.info(f"集成模型保存至: {save_path}")
    
    def load_ensemble_model(self, load_path: Path) -> None:
        """
        加载集成模型
        
        Args:
            load_path: 加载路径
        """
        # 加载元模型
        meta_models_file = load_path / "meta_models.pkl"
        if meta_models_file.exists():
            with open(meta_models_file, 'rb') as f:
                self.meta_models = pickle.load(f)
        
        # 加载权重
        weights_file = load_path / "ensemble_weights.pkl"
        if weights_file.exists():
            with open(weights_file, 'rb') as f:
                self.ensemble_weights = pickle.load(f)
        
        # 加载性能指标
        performance_file = load_path / "ensemble_performance.pkl"
        if performance_file.exists():
            with open(performance_file, 'rb') as f:
                performance_data = pickle.load(f)
                self.model_performances = performance_data.get('model_performances', {})
                self.ensemble_performance = performance_data.get('ensemble_performance', {})
        
        self.logger.info(f"集成模型加载自: {load_path}")


def create_ensemble_model(target_columns: List[str],
                         ensemble_strategy: str = 'stacking',
                         meta_model_type: str = 'ridge',
                         **kwargs) -> ModelEnsemble:
    """
    创建模型集成器的工厂函数
    
    Args:
        target_columns: 目标属性列表
        ensemble_strategy: 集成策略
        meta_model_type: 元模型类型
        **kwargs: 其他参数
    
    Returns:
        ModelEnsemble实例
    """
    return ModelEnsemble(
        target_columns=target_columns,
        ensemble_strategy=ensemble_strategy,
        meta_model_type=meta_model_type,
        **kwargs
    )