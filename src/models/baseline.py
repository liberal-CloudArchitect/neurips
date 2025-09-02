"""
基线模型实现
"""

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Optional, Union, Tuple
import logging
import joblib
from pathlib import Path


class BaselineModel(BaseEstimator, RegressorMixin):
    """基线模型包装器"""
    
    def __init__(self, 
                 model_type: str = 'xgboost',
                 model_params: Optional[Dict] = None,
                 use_multioutput: bool = True):
        """
        初始化基线模型
        
        Args:
            model_type: 模型类型 ('xgboost' 或 'lightgbm')
            model_params: 模型参数
            use_multioutput: 是否使用多输出回归器
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.use_multioutput = use_multioutput
        self.models = {}
        self.is_fitted = False
        
        self.logger = logging.getLogger(__name__)
        
        # 根据模型类型设置默认参数
        if model_type == 'xgboost':
            default_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        elif model_type == 'lightgbm':
            default_params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 合并用户参数和默认参数
        default_params.update(self.model_params)
        self.model_params = default_params
    
    def _create_single_model(self):
        """创建单个模型实例"""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(**self.model_params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(**self.model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.DataFrame],
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            target_names: Optional[List[str]] = None,
            **fit_params) -> 'BaselineModel':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练目标
            X_val: 验证特征（可选）
            y_val: 验证目标（可选）
            target_names: 目标名称列表
            **fit_params: 额外的训练参数
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, pd.DataFrame):
            y_val = y_val.values
        
        self.n_targets = y.shape[1] if len(y.shape) > 1 else 1
        self.target_names = target_names or [f'target_{i}' for i in range(self.n_targets)]
        
        self.logger.info(f"开始训练 {self.model_type} 模型...")
        self.logger.info(f"训练数据形状: X={X.shape}, y={y.shape}")
        
        if self.use_multioutput and self.n_targets > 1:
            # 使用多输出回归器
            base_model = self._create_single_model()
            self.model = MultiOutputRegressor(base_model, n_jobs=1)
            
            # 准备早停参数
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                fit_params.update({
                    'estimator__eval_set': eval_set,
                    'estimator__early_stopping_rounds': 50,
                    'estimator__verbose': False
                })
            
            self.model.fit(X, y, **fit_params)
            
        else:
            # 为每个目标训练单独的模型
            self.models = {}
            for i in range(self.n_targets):
                target_name = self.target_names[i]
                self.logger.info(f"训练目标 {target_name} 的模型...")
                
                model = self._create_single_model()
                
                # 准备目标数据
                y_target = y[:, i] if len(y.shape) > 1 else y
                y_val_target = y_val[:, i] if y_val is not None and len(y_val.shape) > 1 else y_val
                
                # 准备早停参数
                if X_val is not None and y_val_target is not None:
                    if self.model_type == 'xgboost':
                        try:
                            model.fit(X, y_target, 
                                    eval_set=[(X_val, y_val_target)],
                                    early_stopping_rounds=50,
                                    verbose=False)
                        except TypeError:
                            # 新版本XGBoost的参数名称可能不同
                            model.fit(X, y_target)
                    elif self.model_type == 'lightgbm':
                        try:
                            model.fit(X, y_target,
                                    eval_set=[(X_val, y_val_target)],
                                    early_stopping_rounds=50,
                                    verbose=False)
                        except TypeError:
                            # 新版本LightGBM的参数名称可能不同
                            model.fit(X, y_target)
                else:
                    model.fit(X, y_target)
                
                self.models[target_name] = model
        
        self.is_fitted = True
        self.logger.info("模型训练完成")
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit 方法")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.use_multioutput and self.n_targets > 1:
            predictions = self.model.predict(X)
        else:
            predictions = []
            for target_name in self.target_names:
                pred = self.models[target_name].predict(X)
                predictions.append(pred)
            predictions = np.column_stack(predictions)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit 方法")
        
        importance_dict = {}
        
        if self.use_multioutput and self.n_targets > 1:
            for i, target_name in enumerate(self.target_names):
                estimator = self.model.estimators_[i]
                if hasattr(estimator, 'feature_importances_'):
                    importance_dict[target_name] = estimator.feature_importances_
        else:
            for target_name in self.target_names:
                model = self.models[target_name]
                if hasattr(model, 'feature_importances_'):
                    importance_dict[target_name] = model.feature_importances_
        
        return importance_dict
    
    def save_model(self, save_path: Union[str, Path]):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'use_multioutput': self.use_multioutput,
            'n_targets': self.n_targets,
            'target_names': self.target_names,
            'is_fitted': self.is_fitted
        }
        
        if self.use_multioutput and self.n_targets > 1:
            model_data['model'] = self.model
        else:
            model_data['models'] = self.models
        
        joblib.dump(model_data, save_path)
        self.logger.info(f"模型已保存至: {save_path}")
    
    @classmethod
    def load_model(cls, load_path: Union[str, Path]) -> 'BaselineModel':
        """
        加载模型
        
        Args:
            load_path: 模型路径
            
        Returns:
            加载的模型实例
        """
        model_data = joblib.load(load_path)
        
        instance = cls(
            model_type=model_data['model_type'],
            model_params=model_data['model_params'],
            use_multioutput=model_data['use_multioutput']
        )
        
        instance.n_targets = model_data['n_targets']
        instance.target_names = model_data['target_names']
        instance.is_fitted = model_data['is_fitted']
        
        if 'model' in model_data:
            instance.model = model_data['model']
        if 'models' in model_data:
            instance.models = model_data['models']
        
        return instance
    
    def get_params(self, deep: bool = True) -> Dict:
        """获取模型参数"""
        return {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'use_multioutput': self.use_multioutput
        }
    
    def set_params(self, **params) -> 'BaselineModel':
        """设置模型参数"""
        for key, value in params.items():
            setattr(self, key, value)
        return self