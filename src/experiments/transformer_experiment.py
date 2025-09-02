"""
Transformer实验脚本 - 第四阶段：化学Transformer模型开发
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Optional, List, Tuple
import torch
import argparse
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.multi_task_preprocessor import MultiTaskPreprocessor
from src.models.transformer_trainer import TransformerTrainer
from src.utils.metrics import calculate_multi_target_metrics


def update_config_for_transformer(config: Dict) -> Dict:
    """更新配置以适配Transformer模型"""
    
    # 添加Transformer相关配置
    if 'transformer' not in config.get('models', {}):
        config['models']['transformer'] = {
            # 模型架构
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'pooling_strategy': 'cls',
            
            # 分词器配置
            'vocab_size': 1000,
            'max_seq_length': 512,
            
            # 预训练模型配置
            'pretrained_model_name': 'DeepChem/ChemBERTa-77M-MLM',
            'freeze_encoder': False,
            'num_layers_to_freeze': 0,
            
            # 任务配置
            'task_names': ['Density', 'Tc', 'Tg', 'Rg', 'FFV']
        }
    
    # 更新训练配置
    if 'training' not in config:
        config['training'] = {}
    
    config['training'].update({
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 16,  # Transformer通常需要较小的batch size
        'epochs': 50,
        'learning_rate': 1e-4,  # Transformer通常需要较小的学习率
        'weight_decay': 1e-5,
        'scheduler_patience': 5,
        'early_stopping': {
            'patience': 10,
            'min_delta': 1e-6
        },
        'loss': {
            'type': 'smooth_l1',
            'beta': 1.0
        }
    })
    
    return config


def run_transformer_experiment(config_path: str = None,
                              data_dir: str = "data",
                              model_type: str = "custom",
                              max_samples: Optional[int] = None,
                              use_pretrained: bool = False):
    """
    运行Transformer实验
    
    Args:
        config_path: 配置文件路径
        data_dir: 数据目录路径
        model_type: 模型类型 ('custom', 'bert_based', 'with_pretraining')
        max_samples: 最大样本数（用于快速测试）
        use_pretrained: 是否使用预训练模型
    """
    # 加载和更新配置
    config = load_config(config_path)
    config = update_config_for_transformer(config)
    
    # 如果使用预训练模型，调整模型类型
    if use_pretrained:
        model_type = 'bert_based'
    
    # 设置日志
    logger = setup_logger(
        name="transformer_experiment",
        level=config['experiment']['logging']['level'],
        log_dir=config['experiment']['logging']['log_dir'] if config['experiment']['logging']['save_logs'] else None,
        log_to_file=config['experiment']['logging']['save_logs']
    )
    
    logger.info("=" * 80)
    logger.info(f"开始第四阶段Transformer实验 - {model_type.upper()}模型")
    logger.info("=" * 80)
    
    # 设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # 1. 数据加载和预处理
        logger.info("步骤 1: 加载竞赛数据")
        preprocessor = MultiTaskPreprocessor(config)
        
        # 加载所有数据
        datasets = preprocessor.load_competition_data(data_dir)
        analysis = preprocessor.analyze_data_coverage(datasets)
        
        # 2. 准备任务特定数据集
        logger.info("步骤 2: 准备任务特定数据集")
        
        # 首先准备所有任务的数据集
        preprocessor.prepare_task_specific_datasets(datasets)
        
        task_datasets = {}
        
        for task in config['data']['target_columns']:
            logger.info(f"\n准备任务: {task}")
            
            # 检查任务数据是否存在
            if task not in preprocessor.task_datasets or preprocessor.task_datasets[task].empty:
                logger.warning(f"任务 {task} 没有数据，跳过")
                continue
            
            # 获取任务数据
            task_df = preprocessor.task_datasets[task].copy()
            
            # 移除无效的SMILES
            valid_mask = task_df['SMILES'].apply(
                lambda x: preprocessor.feature_extractor.smiles_to_mol(x) is not None
            )
            task_df = task_df[valid_mask].reset_index(drop=True)
            
            if len(task_df) == 0:
                logger.warning(f"任务 {task} 没有有效的SMILES数据，跳过")
                continue
            
            # 准备Transformer所需的数据格式
            smiles_list = task_df['SMILES'].tolist()
            targets = task_df[task].values
            
            # 限制样本数（用于快速测试）
            if max_samples and len(smiles_list) > max_samples:
                indices = np.random.choice(len(smiles_list), max_samples, replace=False)
                smiles_list = [smiles_list[i] for i in indices]
                targets = targets[indices]
                logger.info(f"限制样本数为 {max_samples}")
            
            # 标准化目标值
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            targets_scaled = scaler.fit_transform(targets.reshape(-1, 1)).ravel()
            
            task_datasets[task] = {
                'smiles': smiles_list,
                'targets': targets_scaled,
                'scaler': scaler
            }
            
            logger.info(f"任务 {task}: {len(smiles_list)} 个样本")
        
        if not task_datasets:
            logger.error("没有有效的任务数据")
            return
        
        # 3. 初始化Transformer训练器
        logger.info("步骤 3: 初始化Transformer训练器")
        trainer = TransformerTrainer(config)
        
        # 4. 训练Transformer模型
        logger.info(f"步骤 4: 开始训练 {model_type.upper()} Transformer模型")
        
        # 准备训练数据格式
        train_data = {}
        for task, data in task_datasets.items():
            train_data[task] = (data['smiles'], data['targets'])
        
        # 训练模型
        trained_models = trainer.train_multi_task_models(train_data, model_type)
        
        if not trained_models:
            logger.error("没有成功训练的模型")
            return
        
        # 5. 模型性能评估
        logger.info("步骤 5: 模型性能评估")
        results_summary = []
        
        for task, (model, history) in trained_models.items():
            best_val_mae = min(history['val_mae'])
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            
            logger.info(f"\n任务 {task} 结果:")
            logger.info(f"  最佳验证MAE: {best_val_mae:.6f}")
            logger.info(f"  最终训练损失: {final_train_loss:.6f}")
            logger.info(f"  最终验证损失: {final_val_loss:.6f}")
            
            results_summary.append({
                'Task': task,
                'Model': f'{model_type.upper()}_Transformer',
                'Best_Val_MAE': best_val_mae,
                'Final_Train_Loss': final_train_loss,
                'Final_Val_Loss': final_val_loss,
                'Num_Epochs': len(history['train_loss']),
                'Model_Type': model_type
            })
        
        # 6. 保存结果
        logger.info("步骤 6: 保存实验结果")
        
        # 创建结果目录
        results_dir = Path(config['output']['results_dir']) / 'transformer_models'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型和分词器
        models_to_save = {task: model for task, (model, _) in trained_models.items()}
        trainer.save_models_and_tokenizer(models_to_save, str(results_dir))
        
        # 保存实验报告
        results_df = pd.DataFrame(results_summary)
        report_path = results_dir / f"transformer_{model_type}_experiment_report.csv"
        results_df.to_csv(report_path, index=False)
        logger.info(f"实验报告已保存: {report_path}")
        
        # 保存训练历史
        history_path = results_dir / f"transformer_{model_type}_training_history.pkl"
        import pickle
        with open(history_path, 'wb') as f:
            pickle.dump({task: history for task, (_, history) in trained_models.items()}, f)
        logger.info(f"训练历史已保存: {history_path}")
        
        # 7. 生成测试集预测（如果需要）
        logger.info("步骤 7: 生成测试集预测")
        test_data_path = Path(data_dir) / "test.csv"
        
        if test_data_path.exists():
            test_df = pd.read_csv(test_data_path)
            test_smiles = test_df[config['data']['smiles_column']].tolist()
            
            # 为每个任务生成预测
            test_predictions = {}
            for task, (model, _) in trained_models.items():
                logger.info(f"预测任务: {task}")
                predictions = trainer.predict(model, test_smiles, task)
                
                # 反标准化
                if task in task_datasets:
                    scaler = task_datasets[task]['scaler']
                    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                
                test_predictions[task] = predictions
            
            # 保存预测结果
            submission_df = pd.DataFrame({
                'id': test_df['id'],
                **test_predictions
            })
            
            submission_path = results_dir / f"transformer_{model_type}_submission.csv"
            submission_df.to_csv(submission_path, index=False)
            logger.info(f"提交文件已保存: {submission_path}")
        
        # 8. 总结
        logger.info("\n" + "=" * 80)
        logger.info("Transformer实验完成总结")
        logger.info("=" * 80)
        
        total_mae = np.mean([result['Best_Val_MAE'] for result in results_summary])
        logger.info(f"总体平均MAE: {total_mae:.6f}")
        
        logger.info("\n各任务最佳性能:")
        for result in results_summary:
            logger.info(f"  {result['Task']}: MAE {result['Best_Val_MAE']:.6f}")
        
        logger.info(f"\n模型类型: {model_type.upper()} Transformer")
        logger.info(f"设备: {device}")
        logger.info(f"训练样本总数: {sum(len(data['smiles']) for data in task_datasets.values())}")
        logger.info(f"成功训练任务数: {len(trained_models)}")
        logger.info(f"分词器词汇表大小: {trainer.tokenizer.get_vocab_size() if trainer.tokenizer else 'N/A'}")
        
        logger.info("\n实验文件保存位置:")
        logger.info(f"  模型文件: {results_dir}/")
        logger.info(f"  分词器: {results_dir}/tokenizer_vocab.json")
        logger.info(f"  实验报告: {report_path}")
        logger.info(f"  训练历史: {history_path}")
        if test_data_path.exists():
            logger.info(f"  提交文件: {submission_path}")
        
        return trained_models, results_summary
        
    except Exception as e:
        logger.error(f"实验执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def compare_transformer_models(config_path: str = None,
                              data_dir: str = "data",
                              models: List[str] = None,
                              max_samples: Optional[int] = None):
    """
    比较不同Transformer模型的性能
    
    Args:
        config_path: 配置文件路径
        data_dir: 数据目录路径
        models: 要比较的模型列表
        max_samples: 最大样本数
    """
    if models is None:
        models = ['custom', 'bert_based']
    
    logger = setup_logger(name="transformer_comparison")
    logger.info("开始Transformer模型对比实验")
    
    all_results = {}
    
    for model_type in models:
        logger.info(f"\n{'='*20} 训练 {model_type.upper()} 模型 {'='*20}")
        
        try:
            use_pretrained = (model_type == 'bert_based')
            trained_models, results = run_transformer_experiment(
                config_path=config_path,
                data_dir=data_dir,
                model_type=model_type,
                max_samples=max_samples,
                use_pretrained=use_pretrained
            )
            all_results[model_type] = results
            
        except Exception as e:
            logger.error(f"{model_type} 模型训练失败: {e}")
            continue
    
    # 生成对比报告
    if len(all_results) > 1:
        logger.info("\n" + "="*50)
        logger.info("Transformer模型对比总结")
        logger.info("="*50)
        
        comparison_data = []
        for model_type, results in all_results.items():
            avg_mae = np.mean([r['Best_Val_MAE'] for r in results])
            comparison_data.append({
                'Model': f'{model_type.upper()}_Transformer',
                'Avg_MAE': avg_mae,
                'Num_Tasks': len(results)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Avg_MAE')
        
        logger.info("\n模型性能排名:")
        for i, row in comparison_df.iterrows():
            logger.info(f"  {row['Model']}: 平均MAE {row['Avg_MAE']:.6f}")
        
        # 保存对比结果
        results_dir = Path("results/transformer_models")
        results_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = results_dir / "transformer_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\n对比结果已保存: {comparison_path}")


def run_quick_smiles_test():
    """快速测试SMILES分词器功能"""
    logger = setup_logger(name="smiles_test")
    logger.info("开始SMILES分词器测试")
    
    try:
        from src.data.smiles_tokenizer import test_smiles_tokenizer
        tokenizer = test_smiles_tokenizer()
        logger.info("✅ SMILES分词器测试完成")
        return tokenizer
    except Exception as e:
        logger.error(f"❌ SMILES分词器测试失败: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer实验脚本")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--model_type", type=str, default="custom", 
                       choices=['custom', 'bert_based', 'with_pretraining'], help="模型类型")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（测试用）")
    parser.add_argument("--compare", action="store_true", help="比较多个模型")
    parser.add_argument("--use_pretrained", action="store_true", help="使用预训练模型")
    parser.add_argument("--test_tokenizer", action="store_true", help="测试分词器")
    
    args = parser.parse_args()
    
    if args.test_tokenizer:
        run_quick_smiles_test()
    elif args.compare:
        compare_transformer_models(
            config_path=args.config,
            data_dir=args.data_dir,
            max_samples=args.max_samples
        )
    else:
        run_transformer_experiment(
            config_path=args.config,
            data_dir=args.data_dir,
            model_type=args.model_type,
            max_samples=args.max_samples,
            use_pretrained=args.use_pretrained
        )