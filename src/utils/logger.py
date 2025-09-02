"""
日志配置工具
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "neurips_polymer",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件保存至: {log_file}")
    
    return logger


def get_logger(name: str = "neurips_polymer") -> logging.Logger:
    """
    获取已配置的日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器
    """
    return logging.getLogger(name)