"""
日志工具
========
统一的日志管理
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# 日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 日志级别
LOG_LEVEL = logging.INFO


def get_logger(name: str, level: int = LOG_LEVEL) -> logging.Logger:
    """
    获取logger实例

    Args:
        name: logger名称，通常使用 __name__
        level: 日志级别

    Returns:
        Logger实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def setup_file_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    设置文件日志

    Args:
        name: logger名称
        log_dir: 日志目录

    Returns:
        Logger实例
    """
    logger = get_logger(name)

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 文件handler
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL)
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    return logger
