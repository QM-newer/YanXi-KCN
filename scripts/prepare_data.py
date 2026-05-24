"""
数据准备脚本
============
准备来电训练数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.utils.logger import get_logger
from src.data.loader import load_call_records, load_contacts, load_training_data
from src.data.cleaner import clean_call_text

logger = get_logger(__name__)


def prepare_training_data(data_dir: str = "data") -> dict:
    """准备训练数据"""
    stats = {
        "call_records": 0,
        "contacts": 0,
        "training_data": {},
        "cleaned_texts": 0
    }

    logger.info("加载通话记录...")
    records = load_call_records(data_dir)
    stats["call_records"] = len(records)

    logger.info("清洗文本...")
    cleaned = []
    for record in records:
        text = clean_call_text(record.text)
        if text:
            cleaned.append({
                "id": record.id,
                "text": text,
                "category": record.category,
                "sub_category": record.sub_category
            })
            stats["cleaned_texts"] += 1

    logger.info(f"清洗后: {stats['cleaned_texts']} 条")

    logger.info("加载通讯录...")
    contacts = load_contacts(data_dir)
    stats["contacts"] = len(contacts)

    logger.info("加载训练数据...")
    training_data = load_training_data(data_dir)
    stats["training_data"] = {k: len(v) for k, v in training_data.items()}

    return stats


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="数据准备")
    parser.add_argument("--data-dir", "-d", default="data", help="数据目录")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("开始数据准备...")
    logger.info("=" * 60)

    stats = prepare_training_data(args.data_dir)

    logger.info("=" * 60)
    logger.info("数据准备完成!")
    logger.info("=" * 60)
    logger.info(f"通话记录: {stats['call_records']}")
    logger.info(f"清洗后文本: {stats['cleaned_texts']}")
    logger.info(f"通讯录: {stats['contacts']}")
    logger.info("训练数据分布:")
    for cat, count in stats["training_data"].items():
        logger.info(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
