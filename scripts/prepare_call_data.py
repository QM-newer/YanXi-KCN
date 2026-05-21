"""
来电数据预处理脚本
================
参考 RAG-CITY scripts/preprocess.py 设计

功能:
1. 加载来电通话记录数据
2. 数据清洗与规范化
3. 输出 Parquet 格式供后续索引构建
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.loader import load_json, load_call_records
from src.data.cleaner import clean_call_text, normalize_category
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_call_records(records: list) -> pd.DataFrame:
    """
    预处理通话记录

    Args:
        records: 原始通话记录列表

    Returns:
        清洗后的 DataFrame
    """
    processed = []

    for record in records:
        text = record.get("transcript", "") or record.get("text", "")
        if not text:
            continue

        # 清洗文本
        cleaned_text = clean_call_text(text)
        if not cleaned_text or len(cleaned_text) < 3:
            continue

        # 规范化分类
        category = normalize_category(record.get("category", "其他"))

        processed.append({
            "call_id": record.get("id", ""),
            "text": cleaned_text,
            "original_text": text,
            "category": category,
            "summary": record.get("summary", ""),
            "response": record.get("response", ""),
            "is_risky": record.get("is_risky", False),
            "risk_type": record.get("risk_type", ""),
            "confidence": record.get("confidence", 0.0),
            "timestamp": record.get("timestamp", ""),
            "duration": record.get("duration", 0),
            "phone_number": record.get("phone_number", ""),
            "caller_name": record.get("caller_name", ""),
            "tags": "|".join(record.get("tags", [])) if record.get("tags") else "",
        })

    df = pd.DataFrame(processed)

    # 去重
    df = df.drop_duplicates(subset=["text"], keep="first")

    logger.info(f"预处理完成: {len(df)} 条记录 (去重后)")

    return df


def generate_category_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    按类别聚合，生成类别摘要统计

    Args:
        df: 通话记录 DataFrame

    Returns:
        类别统计 DataFrame
    """
    category_stats = df.groupby("category").agg({
        "call_id": "count",
        "is_risky": ["sum", "mean"],
        "confidence": "mean"
    }).reset_index()

    category_stats.columns = [
        "category", "total_count", "risky_count",
        "risky_ratio", "avg_confidence"
    ]

    return category_stats


def main():
    parser = argparse.ArgumentParser(description="来电数据预处理")
    parser.add_argument("--data-dir", "-d", default="data", help="数据目录")
    parser.add_argument("--input", "-i", default=None, help="输入文件路径")
    parser.add_argument("--output-dir", "-o", default="data/processed", help="输出目录")
    parser.add_argument("--min-length", type=int, default=3, help="最小文本长度")
    parser.add_argument("--max-length", type=int, default=500, help="最大文本长度")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("来电数据预处理")
    logger.info("=" * 60)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    if args.input:
        # 指定输入文件
        records = load_json(args.input)
        logger.info(f"从 {args.input} 加载 {len(records)} 条记录")
    else:
        # 从数据目录加载 (使用绝对路径)
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / args.data_dir
        records = load_json(str(data_dir / "call_records.json"))

        # 补充其他数据源
        for extra_file in ["call_logs.json", "delivery_call_records.json"]:
            extra_path = data_dir / extra_file
            if extra_path.exists():
                extra = load_json(str(extra_path))
                records.extend(extra)
                logger.info(f"补充加载 {len(extra)} 条 from {extra_file}")

    logger.info(f"共加载 {len(records)} 条原始记录")

    # 预处理
    df = preprocess_call_records(records)

    # 过滤长度
    df = df[
        (df["text"].str.len() >= args.min_length) &
        (df["text"].str.len() <= args.max_length)
    ]

    # 保存主数据
    output_path = output_dir / "call_records.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"保存主数据到 {output_path}")

    # 生成类别统计
    category_stats = generate_category_summaries(df)
    stats_path = output_dir / "category_stats.csv"
    category_stats.to_csv(stats_path, index=False, encoding="utf-8")
    logger.info(f"保存类别统计到 {stats_path}")

    # 打印统计
    logger.info("=" * 60)
    logger.info("预处理统计")
    logger.info("=" * 60)
    logger.info(f"总记录数: {len(df)}")
    logger.info(f"风险记录: {df['is_risky'].sum()}")
    logger.info("\n各类别统计:")
    for _, row in category_stats.iterrows():
        logger.info(f"  {row['category']}: {row['total_count']}条 "
                   f"(风险比例: {row['risky_ratio']:.1%})")

    logger.info("=" * 60)
    logger.info("预处理完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
