"""
从现有 ChromaDB SQLite 中提取数据并重建向量库
==============================================
解决 ChromaDB 1.5.x 的 HNSW 索引不兼容问题
"""

import sys
import os
import sqlite3
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.indexing.embeddings import load_bge_embedding
from src.utils.config import load_config

logger = get_logger(__name__)


def extract_records_from_chroma(sqlite_path: str) -> list:
    """
    从 ChromaDB SQLite 文件中提取文档和元数据

    Args:
        sqlite_path: chroma.sqlite3 文件路径

    Returns:
        记录列表，每条包含 text 和 metadata
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    # 获取所有 embedding_id 和对应的文档
    cur = conn.execute("""
        SELECT id, string_value
        FROM embedding_metadata
        WHERE key = 'chroma:document'
        ORDER BY id
    """)
    docs = {row['id']: row['string_value'] for row in cur.fetchall()}

    # 获取每个 embedding_id 的所有元数据
    cur = conn.execute("""
        SELECT id, key, string_value
        FROM embedding_metadata
        WHERE key != 'chroma:document'
        ORDER BY id, key
    """)
    metadata_map = {}
    for row in cur.fetchall():
        eid = row['id']
        if eid not in metadata_map:
            metadata_map[eid] = {}
        metadata_map[eid][row['key']] = row['string_value']

    conn.close()

    records = []
    for eid, text in docs.items():
        meta = metadata_map.get(eid, {})
        records.append({
            "text": text,
            "id": meta.get("call_id", ""),
            "category": meta.get("category", ""),
            "sub_category": meta.get("sub_category", ""),
            "is_risky": meta.get("is_risky", "").lower() == "true" if meta.get("is_risky") else False,
            "risk_type": meta.get("risk_type", ""),
            "summary": meta.get("summary", ""),
            "response": meta.get("response", ""),
            "timestamp": meta.get("timestamp", ""),
            "tags": meta.get("tags", ""),
        })

    logger.info(f"从 SQLite 提取 {len(records)} 条记录")
    return records


def rebuild_vector_store(
    source_sqlite: str,
    target_dir: str,
    collection_name: str = "call_records"
):
    """
    重建向量库

    Args:
        source_sqlite: 源 chroma.sqlite3 路径
        target_dir: 目标向量库目录
        collection_name: 集合名称
    """
    # 提取数据
    records = extract_records_from_chroma(source_sqlite)
    if not records:
        logger.error("未提取到任何记录")
        return

    # 加载配置
    config = load_config()
    embed_cfg = config.get("embedding", {})

    # 加载 embedding 模型
    embedder = load_bge_embedding(
        model_name=embed_cfg.get("model", "BAAI/bge-large-zh-v1.5"),
        device=embed_cfg.get("device", "cpu"),
        batch_size=embed_cfg.get("batch_size", 32)
    )

    # 删除旧目标目录
    target_path = Path(target_dir)
    if target_path.exists():
        import shutil
        shutil.rmtree(target_path)
        logger.info(f"删除旧向量库: {target_dir}")

    # 构建新向量库
    from langchain_core.documents import Document
    from langchain_chroma import Chroma

    documents = []
    for record in records:
        doc = Document(
            page_content=record["text"],
            metadata={
                "call_id": record["id"],
                "category": record["category"],
                "sub_category": record.get("sub_category", ""),
                "is_risky": record.get("is_risky", False),
                "risk_type": record.get("risk_type", ""),
                "summary": record.get("summary", ""),
                "response": record.get("response", ""),
                "timestamp": record.get("timestamp", ""),
                "tags": record.get("tags", ""),
            }
        )
        documents.append(doc)

    logger.info(f"开始构建向量库: {len(documents)} 条文档 -> {target_dir}")

    t0 = time.time()
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        persist_directory=target_dir,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )

    elapsed = time.time() - t0
    count = db._collection.count()
    logger.info(f"向量库重建完成: {count} 条文档 (耗时: {elapsed:.1f}s)")

    # 验证
    results = db.similarity_search("外卖到了", k=3)
    logger.info(f"验证检索: '外卖到了' -> {len(results)} 条结果")
    if results:
        logger.info(f"  首条: {results[0].page_content[:60]}...")
        logger.info(f"  类别: {results[0].metadata.get('category', 'N/A')}")


if __name__ == "__main__":
    # 优先从 vector_store 本身读取（数据可能已在这里），
    # fallback 到 vector_store_v6
    candidates = [
        PROJECT_ROOT / "indices" / "vector_store" / "chroma.sqlite3",
        PROJECT_ROOT / "indices" / "vector_store_v6" / "chroma.sqlite3",
    ]
    source = None
    for c in candidates:
        if c.exists():
            source = c
            break
    if source is None:
        print(f"错误: 未找到 chroma.sqlite3，尝试过: {candidates}")
        sys.exit(1)
    # 如果源和目标相同，先备份再重建
    target = PROJECT_ROOT / "indices" / "vector_store"

    if not source.exists():
        print(f"错误: 源文件不存在: {source}")
        sys.exit(1)

    print(f"源: {source}")
    print(f"目标: {target}")
    print("=" * 60)

    rebuild_vector_store(
        source_sqlite=str(source),
        target_dir=str(target),
    )

    # 同时更新 config.yaml 路径
    print("\n请确保 config.yaml 中 paths.vector_store 指向正确的目录。")
    print("当前配置中已更新为 indices/vector_store_v6，")
    print("重建后可以使用 indices/vector_store（即 config.yaml 中的原始值）")
