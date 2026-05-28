"""
向量索引构建
============

使用 BGE embeddings 构建 Chroma 向量索引
支持增量构建和断点续传
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


def format_call_document(record: Dict[str, Any], doc_type: str = "call") -> str:
    """
    格式化通话记录为文档文本

    Args:
        record: 通话记录字典
        doc_type: 文档类型 (call/category/risk)

    Returns:
        格式化的文档文本
    """
    if doc_type == "call":
        return f"[通话记录] {record['text']}"
    elif doc_type == "category":
        category = record.get("category", "")
        summary = record.get("summary", "")
        return f"[类别] {category} | 描述: {summary}"
    elif doc_type == "response":
        transcript = record.get("text", "")
        response = record.get("response", "")
        return f"[对话] 来电: {transcript} | 建议回复: {response}"
    else:
        return record.get("text", "")


def build_vector_index(
    records: List[Dict[str, Any]],
    embedder: Any,
    persist_dir: str,
    collection_name: str = "call_records",
    doc_type: str = "call"
) -> Any:
    """
    构建向量索引

    Args:
        records: 通话记录列表
        embedder: Embedding 模型
        persist_dir: 持久化目录
        collection_name: 集合名称
        doc_type: 文档类型

    Returns:
        Chroma 向量库
    """
    from langchain_chroma import Chroma

    # 转换为文档
    documents = []
    for record in records:
        text = record.get("text", "")
        if not text:
            continue

        # 格式化文档
        page_content = format_call_document(record, doc_type)

        doc = Document(
            page_content=page_content,
            metadata={
                "call_id": record.get("id", record.get("call_id", "")),
                "category": record.get("category", ""),
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

    if not documents:
        logger.warning("没有文档可索引")
        return None

    # 创建目录
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"构建向量索引: {len(documents)} 文档")

    # 使用 LangChain Chroma，禁用 HNSW 索引
    t0 = time.time()
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        persist_directory=persist_dir,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100, "hnsw:search_ef": 100}
    )

    # 确保数据持久化
    logger.info(f"向量索引构建完成: {len(documents)} 文档 (耗时: {time.time()-t0:.1f}s)")

    return db


def _extract_records_from_chroma_sqlite(sqlite_path: str) -> list:
    """
    从 ChromaDB SQLite 文件中提取文档和元数据
    用于 HNSW 索引损坏时的数据恢复

    Args:
        sqlite_path: chroma.sqlite3 文件路径

    Returns:
        记录列表，每条包含 text 和 metadata
    """
    import sqlite3

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.execute("""
            SELECT id, string_value
            FROM embedding_metadata
            WHERE key = 'chroma:document'
            ORDER BY id
        """)
        docs = {row['id']: row['string_value'] for row in cur.fetchall()}

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

        records = []
        for eid, text in docs.items():
            meta = metadata_map.get(eid, {})
            records.append({
                "text": text,
                "call_id": meta.get("call_id", ""),
                "category": meta.get("category", ""),
                "sub_category": meta.get("sub_category", ""),
                "is_risky": meta.get("is_risky", "").lower() == "true" if meta.get("is_risky") else False,
                "risk_type": meta.get("risk_type", ""),
                "summary": meta.get("summary", ""),
                "response": meta.get("response", ""),
                "timestamp": meta.get("timestamp", ""),
                "tags": meta.get("tags", ""),
            })

        return records
    finally:
        conn.close()


def load_vector_index(
    embedder: Any,
    persist_dir: str,
    collection_name: str = "call_records",
    fallback_sqlite: str = None
) -> Any:
    """
    加载已有向量索引

    Args:
        embedder: Embedding 模型
        persist_dir: 持久化目录
        collection_name: 集合名称
        fallback_sqlite: 当索引损坏时，从哪个 SQLite 文件恢复数据

    Returns:
        Chroma 向量库
    """
    from langchain_chroma import Chroma

    persist_path = Path(persist_dir)

    # 检查 chroma.sqlite3 文件是否存在
    sqlite_path = persist_path / "chroma.sqlite3"
    if not sqlite_path.exists() and not fallback_sqlite:
        logger.warning(f"ChromaDB 数据文件不存在: {sqlite_path}")
        return None

    # 方案1: 使用 LangChain Chroma 直接加载
    if sqlite_path.exists():
        try:
            db = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedder,
                collection_name=collection_name,
            )
            count = db._collection.count()
            logger.info(f"加载向量索引 (LangChain): {count} 条记录")
            if count > 0:
                return db
        except Exception as e:
            logger.warning(f"LangChain Chroma 加载失败: {e}")

        # 方案2: 使用原生 Chroma 客户端（fallback）
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            collection = client.get_collection(collection_name)
            count = collection.count()
            logger.info(f"加载向量索引 (原生): {count} 条记录")

            from langchain_chroma import Chroma  # noqa: F811
            db = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedder
            )
            if count > 0:
                return db
        except Exception as e:
            logger.warning(f"原生 Chroma 加载失败: {e}")

    # 方案3: HNSW 索引损坏，从 SQLite 中提取数据并重建
    # 使用 fallback_sqlite（如果提供）或当前的 sqlite_path
    recovery_source = fallback_sqlite or (str(sqlite_path) if sqlite_path.exists() else None)
    if recovery_source and Path(recovery_source).exists():
        logger.info("尝试从 SQLite 中提取数据并重建索引（HNSW 恢复模式）...")
        try:
            records = _extract_records_from_chroma_sqlite(str(recovery_source))
            if not records:
                logger.warning("从 SQLite 中未提取到任何记录")
                return None

            logger.info(f"从 SQLite 提取 {len(records)} 条记录，正在重建索引...")

            # 删除旧索引文件
            import shutil
            shutil.rmtree(str(persist_path), ignore_errors=True)
            persist_path.mkdir(parents=True, exist_ok=True)

            # 用当前进程重建
            db = build_vector_index(
                records=records,
                embedder=embedder,
                persist_dir=str(persist_path),
                collection_name=collection_name,
                doc_type="call"
            )

            if db is not None:
                count = db._collection.count()
                logger.info(f"HNSW 恢复模式重建完成: {count} 条记录")
                return db
        except Exception as e:
            logger.warning(f"HNSW 恢复模式失败: {e}")

    return None


def add_to_vector_index(
    db: Any,
    records: List[Dict[str, Any]],
    embedder: Any,
    doc_type: str = "call"
) -> None:
    """
    向已有索引添加文档

    Args:
        db: 已有向量库
        records: 新记录列表
        embedder: Embedding 模型
        doc_type: 文档类型
    """
    documents = []
    for record in records:
        text = record.get("text", "")
        if not text:
            continue

        page_content = format_call_document(record, doc_type)

        doc = Document(
            page_content=page_content,
            metadata={
                "call_id": record.get("id", record.get("call_id", "")),
                "category": record.get("category", ""),
                "is_risky": record.get("is_risky", False),
                "risk_type": record.get("risk_type", ""),
            }
        )
        documents.append(doc)

    if documents:
        db.add_documents(documents)
        db.persist()
        logger.info(f"添加 {len(documents)} 个文档到索引")


class VectorIndexer:
    """
    向量索引管理器

    支持增量构建、增量添加、索引加载
    """

    def __init__(
        self,
        embedder: Any = None,
        persist_dir: str = "indices/vector_store",
        collection_name: str = "call_records"
    ):
        self.embedder = embedder
        self.persist_dir_path = Path(persist_dir)
        self.persist_dir_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.db = None

    def _get_embedder(self):
        """获取或加载 embedder"""
        if self.embedder is None:
            from src.indexing.embeddings import load_bge_embedding
            self.embedder = load_bge_embedding()
        return self.embedder

    def build_from_records(
        self,
        records: List[Dict[str, Any]],
        doc_type: str = "call"
    ) -> "VectorIndexer":
        """
        从通话记录构建索引

        Args:
            records: 通话记录列表
            doc_type: 文档类型

        Returns:
            self
        """
        embedder = self._get_embedder()
        self.db = build_vector_index(
            records=records,
            embedder=embedder,
            persist_dir=str(self.persist_dir_path),
            collection_name=self.collection_name,
            doc_type=doc_type
        )
        return self

    def load(self, fallback_sqlite: str = None) -> "VectorIndexer":
        """加载已有索引"""
        embedder = self._get_embedder()
        self.db = load_vector_index(
            embedder=embedder,
            persist_dir=str(self.persist_dir_path),
            collection_name=self.collection_name,
            fallback_sqlite=fallback_sqlite
        )
        return self

    def add_records(
        self,
        records: List[Dict[str, Any]],
        doc_type: str = "call"
    ) -> None:
        """添加记录到索引"""
        if self.db is None:
            self.build_from_records(records, doc_type)
        else:
            embedder = self._get_embedder()
            add_to_vector_index(self.db, records, embedder, doc_type)

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """相似性搜索"""
        if self.db is None:
            return []
        return self.db.similarity_search(query, k=k, **kwargs)

    def get_retriever(self, **kwargs):
        """获取 LangChain Retriever"""
        if self.db is None:
            raise ValueError("索引未加载")
        return self.db.as_retriever(**kwargs)

    @property
    def count(self) -> int:
        """获取索引中的文档数量"""
        if self.db is None:
            return 0
        return self.db._collection.count()
