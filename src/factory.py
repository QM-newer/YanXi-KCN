"""
依赖注入工厂
============

统一管理所有组件的初始化
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.utils.config import load_config, get_path
from src.utils.logger import get_logger
from src.utils.llm_client import QwenClient
from src.indexing.vector_builder import VectorIndexer
from src.indexing.graph_builder import GraphBuilder
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.llm_router import LLMRouter
from src.retrieval.reranker import SimpleReranker
from src.retrieval.fusion import RRFFuser
from src.generation.answerer import Answerer

logger = get_logger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _abs(path: str) -> str:
    """转换为绝对路径"""
    p = Path(path)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


def create_vector_store(config: Dict[str, Any] | None = None) -> Optional[Any]:
    """
    创建向量存储

    Args:
        config: 配置字典

    Returns:
        VectorIndexer实例
    """
    config = config or load_config()
    persist_dir = _abs(config.get("paths", {}).get("vector_store", "indices/vector_store"))

    # 根据配置选择 Embedding 后端
    embed_cfg = config.get("embedding", {})
    backend = embed_cfg.get("backend", "bge")

    if backend == "dashscope":
        from src.indexing.embeddings import DashScopeEmbedder
        embedder = DashScopeEmbedder(batch_size=embed_cfg.get("batch_size", 25))
        logger.info("使用 DashScope Embedding API")
    else:
        from src.indexing.embeddings import load_bge_embedding
        embedder = load_bge_embedding(
            model_name=embed_cfg.get("model", "BAAI/bge-large-zh-v1.5"),
            device=embed_cfg.get("device", "cpu"),
            batch_size=embed_cfg.get("batch_size", 32)
        )
        logger.info(f"使用 BGE Embedding: {embed_cfg.get('model')}")

    indexer = VectorIndexer(persist_dir=persist_dir, embedder=embedder)

    # 尝试加载已有索引
    try:
        # 如果 vector_store 加载失败，尝试从 vector_store_v6 的 SQLite 恢复
        v6_sqlite = str(PROJECT_ROOT / config.get("paths", {}).get("vector_store_v6", "indices/vector_store_v6") / "chroma.sqlite3")
        indexer.load(fallback_sqlite=v6_sqlite)
        logger.info(f"加载向量索引: {persist_dir}")
    except Exception as e:
        logger.warning(f"向量索引加载失败: {e}")

    return indexer


def create_graph_store(config: Dict[str, Any] | None = None) -> Optional[GraphBuilder]:
    """
    创建图谱存储

    Args:
        config: 配置字典

    Returns:
        GraphBuilder实例
    """
    config = config or load_config()
    graph_path = _abs(config.get("paths", {}).get("graph_store", "indices/graph.pkl"))

    builder = GraphBuilder()

    # 尝试加载已有图谱
    if Path(graph_path).exists():
        try:
            builder.load(graph_path)
            logger.info(f"加载图谱: {graph_path}")
        except Exception as e:
            logger.warning(f"图谱加载失败: {e}")

    return builder


def create_llm_client(config: Dict[str, Any] | None = None) -> QwenClient:
    """
    创建LLM客户端

    Args:
        config: 配置字典

    Returns:
        QwenClient实例
    """
    config = config or load_config()
    qwen_cfg = config.get("qwen", {})

    api_key = os.environ.get(qwen_cfg.get("api_key_env", "QWEN_API_KEY"), "")

    return QwenClient(
        api_key=api_key,
        model=qwen_cfg.get("model", "qwen-plus"),
        base_url=qwen_cfg.get("base_url", "https://dashscope-intl.aliyuncs.com/api/v1"),
        temperature=qwen_cfg.get("temperature", 0.3),
        max_tokens=qwen_cfg.get("max_tokens", 512),
        timeout=qwen_cfg.get("timeout", 30),
        max_retries=qwen_cfg.get("max_retries", 3)
    )


def build_indices(
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    构建所有索引

    Args:
        config_path: 配置文件路径
        data_dir: 数据目录

    Returns:
        索引构建结果
    """
    config = load_config(config_path) if config_path else load_config()
    data_dir = data_dir or _abs(config.get("paths", {}).get("data_dir", "data"))

    results: Dict[str, Any] = {
        "vector_index": None,  # type: ignore[assignment]
        "graph": None  # type: ignore[assignment]
    }

    # 加载数据
    from src.data.loader import load_call_records, load_contacts
    from src.data.cleaner import clean_call_text
    from langchain_core.documents import Document

    call_records = load_call_records(data_dir)
    contacts = load_contacts(data_dir)

    # 构建文档
    documents = []
    for record in call_records:
        cleaned_text = clean_call_text(record.text)
        if cleaned_text:
            documents.append(Document(
                page_content=cleaned_text,
                metadata={
                    "id": record.id,
                    "category": record.category,
                    "sub_category": record.sub_category
                }
            ))

    logger.info(f"准备构建索引: {len(documents)} 文档")

    # 构建向量索引
    if documents:
        try:
            from src.indexing.embeddings import load_bge_embedding
            embedder = load_bge_embedding()
            vector_indexer = VectorIndexer(
                persist_dir=_abs(config.get("paths", {}).get("vector_store", "indices/vector_store")),
                embedder=embedder
            )
            vector_indexer.build_from_records([
                {"text": doc.page_content, **doc.metadata}
                for doc in documents
            ])
            results["vector_index"] = vector_indexer
            logger.info("向量索引构建完成")
        except Exception as e:
            logger.error(f"向量索引构建失败: {e}")

    # 构建图谱
    if call_records or contacts:
        try:
            graph_builder = GraphBuilder()

            # 添加通话记录
            for record in call_records:
                graph_builder.add_call_record(
                    call_id=record.id,
                    text=record.text,
                    category=record.category
                )

            # 添加联系人
            for contact in contacts:
                graph_builder.add_contact(
                    contact_id=contact.id,
                    name=contact.name,
                    company=contact.company or "",
                    relation=contact.relation or ""
                )

            # 构建NetworkX图
            graph_builder.build_networkx()

            # 保存
            graph_path = _abs(config.get("paths", {}).get("graph_store", "indices/graph.pkl"))
            graph_builder.save(graph_path)

            results["graph"] = graph_builder
            logger.info("图谱构建完成")
        except Exception as e:
            logger.error(f"图谱构建失败: {e}")

    return results


def build_call_rag(config_path: str = "configs/config.yaml") -> Any:
    """
    构建 Hybrid RAG 实例

    这是交互查询的入口，创建完整的 RAG 组件：
    - 向量检索器
    - 图检索器
    - 路由决策器
    - RRF 融合器
    - 答案生成器

    Args:
        config_path: 配置文件路径

    Returns:
        HybridRAG 实例
    """
    from src.hybrid_rag import HybridRAG

    cfg = load_config(config_path)
    api_key = os.environ.get(cfg.get("qwen", {}).get("api_key_env", "QWEN_API_KEY"), "")

    # LLM 客户端
    llm_client = QwenClient(
        api_key=api_key,
        model=cfg.get("qwen", {}).get("model", "qwen-plus"),
        base_url=cfg.get("qwen", {}).get("base_url"),
        temperature=0.0,
    )

    # 向量检索器
    vector_store = create_vector_store(cfg)
    vector_retriever = VectorRetriever(db=vector_store)

    # 图检索器
    graph_retriever = GraphRetriever(
        graph_path=_abs(cfg.get("paths", {}).get("graph_store", "indices/graph.pkl")),
        communities_path=_abs(cfg.get("paths", {}).get("communities", "indices/communities.json")),
    )

    # 路由决策器
    router = LLMRouter(llm_client)

    # RRF 融合器
    rrf_k = cfg.get("retrieval", {}).get("rrf_k_const", 60)
    fuser = RRFFuser(k_const=rrf_k)

    # 重排器
    reranker = SimpleReranker()

    # 答案生成器
    gen_cfg = cfg.get("generation", {})
    answerer = Answerer(
        llm_client=QwenClient(
            api_key=api_key,
            model=cfg.get("qwen", {}).get("model", "qwen-plus"),
            base_url=cfg.get("qwen", {}).get("base_url"),
            temperature=gen_cfg.get("temperature", 0.3),
            max_tokens=gen_cfg.get("max_tokens", 512),
        )
    )

    # 构建 HybridRAG
    return HybridRAG(
        router=router,
        vec_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        fuser=fuser,
        reranker=reranker,
        answerer=answerer,
        vector_k=cfg.get("retrieval", {}).get("vector_k", 10),
        top_n=cfg.get("retrieval", {}).get("top_n", 5),
    )


# 便捷别名
create_hybrid_rag = build_call_rag
