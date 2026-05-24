"""
向量检索器
==========
封装向量数据库的检索接口
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """
    向量检索器

    封装不同向量数据库的检索接口
    """

    def __init__(self, db=None, top_k: int = 10):
        self.db = db
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        检索相似文档

        Args:
            query: 查询文本
            k: 返回数量
            filter: 元数据过滤条件

        Returns:
            文档列表
        """
        k = k or self.top_k

        if self.db is None:
            logger.warning("向量数据库未初始化")
            return []

        try:
            # 尝试使用Chroma接口
            return self.db.similarity_search(query, k=k, filter=filter)
        except AttributeError:
            # 尝试使用FAISS接口
            try:
                docs = self.db.similarity_search_with_score(query, k=k)
                return [doc for doc, score in docs]
            except:
                logger.error("不支持的向量数据库类型")
                return []

    def retrieve_with_scores(
        self,
        query: str,
        k: int = None
    ) -> List[tuple[Document, float]]:
        """检索并返回相似度分数"""
        k = k or self.top_k

        if self.db is None:
            return []

        try:
            # Chroma with scores
            results = self.db.similarity_search_with_score(query, k=k)
            return results
        except:
            # Simple fallback
            docs = self.retrieve(query, k)
            return [(doc, 1.0) for doc in docs]

    def set_top_k(self, k: int):
        """设置默认返回数量"""
        self.top_k = k
