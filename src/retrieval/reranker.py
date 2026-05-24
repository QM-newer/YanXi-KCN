"""
重排器
======
对检索结果进行重排序
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleReranker:
    """
    简单重排器

    基于关键词匹配和分类标签进行重排
    """

    def __init__(self, boost_keywords: Dict[str, List[str]] = None):
        """
        Args:
            boost_keywords: 关键词boost配置
        """
        self.boost_keywords = boost_keywords or {
            "delivery": ["外卖", "饿了么", "美团", "取餐", "骑手", "快递"],
            "normal": ["领导", "老板", "家人", "面试", "打车"],
            "risk": ["转账", "验证码", "中奖", "诈骗"]
        }

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
        category: str = None
    ) -> List[Document]:
        """
        重排文档

        Args:
            query: 查询文本
            documents: 检索文档列表
            top_n: 返回前N个
            category: 当前分类（用于关键词boost）

        Returns:
            重排后的文档
        """
        if not documents:
            return []

        # 计算每个文档的得分
        scored_docs = []
        query_lower = query.lower()

        for doc in documents:
            score = self._compute_score(doc, query_lower, category)
            scored_docs.append((doc, score))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_n]]

    def _compute_score(
        self,
        doc: Document,
        query: str,
        category: str = None
    ) -> float:
        """计算文档得分"""
        score = 0.0
        content_lower = doc.page_content.lower()

        # 基础相关度：query词在文档中出现的次数
        query_terms = set(query)
        content_terms = set(content_lower)
        overlap = len(query_terms & content_terms)
        score += overlap * 0.1

        # 分类boost
        if category and category in self.boost_keywords:
            for keyword in self.boost_keywords[category]:
                if keyword in content_lower:
                    score += 0.5

        # 元数据中的标签
        meta = doc.metadata or {}
        if "category" in meta and meta["category"] == category:
            score += 1.0
        if "is_risky" in meta:
            score -= 0.5 if meta["is_risky"] else 0

        return score


class BgeReranker:
    """
    BGE重排器

    使用BGE reranker模型进行重排（需要GPU）
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def _load_model(self):
        """懒加载模型"""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device
                )
            except ImportError:
                logger.error("sentence-transformers未安装")
                return None
        return self.model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5
    ) -> List[Document]:
        """使用BGE模型重排"""
        model = self._load_model()
        if model is None:
            # 回退到简单重排
            reranker = SimpleReranker()
            return reranker.rerank(query, documents, top_n)

        texts = [doc.page_content for doc in documents]
        pairs = [[query, text] for text in texts]

        scores = model.predict(pairs)

        # 按分数排序
        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:top_n]]
