"""
RRF融合器
=========
Reciprocal Rank Fusion 融合多个检索结果
"""

from collections import defaultdict
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


def doc_key(doc: Document) -> str:
    """生成文档唯一键"""
    meta = doc.metadata or {}
    # 优先使用ID
    if "id" in meta:
        return meta["id"]
    if "doc_id" in meta:
        return meta["doc_id"]
    # 使用内容哈希
    return f"content:{doc.page_content[:100]}"


class RRFFuser:
    """
    RRF融合器

    Reciprocal Rank Fusion算法，将多个检索源的排名融合
    """

    def __init__(self, k_const: int = 60):
        """
        Args:
            k_const: RRF常数，通常取60
        """
        self.k_const = k_const

    def fuse(self, *doc_lists: List[Document]) -> List[Document]:
        """
        融合多个文档列表

        Args:
            *doc_lists: 多个文档列表（如向量检索结果、图检索结果）

        Returns:
            融合后的文档列表
        """
        scores: Dict[str, float] = defaultdict(float)
        store: Dict[str, Document] = {}

        for docs in doc_lists:
            if not docs:
                continue
            for rank, doc in enumerate(docs):
                key = doc_key(doc)
                # RRF公式: 1 / (k + rank + 1)
                scores[key] += 1.0 / (self.k_const + rank + 1)
                store.setdefault(key, doc)

        # 按分数排序
        ranked_keys = sorted(store.keys(), key=lambda k: scores[k], reverse=True)
        return [store[k] for k in ranked_keys]

    def fuse_with_scores(
        self,
        *doc_score_lists: Tuple[List[Document], List[float]]
    ) -> List[Tuple[Document, float]]:
        """
        融合带分数的检索结果

        Args:
            *doc_score_lists: (文档列表, 分数列表)的元组

        Returns:
            融合后的(文档, 分数)列表
        """
        fused_scores: Dict[str, float] = defaultdict(float)
        store: Dict[str, Document] = {}

        for docs, scores in doc_score_lists:
            if not docs:
                continue
            for rank, (doc, score) in enumerate(zip(docs, scores)):
                key = doc_key(doc)
                # 结合原始分数和RRF排名
                rrf_score = 1.0 / (self.k_const + rank + 1)
                fused_scores[key] += score * 0.5 + rrf_score * 0.5
                store.setdefault(key, doc)

        ranked_keys = sorted(store.keys(), key=lambda k: fused_scores[k], reverse=True)
        return [(store[k], fused_scores[k]) for k in ranked_keys]
