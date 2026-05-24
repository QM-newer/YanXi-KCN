"""
Hybrid RAG 主类
===============
参考 RAG-CITY src/pipeline.py 设计

流程: 路由决策 → 向量检索 → 图检索 → 融合 → 重排 → 生成答案
"""

from dataclasses import dataclass
from typing import List, Optional
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """查询结果"""
    text: str                    # 生成的答案
    citations: dict             # 引用
    route: str                  # 路由类型
    route_reason: str           # 路由原因
    retrieved_docs: List[Document]  # 融合后的文档
    reranked_docs: List[Document]  # 重排后的文档
    vector_docs: List[Document]    # 向量检索文档
    graph_docs: List[Document]     # 图检索文档


class HybridRAG:
    """
    混合 RAG 系统

    检索流程:
    1. 路由决策 (LLM/关键词)
    2. 向量检索 + 图检索
    3. RRF 融合
    4. 重排
    5. LLM 生成答案
    """

    def __init__(
        self,
        router,
        vec_retriever,
        graph_retriever,
        fuser,
        reranker,
        answerer,
        vector_k: int = 10,
        top_n: int = 5
    ):
        """
        初始化 HybridRAG

        Args:
            router: 路由决策器
            vec_retriever: 向量检索器
            graph_retriever: 图检索器
            fuser: RRF 融合器
            reranker: 重排器
            answerer: 答案生成器
            vector_k: 向量检索数量
            top_n: 最终返回数量
        """
        self.router = router
        self.vec = vec_retriever
        self.graph = graph_retriever
        self.fuser = fuser
        self.reranker = reranker
        self.answerer = answerer
        self.vector_k = vector_k
        self.top_n = top_n

        logger.info("[OK] HybridRAG 初始化完成")

    def query(self, question: str) -> dict:
        """
        处理查询

        Args:
            question: 用户问题

        Returns:
            dict: 包含 text, citations, route 等字段
        """
        # Step 1: 路由决策
        decision = self.router.route(question)
        logger.info(f"[Route] {decision.route} - {decision.reason}")

        v_docs = []
        g_docs = []
        fused_docs = []

        # Step 2: 检索
        if decision.route == "vector":
            v_docs = self.vec.retrieve(question, k=self.vector_k)
            fused_docs = v_docs
        elif decision.route == "graph":
            g_docs = self.graph.retrieve(question)
            fused_docs = g_docs
        else:  # hybrid
            v_docs = self.vec.retrieve(question, k=self.vector_k)
            g_docs = self.graph.retrieve(question)
            if v_docs or g_docs:
                fused_docs = self.fuser.fuse(v_docs, g_docs)
            else:
                fused_docs = v_docs or g_docs or []

        # Step 3: 重排
        reranked = self.reranker.rerank(question, fused_docs, top_n=self.top_n)

        # Step 4: 生成答案
        ans = self.answerer.answer(question, reranked, route=decision.route)

        return {
            "text": ans["text"],
            "citations": ans["citations"],
            "route": decision.route,
            "route_reason": decision.reason,
            "vector_docs": v_docs,
            "graph_docs": g_docs,
            "retrieved_docs": fused_docs,
            "reranked_docs": reranked,
        }

    def retrieve_only(self, question: str) -> dict:
        """
        仅检索，不生成答案

        用于调试和查看检索结果
        """
        decision = self.router.route(question)

        v_docs = []
        g_docs = []
        fused_docs = []

        if decision.route == "vector":
            v_docs = self.vec.retrieve(question, k=self.vector_k)
            fused_docs = v_docs
        elif decision.route == "graph":
            g_docs = self.graph.retrieve(question)
            fused_docs = g_docs
        else:  # hybrid
            v_docs = self.vec.retrieve(question, k=self.vector_k)
            g_docs = self.graph.retrieve(question)
            if v_docs or g_docs:
                fused_docs = self.fuser.fuse(v_docs, g_docs)
            else:
                fused_docs = v_docs or g_docs or []

        reranked = self.reranker.rerank(question, fused_docs, top_n=self.top_n)

        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "vector_docs": v_docs,
            "graph_docs": g_docs,
            "retrieved_docs": fused_docs,
            "reranked_docs": reranked,
        }
