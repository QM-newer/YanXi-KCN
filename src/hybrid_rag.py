"""
Hybrid RAG 主类
===============

流程: 路由决策 → 向量检索 → 图检索 → 融合 → 重排 → 生成答案
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """查询结果（统一 dataclass 接口）"""
    text: str                    # 生成的答案
    citations: dict             # 引用
    route: str                  # 路由类型
    route_reason: str           # 路由原因
    retrieved_docs: List[Document]  # 融合后的文档
    reranked_docs: List[Document]  # 重排后的文档
    vector_docs: List[Document]    # 向量检索文档
    graph_docs: List[Document]     # 图检索文档

    def to_dict(self) -> dict:
        """转为 dict，保持向后兼容"""
        return asdict(self)


class HybridRAG:
    """
    混合 RAG 系统

    检索流程:
    1. 路由决策 (LLM/关键词)
    2. 向量检索 + 图检索（graph 路由时向量作为 fallback）
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

    def query(self, question: str) -> QueryResult:
        """
        处理查询（返回统一 dataclass）

        Args:
            question: 用户问题

        Returns:
            QueryResult: 统一的结果对象
        """
        # Step 1: 路由决策
        decision = self.router.route(question)
        logger.info(f"[Route] {decision.route} - {decision.reason}")

        v_docs, g_docs, fused_docs = self._retrieve(question, decision)

        # Step 3: 重排
        reranked = self.reranker.rerank(question, fused_docs, top_n=self.top_n)

        # Step 4: 生成答案
        ans = self.answerer.answer(question, reranked, route=decision.route)

        return QueryResult(
            text=ans["text"],
            citations=ans["citations"],
            route=decision.route,
            route_reason=decision.reason,
            retrieved_docs=fused_docs,
            reranked_docs=reranked,
            vector_docs=v_docs,
            graph_docs=g_docs,
        )

    def _retrieve(self, question: str, decision) -> tuple:
        """
        执行检索，graph 路由时自动 fallback 到 vector

        Returns:
            (vector_docs, graph_docs, fused_docs)
        """
        v_docs = []
        g_docs = []

        if decision.route == "vector":
            v_docs = self.vec.retrieve(question, k=self.vector_k)
            return v_docs, g_docs, v_docs

        elif decision.route == "graph":
            g_docs = self.graph.retrieve(question)
            # 图检索未命中时，自动 fallback 到向量检索
            if not g_docs:
                logger.info("图检索未命中，自动 fallback 到向量检索")
                v_docs = self.vec.retrieve(question, k=self.vector_k)
                return v_docs, g_docs, v_docs
            return v_docs, g_docs, g_docs

        else:  # hybrid
            v_docs = self.vec.retrieve(question, k=self.vector_k)
            g_docs = self.graph.retrieve(question)
            if v_docs or g_docs:
                fused = self.fuser.fuse(v_docs, g_docs)
            else:
                fused = v_docs or g_docs or []
            return v_docs, g_docs, fused

    def retrieve_only(self, question: str) -> QueryResult:
        """
        仅检索，不生成答案

        用于调试和查看检索结果
        """
        decision = self.router.route(question)
        v_docs, g_docs, fused_docs = self._retrieve(question, decision)
        reranked = self.reranker.rerank(question, fused_docs, top_n=self.top_n)

        return QueryResult(
            text="",
            citations={},
            route=decision.route,
            route_reason=decision.reason,
            retrieved_docs=fused_docs,
            reranked_docs=reranked,
            vector_docs=v_docs,
            graph_docs=g_docs,
        )
