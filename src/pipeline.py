"""
来电助手主流程
==============
参考 RAG-CITY 架构

流程: 分类路由 → Agent处理 → 生成回复
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from src.utils.logger import get_logger
from src.retrieval.router import CallRouter, RouteDecision, CallCategory
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.fusion import RRFFuser
from src.retrieval.reranker import SimpleReranker
from src.agents.base import AgentResult
from src.agents.factory import get_agent

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Pipeline处理结果"""
    # 分类信息
    category: str
    category_desc: str
    confidence: float
    reason: str
    sub_scenario: str

    # 处理结果
    action: str
    response: str
    notification: Optional[str]

    # 元信息
    timing: Dict[str, float]
    citations: Dict[str, Any]


class CallAssistantPipeline:
    """
    来电助手主流程

    流程:
    1. CallRouter 分类路由（关键词+RAG辅助）
    2. Agent 处理（外卖/正常/诈骗）
    3. 返回结果
    """

    def __init__(
        self,
        vector_store=None,
        contact_store=None,
        call_store=None,
        llm_client=None,
        config: Optional[Dict] = None
    ):
        """
        初始化Pipeline

        Args:
            vector_store: 向量存储
            contact_store: 联系人存储
            call_store: 通话记录存储
            llm_client: LLM客户端
            config: 配置字典
        """
        self.config = config or {}
        self.vector_store = vector_store
        self.contact_store = contact_store
        self.call_store = call_store
        self.llm_client = llm_client

        # 初始化组件
        self.router = CallRouter(
            vector_store=vector_store,
            contact_store=contact_store,
            call_store=call_store,
            llm_client=llm_client
        )

        # 融合器
        self.fuser = RRFFuser(k_const=self.config.get("rrf_k_const", 60))

        # 重排器
        self.reranker = SimpleReranker()

        # 向量检索器
        self.vector_retriever = VectorRetriever(db=vector_store)

        # 上课状态
        self._is_class_in_session = False

        logger.info("[OK] CallAssistantPipeline 初始化完成")

    def query(self, call_text: str, **kwargs) -> PipelineResult:
        """
        处理来电（主入口）

        Args:
            call_text: 来电内容
            **kwargs: 支持 is_class_in_session 覆盖默认状态

        Returns:
            PipelineResult: 处理结果
        """
        start_time = time.time()
        timing = {}

        logger.info("=" * 60)
        logger.info("[1/3] 分类路由...")
        route_start = time.time()

        # Step 1: 路由分类
        route_result = self.router.route(call_text)
        timing["route"] = time.time() - route_start

        logger.info(f"   分类: {route_result.category}")
        logger.info(f"   置信度: {route_result.confidence:.2f}")
        logger.info(f"   理由: {route_result.reason}")

        # 获取上课状态
        is_class = kwargs.get("is_class_in_session", self._is_class_in_session)

        # Step 2: Agent处理
        logger.info("[2/3] Agent处理...")
        agent_start = time.time()

        agent = get_agent(route_result.category, is_class_in_session=is_class)
        agent_result = agent.process(call_text, is_class_in_session=is_class)
        timing["agent"] = time.time() - agent_start

        # Step 3: 构建结果
        logger.info("[3/3] 生成结果...")
        result = PipelineResult(
            category=route_result.category,
            category_desc=CallCategory(route_result.category).emoji + " " + CallCategory(route_result.category).description,
            confidence=route_result.confidence,
            reason=route_result.reason,
            sub_scenario=agent_result.sub_scenario,
            action=agent_result.action,
            response=agent_result.response,
            notification=agent_result.notification,
            timing=timing,
            citations=agent_result.citations
        )

        total_time = time.time() - start_time
        timing["total"] = total_time
        logger.info(f"⏱️ 耗时: {total_time*1000:.0f}ms")

        return result

    def process(self, call_text: str, **kwargs) -> PipelineResult:
        """query的别名"""
        return self.query(call_text, **kwargs)

    def set_class_mode(self, in_session: bool):
        """
        设置上课/下课模式

        Args:
            in_session: True=上课中, False=下课
        """
        self._is_class_in_session = in_session
        status = "上课中" if in_session else "下课"
        logger.info(f"[MODE] 课堂模式: {status}")

    def format_display(self, result: PipelineResult) -> str:
        """
        格式化显示结果

        Args:
            result: 处理结果

        Returns:
            格式化字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("📋 来电助手分析结果")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"🔖 分类: {result.category_desc}")
        lines.append(f"📌 子场景: {result.sub_scenario}")
        lines.append(f"📌 操作: {result.action}")
        lines.append(f"📊 置信度: {result.confidence:.0%}")
        lines.append("")

        if result.response:
            lines.append("-" * 40)
            lines.append("💬 回复话术:")
            lines.append("-" * 40)
            lines.append(result.response)
            lines.append("")

        if result.notification:
            lines.append("-" * 40)
            lines.append("📱 通知内容:")
            lines.append("-" * 40)
            lines.append(result.notification)
            lines.append("")

        lines.append(f"⏱️ 耗时: {result.timing.get('total', 0)*1000:.0f}ms")
        lines.append("=" * 60)

        return "\n".join(lines)
