"""
LLM 路由决策
============
使用 LLM 判断查询类型

参考 RAG-CITY src/retrieval/router.py 设计
"""

from dataclasses import dataclass
from typing import Optional
from src.utils.llm_client import QwenClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RouteDecision:
    """路由决策结果"""
    route: str          # vector | graph | hybrid
    reason: str
    confidence: float = 1.0


# 路由提示词
ROUTE_PROMPT = """你是一个来电助手 RAG 系统的路由决策器。

根据用户查询，判断最适合的检索策略：

1. **vector** (向量检索): 适合事实性查询
   - "这个号码是谁的？"
   - "外卖到了吗？"
   - "上次和王总通话说了什么？"
   - "帮我查一下快递单号"

2. **graph** (图谱检索): 适合关系推理、风险分析
   - "这类电话有什么风险？"
   - "最近有哪些可疑来电？"
   - "诈骗电话有什么特征？"
   - "哪些号码是推销？"

3. **hybrid** (混合检索): 默认策略，适合综合性查询
   - "关于 XX 的所有信息"
   - "查一下这个人的记录"
   - 其他模糊或多维度查询

请只输出 JSON 格式：
{{"route": "vector|graph|hybrid", "reason": "简短原因"}}
"""


class LLMRouter:
    """
    LLM 路由决策器

    使用 LLM 判断查询类型，决定使用哪种检索策略
    """

    def __init__(self, llm_client: QwenClient):
        """
        初始化路由器

        Args:
            llm_client: LLM 客户端
        """
        self.llm = llm_client

    def route(self, query: str) -> RouteDecision:
        """
        路由决策

        Args:
            query: 用户查询

        Returns:
            RouteDecision: 路由决策结果
        """
        # 快速关键词匹配作为 fallback
        quick_decision = self._quick_route(query)
        if quick_decision:
            logger.info(f"快速路由: {quick_decision.route} - {quick_decision.reason}")
            return quick_decision

        # LLM 决策
        try:
            # 使用 call 方法，system prompt 作为第一个参数
            response = self.llm.call(query, system=ROUTE_PROMPT)
            return self._parse_response(response)

        except Exception as e:
            logger.warning(f"LLM 路由失败: {e}，使用默认 hybrid")
            return RouteDecision(
                route="hybrid",
                reason=f"LLM 失败: {str(e)[:50]}",
                confidence=0.5
            )

    def _quick_route(self, query: str) -> Optional[RouteDecision]:
        """快速关键词路由"""
        query_lower = query.lower()

        # 图谱关键词
        graph_keywords = ["风险", "诈骗", "可疑", "推销", "骚扰", "特征", "关系", "类型", "哪些"]
        if any(kw in query_lower for kw in graph_keywords):
            return RouteDecision(
                route="graph",
                reason="检测到图谱关键词",
                confidence=0.8
            )

        # 向量关键词
        vector_keywords = ["是谁", "什么", "号码", "内容", "说了什么", "查", "找"]
        if any(kw in query_lower for kw in vector_keywords):
            return RouteDecision(
                route="vector",
                reason="检测到向量检索关键词",
                confidence=0.8
            )

        return None

    def _parse_response(self, response: str) -> RouteDecision:
        """解析 LLM 响应"""
        import json
        import re

        try:
            # 提取 JSON
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return RouteDecision(
                    route=data.get("route", "hybrid"),
                    reason=data.get("reason", ""),
                    confidence=0.9
                )
        except Exception as e:
            logger.warning(f"解析路由响应失败: {e}")

        return RouteDecision(
            route="hybrid",
            reason="解析失败，使用默认",
            confidence=0.5
        )
