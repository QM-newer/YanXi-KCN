"""
Agent 工厂
=========
根据分类类别返回对应的 Agent 处理器
"""

import sys
from pathlib import Path
from typing import Any

from src.agents.base import AgentResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_agent(category: str, **kwargs) -> Any:
    """
    根据类别获取对应的 Agent

    Args:
        category: 来电分类类别
        **kwargs: 额外参数（如 is_class_in_session）

    Returns:
        一个具有 process(text, **kwargs) → AgentResult 方法的对象
    """
    # 尝试导入真实的 agent 实现
    try:
        # 匹配 CallCategory 枚举值: delivery/normal/risk
        if category == "delivery":
            from src.agents.call_classifier import CallClassifierAgent
            return _DeliveryAgent(CallClassifierAgent)
        elif category in ("risk", "诈骗风险", "诈骗电话", "推销电话"):
            return _RiskAgent()
        else:
            return _DefaultAgent()
    except Exception as e:
        logger.warning(f"Agent 创建失败 ({category}): {e}，回退到默认 Agent")
        return _DefaultAgent()


class _DefaultAgent:
    """默认 Agent - 兜底处理"""

    def process(self, text: str, **kwargs) -> AgentResult:
        return AgentResult(
            sub_scenario="default",
            action="pass",
            response="",
            notification=None,
            citations={}
        )


class _DeliveryAgent:
    """外卖/快递配送 Agent"""

    def __init__(self, classifier_cls):
        self.classifier = classifier_cls(config_path="configs/config.yaml")

    def process(self, text: str, **kwargs) -> AgentResult:
        # 先分类确认
        result = self.classifier.classify(text)
        return AgentResult(
            sub_scenario=result.get("sub_category", "delivery"),
            action="notify",
            response="好的，已收到您的来电信息，将为您处理。",
            notification=f"您有新的{result.get('category', '外卖')}来电",
            citations=result
        )


class _RiskAgent:
    """风险/诈骗 Agent"""

    def process(self, text: str, **kwargs) -> AgentResult:
        return AgentResult(
            sub_scenario="risk_warning",
            action="warn",
            response="",
            notification="⚠️ 请注意，该来电存在风险",
            citations={}
        )
