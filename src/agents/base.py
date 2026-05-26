"""
Agent 基础数据类
===============
所有 Agent 的通用返回类型
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AgentResult:
    """Agent 处理结果"""
    sub_scenario: str = ""
    action: str = ""
    response: str = ""
    notification: Optional[str] = None
    citations: Dict[str, Any] = field(default_factory=dict)
