"""
来电分类路由器
==============
结合关键词快速匹配 + RAG检索辅助进行来电分类
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CallCategory(Enum):
    """来电分类枚举"""
    DELIVERY = "delivery"       # 外卖/快递
    NORMAL = "normal"          # 正常来电
    RISK = "risk"              # 诈骗/骚扰

    @property
    def description(self) -> str:
        return {
            "delivery": "外卖/快递",
            "normal": "正常来电",
            "risk": "诈骗风险"
        }.get(self.value, "未知")

    @property
    def emoji(self) -> str:
        return {
            "delivery": "🍜",
            "normal": "📞",
            "risk": "⚠️"
        }.get(self.value, "❓")


@dataclass
class RouteDecision:
    """路由决策结果"""
    category: str
    confidence: float
    reason: str
    sub_scenario: Optional[str] = None
    keywords_matched: Optional[list] = None


class CallRouter:
    """
    来电分类路由器

    采用多通道评分机制:
    1. 关键词快速匹配（低成本、高准确）
    2. RAG检索辅助（结合历史记录）
    3. 综合评分决策
    """

    # 分类关键词定义 - 按优先级排序
    PATTERNS = {
        CallCategory.RISK: [
            # 高风险关键词（需要严格匹配）
            (r"转账|汇款|安全账户|打钱", "financial"),
            (r"验证码|短信码|动态码|告诉我|说下", "verify_code"),
            (r"中奖|恭喜|奖金|免费领", "scam"),
            # 诈骗模式（需要上下文）
            (r"账户.*异常|涉及.*案件|涉嫌.*违法", "judicial"),
            (r"客服.*退款|订单.*异常", "fake_service"),
            # 只有当"领导"+"转账"同时出现才判定风险
            (r"领导.*转账|老板.*汇款|上级.*钱", "fake_leader"),
            # 推销
            (r"推销|保险.*推荐|贷款.*咨询", "solicit"),
        ],
        CallCategory.DELIVERY: [
            # 外卖关键词
            (r"外卖|饿了么|美团外卖", "food_delivery"),
            (r"送餐|取餐|餐品|餐.*好了", "rider"),
            (r"取餐码|订单号|单号", "order_code"),
            (r"骑手|小哥", "rider_info"),
            # 快递关键词
            (r"快递|包裹|顺丰|中通|圆通|韵达|菜鸟驿站", "express"),
            (r"取件|快递柜|代收点", "pickup"),
        ],
        CallCategory.NORMAL: [
            # 重要来电（独立出现）
            (r"^喂|^你好|^您好|^在吗", "greeting"),
            (r"领导|老板|经理|总经理|董事", "leader"),
            (r"妈妈|爸爸|老婆|老公|家人|孩子", "family"),
            (r"朋友|同学|同事|闺蜜|兄弟", "friend"),
            # 日常
            (r"打车|车到了|车在.*等|网约车", "taxi"),
            (r"面试|会议|约定|预约", "appointment"),
            (r"老师|导师|教授|学校", "education"),
            (r"客户|合作|方案|报价", "business"),
            # 礼貌开头
            (r"请问|方便", "polite"),
        ],
    }

    def __init__(
        self,
        vector_store=None,
        contact_store=None,
        call_store=None,
        llm_client=None
    ):
        self.vector_store = vector_store
        self.contact_store = contact_store
        self.call_store = call_store
        self.llm_client = llm_client

    def route(self, call_text: str) -> RouteDecision:
        """
        路由决策

        Args:
            call_text: 来电内容

        Returns:
            RouteDecision: 分类结果
        """
        # Step 1: 关键词快速匹配
        keyword_scores, matched_keywords = self._keyword_match(call_text)

        # Step 2: RAG检索辅助
        rag_scores = self._rag_assist(call_text)

        # Step 3: 综合评分
        final_scores = {
            CallCategory.DELIVERY: keyword_scores[CallCategory.DELIVERY] + rag_scores[CallCategory.DELIVERY],
            CallCategory.NORMAL: keyword_scores[CallCategory.NORMAL] + rag_scores[CallCategory.NORMAL],
            CallCategory.RISK: keyword_scores[CallCategory.RISK] + rag_scores[CallCategory.RISK],
        }

        # 风险保护：关键词命中直接返回
        if keyword_scores[CallCategory.RISK] >= 2.0:
            return RouteDecision(
                category=CallCategory.RISK.value,
                confidence=0.95,
                reason="检测到诈骗关键词",
                sub_scenario=self._detect_scam_type(call_text),
                keywords_matched=matched_keywords.get(CallCategory.RISK, [])
            )

        # 外卖关键词命中
        if keyword_scores[CallCategory.DELIVERY] >= 1.5:
            return RouteDecision(
                category=CallCategory.DELIVERY.value,
                confidence=0.90,
                reason="检测到外卖/快递关键词",
                sub_scenario=self._detect_delivery_type(call_text),
                keywords_matched=matched_keywords.get(CallCategory.DELIVERY, [])
            )

        # 返回得分最高的
        best_category = max(final_scores, key=final_scores.get)
        confidence = min(final_scores[best_category], 0.95)

        return RouteDecision(
            category=best_category.value,
            confidence=confidence,
            reason=f"综合评分: delivery={final_scores[CallCategory.DELIVERY]:.2f}, "
                   f"normal={final_scores[CallCategory.NORMAL]:.2f}, "
                   f"risk={final_scores[CallCategory.RISK]:.2f}",
            sub_scenario=self._detect_sub_scenario(call_text, best_category),
            keywords_matched=matched_keywords.get(best_category, [])
        )

    def _keyword_match(self, text: str) -> tuple[Dict[CallCategory, float], Dict[CallCategory, list]]:
        """关键词快速匹配"""
        scores = {cat: 0.0 for cat in CallCategory}
        matched = {cat: [] for cat in CallCategory}

        for category, patterns in self.PATTERNS.items():
            for pattern, tag in patterns:
                if re.search(pattern, text):
                    # 风险类别权重更高
                    weight = 2.0 if category == CallCategory.RISK else 1.0
                    scores[category] += weight
                    matched[category].append(tag)

        return scores, matched

    def _rag_assist(self, text: str) -> Dict[CallCategory, float]:
        """RAG检索辅助分类"""
        scores = {cat: 0.0 for cat in CallCategory}

        if self.vector_store is None:
            return scores

        try:
            results = self.vector_store.similarity_search(text, k=5)

            delivery_count = 0
            risk_count = 0
            normal_count = 0

            for doc in results:
                content = doc.page_content.lower() if doc.page_content else ""
                meta = doc.metadata or {}

                # 统计命中 - 同时用 content 关键词 + metadata
                if any(kw in content for kw in ["外卖", "饿了么", "美团", "取餐", "骑手", "快递", "配送", "送餐", "取件"]):
                    delivery_count += 1
                # RISK 检测：metadata 标签 + 内容关键词双重保障
                is_risky_meta = meta.get("is_risky") or any(
                    t in str(meta.get("topics", "")) for t in ["诈骗", "可疑", "推销"]
                ) or meta.get("category") in ["诈骗风险", "诈骗电话", "推销电话", "诈骗"]
                is_risky_content = any(
                    kw in content for kw in ["转账", "验证码", "中奖", "诈骗", "汇款", "安全账户", "涉嫌", "免费领"]
                )
                if is_risky_meta or is_risky_content:
                    risk_count += 1
                if any(kw in content for kw in ["领导", "老板", "客户", "家人", "面试", "会议", "同事", "朋友", "老婆", "老公", "妈妈", "爸爸", "打车"]):
                    normal_count += 1

            total = delivery_count + risk_count + normal_count
            if total > 0:
                scores[CallCategory.DELIVERY] = delivery_count / total * 0.3
                scores[CallCategory.RISK] = risk_count / total * 0.3
                scores[CallCategory.NORMAL] = normal_count / total * 0.3

        except Exception as e:
            logger.warning(f"RAG辅助分类失败: {e}")

        return scores

    def _detect_scam_type(self, text: str) -> str:
        """识别诈骗类型"""
        if "客服" in text or "退款" in text:
            return "冒充客服诈骗"
        if "公安" in text or "检察院" in text:
            return "冒充公检法"
        if "领导" in text or "老板" in text:
            return "冒充领导诈骗"
        if "中奖" in text:
            return "中奖诈骗"
        if "贷款" in text:
            return "贷款诈骗"
        if "投资" in text or "理财" in text:
            return "投资诈骗"
        return "疑似诈骗"

    def _detect_delivery_type(self, text: str) -> str:
        """识别外卖/快递类型"""
        if any(kw in text for kw in ["外卖", "饿了么", "美团", "餐品", "取餐"]):
            return "外卖配送"
        if any(kw in text for kw in ["快递", "包裹", "顺丰", "中通", "圆通"]):
            return "快递送达"
        return "配送"

    def _detect_sub_scenario(self, text: str, category: CallCategory) -> str:
        """识别子场景"""
        if category == CallCategory.NORMAL:
            if any(kw in text for kw in ["领导", "老板", "经理"]):
                return "领导/老板来电"
            if any(kw in text for kw in ["妈妈", "爸爸", "家人"]):
                return "家人来电"
            if any(kw in text for kw in ["面试", "会议"]):
                return "面试/会议"
            if any(kw in text for kw in ["打车", "车到了"]):
                return "打车到达"
        elif category == CallCategory.RISK:
            if "验证码" in text:
                return "验证码诈骗"
            if "转账" in text:
                return "转账诈骗"
        return "普通"
