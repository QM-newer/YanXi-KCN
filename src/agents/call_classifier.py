"""
电话类型分类 Agent
=================
基于 RAG 向量检索判断电话类型
"""

import os
from typing import List, Optional, Dict, Any
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

from src.utils.logger import get_logger

logger = get_logger(__name__)


# 处理规则配置
HANDLING_RULES = {
    "外卖配送": {
        "action": "代接",
        "response": "机主当前正在上课，请你告知配送员放在东门传达室，提取配送信息发短信通知机主。"
    },
    "快递取件": {
        "action": "代接",
        "response": "机主当前正在上课，请你告知配送员放在东门传达室或快递柜，提取配送信息发短信通知机主。"
    },
    "打车到达": {
        "action": "代接",
        "response": "机主正在上课，请告知对方稍等或让其自行等待，记录短信通知机主。"
    },
    "推销电话": {
        "action": "拦截",
        "response": "直接挂断，给机主推送推销来电提醒短信。"
    },
    "诈骗电话": {
        "action": "拦截",
        "response": "直接拦截挂断，给机主推送风险提醒短信，告知识别到诈骗来电。"
    },
    "诈骗风险": {
        "action": "拦截",
        "response": "直接拦截挂断，给机主推送风险提醒短信，告知识别到诈骗来电。"
    },
    "熟人问候": {
        "action": "记录",
        "response": "告知对方机主正在上课，询问是否有急事，不紧急请晚点联系，紧急则记录消息通知机主。"
    },
    "家人电话": {
        "action": "询问",
        "response": "告知对方机主正在上课，询问是否紧急，不紧急请晚点联系，紧急则记录消息通知机主。"
    },
    "领导来电": {
        "action": "优先处理",
        "response": "告知对方机主正在上课，询问是否有急事，记录留言后发短信通知机主下课回电。"
    },
    "同事协作": {
        "action": "记录",
        "response": "告知对方机主正在上课，询问是否紧急，不紧急请留言，紧急则记录消息通知机主。"
    },
    "客户来电": {
        "action": "记录",
        "response": "告知对方机主正在上课，询问是否有紧急事项，记录留言后发短信通知机主。"
    },
    "银行电话": {
        "action": "记录",
        "response": "告知对方机主正在上课，询问是否有紧急事项，记录留言后发短信通知机主回电。"
    },
    "游戏周年庆": {
        "action": "拦截",
        "response": "直接挂断，标记为营销来电。"
    },
    "面试通知": {
        "action": "优先处理",
        "response": "告知对方机主正在上课，询问面试时间和联系方式，记录后发短信通知机主。"
    },
    "无意义": {
        "action": "询问",
        "response": "告知对方没有听清，请对方重复一遍。"
    },
    "其他": {
        "action": "询问",
        "response": "告知对方机主正在上课，询问来电事由并记录。"
    }
}


class CallClassifierAgent:
    """
    基于 RAG 的电话类型分类 Agent

    使用向量检索匹配数据库中的通话记录，根据匹配结果判断类型
    """

    def __init__(
        self,
        vector_store: Any = None,
        config_path: Optional[str] = None,
        top_k: int = 10
    ):
        """
        初始化分类器

        Args:
            vector_store: 向量存储实例（可选，会自动创建）
            config_path: 配置文件路径
            top_k: 检索的最近邻数量
        """
        self.top_k = top_k
        self.vector_store = vector_store
        self.config_path = config_path or "configs/config.yaml"

        if vector_store is None:
            self._init_vector_store()

    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            from src.factory import create_vector_store
            from src.utils.config import load_config

            config = load_config(self.config_path)
            self.vector_store = create_vector_store(config)
            logger.info("向量存储初始化完成")

        except Exception as e:
            logger.warning(f"向量存储初始化失败: {e}")
            self.vector_store = None

    def _retrieve_and_classify(self, text: str) -> Dict[str, Any]:
        """
        通过 RAG 检索匹配通话记录进行分类

        Args:
            text: 对话文本

        Returns:
            分类结果
        """
        if self.vector_store is None or self.vector_store.db is None:
            return self._keyword_fallback(text)

        try:
            # 向量检索
            docs = self.vector_store.similarity_search(text, k=self.top_k)

            if not docs:
                return self._keyword_fallback(text)

            # 统计检索结果的分类
            categories = []
            scores = []
            for doc in docs:
                cat = doc.metadata.get("category", "其他")
                categories.append(cat)

            # 计算分类分布
            cat_counts = Counter(categories)
            most_common = cat_counts.most_common(1)[0]

            # 判断置信度
            top_category, count = most_common
            confidence = min(0.95, 0.5 + (count / self.top_k) * 0.5)

            # 获取检索样例
            samples = []
            for doc in docs[:3]:
                samples.append({
                    "text": doc.page_content[:50],
                    "category": doc.metadata.get("category", "其他")
                })

            return {
                "category": top_category,
                "confidence": confidence,
                "method": "rag",
                "match_count": count,
                "total_results": len(docs),
                "samples": samples,
                "category_distribution": dict(cat_counts)
            }

        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> Dict[str, Any]:
        """
        关键词回退分类

        Args:
            text: 对话文本

        Returns:
            分类结果
        """
        # 电话类型关键词定义
        categories = {
            "外卖配送": ["外卖", "饿了么", "美团", "送餐", "取餐", "骑手", "小哥", "餐", "送达", "取餐码"],
            "快递取件": ["快递", "包裹", "顺丰", "中通", "圆通", "韵达", "菜鸟", "驿站", "快递柜", "取件"],
            "推销电话": ["推销", "优惠", "套餐", "升级", "办理", "了解一下"],
            "诈骗风险": ["中奖", "验证码", "安全账户", "账户异常", "涉嫌", "银行卡", "密码", "转账", "汇款", "免费领取", "恭喜"],
            "熟人问候": ["好久不见", "朋友", "同学", "吃饭", "聚会", "借钱", "帮忙", "最近怎么样"],
            "家人电话": ["妈", "爸", "老婆", "老公", "儿子", "女儿", "家里"],
            "领导来电": ["领导", "老板", "经理", "开会", "会议", "工作安排"],
            "同事协作": ["同事", "同事们", "技术", "资料", "文档", "对接", "协作"],
            "客户来电": ["客户", "报价", "合同", "方案", "需求", "合作"],
            "打车到达": ["打车", "车到了", "网约车", "司机", "接驾"],
            "银行电话": ["银行", "信用卡", "还款", "账单"],
            "游戏周年庆": ["周年庆", "游戏", "回归", "福利", "礼包", "玩家"],
            "面试通知": ["面试", "offer", "入职", "岗位", "简历"],
        }

        # 检测诈骗话术（高优先级）
        if "换号" in text and ("总" in text or "领导" in text):
            return {
                "category": "诈骗风险",
                "confidence": 0.95,
                "method": "keyword",
                "reason": "检测到'换号'+'领导'组合，典型诈骗话术"
            }

        # 统计匹配
        matches = []
        for cat, keywords in categories.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > 0:
                matches.append((cat, count))

        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            top_cat, count = matches[0]
            confidence = min(0.85, 0.4 + count * 0.1)
            return {
                "category": top_cat,
                "confidence": confidence,
                "method": "keyword",
                "matches": matches[:3]
            }

        return {"category": "其他", "confidence": 0.3, "method": "default"}

    def _is_meaningless(self, text: str) -> bool:
        """检测文本是否为无意义内容（重复字符、纯数字、无实际语义）"""
        import re

        text = text.strip()
        if len(text) < 1:
            return True

        clean = text.strip()
        if len(clean) < 2:
            return True

        # 检测：去空格后连续数字长度 >= 15（如"123 123 123..."、"123123123..."）
        no_space = re.sub(r'\s+', '', clean)
        if re.search(r'\d{15,}', no_space):
            return True

        # 检测：同一个单字重复 >= 3 次且占比 >= 40%（如"有三有三有三"）
        for ch in set(clean):
            count = clean.count(ch)
            if count >= 3 and count / len(clean) >= 0.4:
                return True

        # 检测：同一个词组重复 >= 3 次且覆盖超过一半文本
        for wlen in [2, 3]:
            seen = set()
            for i in range(len(clean) - wlen + 1):
                word = clean[i:i + wlen]
                if word in seen:
                    continue
                seen.add(word)
                count = clean.count(word)
                if count >= 3 and len(word) * count >= len(clean) * 0.5:
                    return True

        # 检测：纯数字/计数序列（如"一二三"、"四五六七八"、"123 456"）
        # 中文数字 + 阿拉伯数字占比超过 60% → 无意义计数
        chinese_numerals_chars = '一二三四五六七八九十百千万亿零两'
        numerals = set(chinese_numerals_chars)
        # 用去空格后的长度计算占比，避免空格稀释比例
        compact = re.sub(r'\s+', '', clean)
        if len(compact) >= 2:
            numeral_count = sum(1 for ch in compact if ch in numerals or ch.isdigit())
            if numeral_count / len(compact) >= 0.6:
                # 排除混有明确语义词的情况，如"送餐"
                meaningful_keys = {'送餐', '取餐', '取件', '打钱', '汇款', '转账',
                                   '快递', '外卖', '到了', '面试', '开会', '同学'}
                if not any(kw in clean for kw in meaningful_keys):
                    return True

        return False

    def classify(self, text: str) -> Dict[str, Any]:
        """
        分类对话内容

        Args:
            text: 对话文本

        Returns:
            {
                "category": 类型,
                "confidence": 置信度,
                "method": 方法 (rag/keyword),
                ...
            }
        """
        if not text or not text.strip():
            return {"category": "其他", "confidence": 0.0, "method": "none"}

        # 优先检测无意义文本
        if self._is_meaningless(text):
            return {
                "category": "无意义",
                "confidence": 0.95,
                "method": "meaningless_detect",
                "action": "询问",
                "handling_response": "告知对方没有听清，请对方重复一遍。"
            }

        # 优先使用 RAG 检索
        result = self._retrieve_and_classify(text)

        # 如果置信度低，尝试用 LLM 增强
        if result.get("confidence", 0) < 0.6 and self._has_llm():
            llm_result = self._llm_enhance(text)
            if llm_result:
                result = llm_result

        # 添加处理规则
        category = result.get("category", "其他")
        handling_rule = HANDLING_RULES.get(category, HANDLING_RULES["其他"])
        result["action"] = handling_rule["action"]
        result["handling_response"] = handling_rule["response"]

        return result

    def _has_llm(self) -> bool:
        """检查是否有 LLM 可用"""
        return bool(os.environ.get("QWEN_API_KEY", ""))

    def _llm_enhance(self, text: str) -> Optional[Dict[str, Any]]:
        """使用 LLM 增强分类"""
        try:
            from src.utils.llm_client import QwenClient

            api_key = os.environ.get("QWEN_API_KEY", "")
            client = QwenClient(api_key=api_key)

            prompt = f"""判断以下通话内容属于哪种类型？

通话内容：{text}

可选类型：外卖配送, 快递取件, 推销电话, 诈骗风险, 熟人问候, 家人电话, 领导来电, 同事协作, 客户来电, 打车到达, 银行电话, 游戏周年庆, 面试通知, 其他

请只输出类型名称，不要其他内容。"""

            result = client.call(prompt)
            if result:
                result = result.strip().rstrip("。.!！?？,，、；;：:\n\r")
                categories = ["外卖配送", "快递取件", "推销电话", "诈骗风险", "熟人问候",
                             "家人电话", "领导来电", "同事协作", "客户来电", "打车到达",
                             "银行电话", "游戏周年庆", "面试通知"]
                # 模糊匹配：检查 LLM 返回文本中是否包含已知类型名称
                for cat in categories:
                    if cat in result:
                        return {
                            "category": cat,
                            "confidence": 0.9,
                            "method": "llm"
                        }
                # 精确匹配兜底
                if result in categories:
                    return {
                        "category": result,
                        "confidence": 0.9,
                        "method": "llm"
                    }

        except Exception as e:
            logger.warning(f"LLM 增强失败: {e}")

        return None


def create_classifier(
    config_path: str = "configs/config.yaml",
    top_k: int = 10
) -> CallClassifierAgent:
    """
    创建分类器实例

    Args:
        config_path: 配置文件路径
        top_k: 检索数量

    Returns:
        CallClassifierAgent 实例
    """
    return CallClassifierAgent(
        config_path=config_path,
        top_k=top_k
    )


def main():
    """测试分类器"""
    agent = create_classifier(top_k=10)

    test_cases = [
        "外卖到了，取餐码7788",
        "您好，您的快递到了，请及时取件",
        "我是王总，换号了，加微信",
        "好久不见，最近怎么样？周末一起吃饭？",
        "老婆，今天买什么菜？",
        "领导，今天下午3点有个会议",
        "您的打车已到达，请注意查收",
    ]

    print("=" * 60)
    print("基于 RAG 的电话类型分类 Agent 测试")
    print("=" * 60)

    for text in test_cases:
        result = agent.classify(text)
        print(f"\n输入: {text}")
        print(f"类型: {result['category']}")
        print(f"置信度: {result['confidence']:.2f} (方法: {result['method']})")

        if result.get("samples"):
            print("检索样例:")
            for s in result["samples"]:
                print(f"  - [{s['category']}] {s['text']}")


if __name__ == "__main__":
    main()
