"""
数据清洗器
==========
来电文本的清洗和标准化
"""

import re
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 常见的文本噪音模式
NOISE_PATTERNS = [
    (re.compile(r"\s+"), " "),  # 多个空格
    (re.compile(r"[,，]{2,}"), "，"),  # 连续逗号
    (re.compile(r"[.。]{2,}"), "。"),  # 连续句号
    (re.compile(r"^\s+|\s+$"), ""),  # 首尾空白
]


def clean_call_text(text: str, min_length: int = 5) -> str:
    """
    清洗来电文本

    Args:
        text: 原始文本
        min_length: 最小长度阈值

    Returns:
        清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""

    # 基本清洗
    cleaned = text.strip()

    # 移除噪音模式
    for pattern, replacement in NOISE_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)

    # 移除特殊字符（保留中文、英文、数字、常用标点）
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、：；""''（）【】]", "", cleaned)

    # 过滤太短的文本
    if len(cleaned) < min_length:
        return ""

    return cleaned


def normalize_phone(phone: str) -> str:
    """
    标准化电话号码

    Args:
        phone: 原始电话

    Returns:
        标准化后的电话
    """
    if not phone:
        return ""

    # 移除非数字字符（保留国际区号前缀+86等）
    normalized = re.sub(r"(?<!\+)\D", "", phone)

    # 处理+86前缀
    if normalized.startswith("86") and len(normalized) > 10:
        normalized = "+" + normalized

    return normalized


def extract_order_code(text: str) -> Optional[str]:
    """
    提取取餐码/订单号

    Args:
        text: 来电文本

    Returns:
        订单号或None
    """
    # 模式1: 取餐码: ABC123
    match = re.search(r"取餐码[：:]?\s*([A-Za-z0-9]{4,})", text)
    if match:
        return match.group(1)

    # 模式2: 订单号: 123456789
    match = re.search(r"订单号[：:]?\s*(\d{8,})", text)
    if match:
        return match.group(1)

    # 模式3: 直接提取4-6位数字
    match = re.search(r"\b(\d{4,6})\b", text)
    if match:
        return match.group(1)

    return None


def extract_name(text: str) -> Optional[str]:
    """
    提取来电人姓名

    Args:
        text: 来电文本

    Returns:
        姓名或None
    """
    patterns = [
        r"(?:我是|叫|您好我是)[：:]?\s*([^\s，,。.]+)",
        r"(?:张|李|王|刘|陈|杨|赵|黄|周|吴|徐|孙|胡|朱|马|郭|何)\s*[某总经理]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def is_delivery_pattern(text: str) -> bool:
    """判断是否是外卖/快递相关"""
    keywords = [
        "外卖", "饿了么", "美团", "送餐", "取餐", "骑手", "小哥",
        "快递", "包裹", "顺丰", "中通", "圆通", "韵达", "菜鸟",
        "取餐码", "订单号", "餐品"
    ]
    return any(kw in text for kw in keywords)


def is_risk_pattern(text: str) -> bool:
    """判断是否包含风险关键词"""
    keywords = [
        "转账", "汇款", "验证码", "安全账户", "账户异常",
        "涉嫌", "案件", "中奖", "恭喜", "奖金", "免费",
        "客服", "退款", "异常", "公安", "检察院", "法院",
        "领导", "老板", "转账"
    ]
    return any(kw in text for kw in keywords)


def is_normal_pattern(text: str) -> bool:
    """判断是否是正常来电"""
    keywords = [
        "领导", "老板", "经理", "妈妈", "爸爸", "家人",
        "朋友", "同学", "同事", "面试", "会议", "约定",
        "打车", "车到了", "司机", "老师", "导师", "客户"
    ]
    return any(kw in text for kw in keywords)


# 类别规范化映射
CATEGORY_MAPPING = {
    # 外卖配送类
    "外卖": "外卖配送",
    "外卖配送": "外卖配送",
    "外卖电话": "外卖配送",
    "送餐": "外卖配送",
    "取餐": "外卖配送",
    "美团": "外卖配送",
    "饿了么": "外卖配送",

    # 快递类
    "快递": "快递取件",
    "快递取件": "快递取件",
    "快递电话": "快递取件",
    "包裹": "快递取件",

    # 诈骗类
    "诈骗": "诈骗电话",
    "诈骗电话": "诈骗电话",
    "风险": "诈骗电话",
    "可疑": "诈骗电话",

    # 推销类
    "推销": "推销电话",
    "推销电话": "推销电话",
    "营销": "推销电话",
    "广告": "推销电话",

    # 熟人/社交类
    "熟人": "熟人问候",
    "朋友": "熟人问候",
    "同学": "熟人问候",
    "问候": "熟人问候",
    "邀约": "熟人问候",
    "借钱": "熟人问候",

    # 家人类
    "家人": "家人电话",
    "妈妈": "家人电话",
    "爸爸": "家人电话",
    "亲戚": "家人电话",

    # 工作类
    "领导": "领导来电",
    "老板": "领导来电",
    "上司": "领导来电",
    "同事": "同事协作",
    "协作": "同事协作",
    "客户": "客户来电",
    "业务": "客户来电",

    # 打车类
    "打车": "打车到达",
    "网约车": "打车到达",
    "出租": "打车到达",

    # 游戏类
    "游戏": "游戏周年庆",
    "周年庆": "游戏周年庆",
    "原神": "游戏周年庆",
    "崩坏": "游戏周年庆",
    "明日方舟": "游戏周年庆",

    # 其他
    "其他": "其他服务",
    "未知": "其他服务",
    "其他服务": "其他服务",
}


def normalize_category(category: str) -> str:
    """
    规范化分类名称

    Args:
        category: 原始分类名称

    Returns:
        规范化后的分类名称
    """
    if not category:
        return "其他服务"

    category = category.strip()

    # 先精确匹配
    if category in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[category]

    # 再模糊匹配（检查是否包含关键词）
    for keyword, normalized in CATEGORY_MAPPING.items():
        if keyword in category or category in keyword:
            return normalized

    return "其他服务"
