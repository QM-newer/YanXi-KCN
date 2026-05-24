"""
数据加载器
==========
从JSON/JSONL文件加载来电数据、通讯录和训练数据
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CallRecord:
    """通话记录"""
    id: str
    text: str
    category: str
    sub_category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Contact:
    """联系人"""
    id: str
    name: str
    phone: str
    relation: str = "unknown"
    company: Optional[str] = None
    tags: List[str] = None


def load_jsonl(path: str) -> List[Dict]:
    """加载JSONL文件"""
    result = []
    file_path = Path(path)
    
    if not file_path.exists():
        logger.warning(f"文件不存在: {path}")
        return result

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logger.info(f"加载 {len(result)} 条记录 from {path}")
    return result


def load_json(path: str) -> List[Dict]:
    """加载JSON文件"""
    file_path = Path(path)
    
    if not file_path.exists():
        logger.warning(f"文件不存在: {path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        result = data if isinstance(data, list) else [data]

    logger.info(f"加载 {len(result)} 条记录 from {path}")
    return result


def load_call_records(data_dir: str = "data") -> List[CallRecord]:
    """
    加载通话记录

    Args:
        data_dir: 数据目录

    Returns:
        通话记录列表
    """
    data_path = Path(data_dir)
    records = []

    # 尝试加载多个来源
    sources = [
        data_path / "call_logs.json",
        data_path / "delivery_call_records.json",
    ]

    for source in sources:
        if source.exists():
            if source.suffix == ".jsonl":
                data = load_jsonl(str(source))
            else:
                data = load_json(str(source))

            for item in data:
                text = item.get("transcript", "") or item.get("text", "")
                if text:
                    records.append(CallRecord(
                        id=item.get("id", ""),
                        text=text,
                        category=item.get("category", "normal"),
                        sub_category=item.get("sub_category"),
                        metadata=item
                    ))

    logger.info(f"共加载 {len(records)} 条通话记录")
    return records


def load_contacts(data_dir: str = "data") -> List[Contact]:
    """
    加载通讯录

    Args:
        data_dir: 数据目录

    Returns:
        联系人列表
    """
    data_path = Path(data_dir) / "contacts.json"
    
    if not data_path.exists():
        logger.warning(f"通讯录不存在: {data_path}")
        return []

    contacts = []
    data = load_json(str(data_path))

    for item in data:
        contacts.append(Contact(
            id=item.get("id", ""),
            name=item.get("name", ""),
            phone=item.get("phone", item.get("phones", [{}])[0].get("number", "")),
            relation=item.get("relation", "unknown"),
            company=item.get("company"),
            tags=item.get("tags", [])
        ))

    logger.info(f"加载 {len(contacts)} 个联系人")
    return contacts


def load_training_data(data_dir: str = "data") -> Dict[str, List[Dict]]:
    """
    加载训练数据

    Args:
        data_dir: 数据目录

    Returns:
        分类的训练数据 {"delivery": [...], "normal": [...], "risk": [...]}
    """
    data_path = Path(data_dir)
    result = {
        "delivery": [],
        "normal": [],
        "risk": []
    }

    # 加载各个分类的训练数据
    sources = {
        "delivery": ["delivery_rag_data.jsonl", "delivery_agent_data.jsonl"],
        "normal": ["important_agent_data.jsonl"],
        "risk": ["scam_agent_data.jsonl"]
    }

    for category, filenames in sources.items():
        for filename in filenames:
            file_path = data_path / filename
            if file_path.exists():
                data = load_jsonl(str(file_path))
                result[category].extend(data)

    # 加载综合分类数据
    class_file = data_path / "classification_data.jsonl"
    if class_file.exists():
        data = load_jsonl(str(class_file))
        for item in data:
            cat = item.get("category", "normal")
            if cat in result:
                result[cat].append(item)

    # 统计
    for cat, items in result.items():
        logger.info(f"{cat}: {len(items)} 条训练数据")

    return result
