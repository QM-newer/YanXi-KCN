"""
社区检测与摘要生成
==================
参考 RAG-CITY community_builder.py 设计

使用 Louvain 算法进行社区检测
使用 LLM 生成社区摘要
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_communities(G: Any, seed: int = 42, min_size: int = 3) -> List[List[str]]:
    """
    使用 Louvain 算法检测社区

    Args:
        G: NetworkX 图
        seed: 随机种子
        min_size: 最小社区大小

    Returns:
        社区列表，每个社区是节点ID列表
    """
    try:
        import networkx as nx
        import community.community_louvain as community_louvain
    except ImportError:
        try:
            from community import community_louvain
        except ImportError:
            logger.error("请安装 python-louvain: pip install python-louvain")
            return []

    # 检测社区
    partition = community_louvain.best_partition(G)

    # 按社区ID分组
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    # 过滤小社区
    result = [nodes for nodes in communities.values() if len(nodes) >= min_size]

    logger.info(f"检测到 {len(result)} 个社区 (最小大小: {min_size})")
    return result


def annotate_community_ids(G: Any, communities: List[List[str]]) -> None:
    """
    为图中的节点标注社区ID

    Args:
        G: NetworkX 图
        communities: 社区列表
    """
    # 创建节点到社区的映射
    node_to_comm = {}
    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            node_to_comm[node] = comm_id

    # 为节点添加社区属性
    nx = None
    try:
        import networkx as nx
    except ImportError:
        return

    for node in G.nodes():
        G.nodes[node]["community_id"] = node_to_comm.get(node, -1)

    logger.info("社区ID标注完成")


def generate_community_summaries(
    G: Any,
    communities: List[List[str]],
    llm_client: Any,
    call_map: Dict[str, str] = None,
    max_calls_per_community: int = 5
) -> List[Dict[str, Any]]:
    """
    为每个社区生成摘要

    Args:
        G: NetworkX 图
        communities: 社区列表
        llm_client: LLM 客户端
        call_map: call_id -> text 的映射
        max_calls_per_community: 每个社区最多采样的话术数

    Returns:
        社区摘要列表
    """
    summaries = []
    call_map = call_map or {}

    for comm_id, nodes in enumerate(communities):
        if len(nodes) < 3:
            continue

        # 收集社区信息
        node_info = []
        sample_call_ids = []
        categories = set()
        tags = set()

        for node in nodes:
            node_data = G.nodes[node]
            node_type = node_data.get("type", "unknown")

            if node_type == "call_record":
                call_id = node_data.get("name", "").split("_")[-1] if "_" in node_data.get("name", "") else node
                sample_call_ids.append(call_id)

                # 收集分类和标签
                if "category" in node_data:
                    categories.add(node_data["category"])
                if "tags" in node_data:
                    tags.update(node_data["tags"].split(",") if isinstance(node_data["tags"], str) else [])

            node_info.append(f"{node_type}:{node_data.get('name', node)}")

        # 采样话术
        sample_calls = []
        for call_id in sample_call_ids[:max_calls_per_community]:
            if call_id in call_map:
                sample_calls.append(call_map[call_id])

        # 提取核心实体
        core_entities = list(set([
            G.nodes[n].get("name", "")
            for n in nodes
            if G.nodes[n].get("type") in ["person", "company", "concept"]
        ]))[:10]

        # 生成摘要
        summary = generate_single_summary(
            comm_id=comm_id,
            size=len(nodes),
            categories=list(categories),
            core_entities=core_entities,
            sample_calls=sample_calls,
            llm_client=llm_client
        )

        if summary:
            summaries.append(summary)

    logger.info(f"生成 {len(summaries)} 个社区摘要")
    return summaries


def generate_single_summary(
    comm_id: int,
    size: int,
    categories: List[str],
    core_entities: List[str],
    sample_calls: List[str],
    llm_client: Any
) -> Optional[Dict[str, Any]]:
    """
    为单个社区生成摘要

    Args:
        comm_id: 社区ID
        size: 社区大小
        categories: 类别列表
        core_entities: 核心实体
        sample_calls: 示例话术
        llm_client: LLM 客户端

    Returns:
        社区摘要字典
    """
    if not categories and not core_entities:
        categories = ["其他"]

    # 构建提示
    sample_text = "\n".join([f"- {c}" for c in sample_calls[:3]]) if sample_calls else "无"

    prompt = f"""你是来电助手数据分析专家。请为以下通话记录社区生成简短摘要。

社区信息:
- 社区ID: {comm_id}
- 社区大小: {size} 个节点
- 主要类别: {', '.join(categories) if categories else '未分类'}
- 核心实体: {', '.join(core_entities[:5]) if core_entities else '无'}

示例话术:
{sample_text}

请生成2-3句话的社区摘要，概括这个社区的主要特征。

直接输出摘要，不要有其他内容。"""

    try:
        if llm_client:
            response = llm_client.generate(prompt)
            summary_text = response.strip() if response else f"这是一个{categories[0] if categories else '其他'}相关的通话社区。"
        else:
            summary_text = f"这是一个{categories[0] if categories else '其他'}相关的通话社区，包含{size}条记录。"

        return {
            "community_id": comm_id,
            "size": size,
            "categories": categories,
            "core_entities": core_entities,
            "sample_calls": sample_calls[:3],
            "summary": summary_text
        }
    except Exception as e:
        logger.warning(f"生成社区 {comm_id} 摘要失败: {e}")
        return {
            "community_id": comm_id,
            "size": size,
            "categories": categories,
            "core_entities": core_entities,
            "sample_calls": sample_calls[:3],
            "summary": f"这是一个{categories[0] if categories else '其他'}相关的通话社区。"
        }


def save_communities(communities: List[Dict[str, Any]], output_path: str) -> None:
    """
    保存社区摘要到文件

    Args:
        communities: 社区摘要列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(communities, f, ensure_ascii=False, indent=2)

    logger.info(f"保存 {len(communities)} 个社区摘要到 {output_path}")


def load_communities(input_path: str) -> List[Dict[str, Any]]:
    """
    从文件加载社区摘要

    Args:
        input_path: 输入文件路径

    Returns:
        社区摘要列表
    """
    input_path = Path(input_path)
    if not input_path.exists():
        logger.warning(f"社区文件不存在: {input_path}")
        return []

    with open(input_path, "r", encoding="utf-8") as f:
        communities = json.load(f)

    logger.info(f"加载 {len(communities)} 个社区摘要")
    return communities


def build_summary_index(
    communities: List[Dict[str, Any]],
    embedder: Any,
    persist_dir: str
) -> Any:
    """
    为社区摘要构建向量索引

    Args:
        communities: 社区摘要列表
        embedder: Embedding 模型
        persist_dir: 持久化目录

    Returns:
        Chroma 向量库
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
    except ImportError:
        logger.error("请安装 chromadb 和 langchain: pip install chromadb langchain")
        return None

    docs = []
    for comm in communities:
        doc = Document(
            page_content=comm["summary"],
            metadata={
                "community_id": comm["community_id"],
                "size": comm["size"],
                "categories": "|".join(comm["categories"]),
                "core_entities": "|".join(comm["core_entities"][:5])
            }
        )
        docs.append(doc)

    if not docs:
        logger.warning("没有社区摘要可索引")
        return None

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        persist_directory=str(persist_dir),
        collection_name="community_summaries"
    )

    db.persist()
    logger.info(f"社区摘要索引构建完成: {len(docs)} 个文档")

    return db
