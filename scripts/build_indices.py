"""
增强版索引构建脚本
=================
参考 RAG-CITY scripts/build_indices.py 设计

功能:
1. 向量索引构建 (Chroma)
2. 知识图谱构建 (NetworkX)
3. 社区检测与摘要生成 (Louvain + LLM)
4. 社区摘要向量索引
"""

import argparse
import os
import sys
import time
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 清除 Chroma 缓存
try:
    import chromadb
    chromadb.api.core.API.reset_database()
except:
    pass

from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.llm_client import QwenClient
from src.indexing.embeddings import load_bge_embedding
from src.indexing.vector_builder import VectorIndexer
from src.indexing.graph_builder import GraphBuilder
from src.indexing.community_builder import (
    detect_communities,
    annotate_community_ids,
    generate_community_summaries,
    save_communities,
    build_summary_index
)

load_dotenv()
logger = get_logger(__name__)


def load_config():
    """加载配置"""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_call_records(data_dir: Path):
    """加载通话记录"""
    import json

    records = []
    sources = [
        data_dir / "call_records.json",
        data_dir / "calls_rag.json",
        data_dir / "processed" / "call_records.parquet"
    ]

    for source in sources:
        if source.exists():
            if source.suffix == ".parquet":
                try:
                    import pandas as pd
                    df = pd.read_parquet(source)
                    records = df.to_dict("records")
                    # 统一字段名
                    for r in records:
                        if "text" not in r and "transcript" in r:
                            r["text"] = r["transcript"]
                    logger.info(f"从 {source} 加载 {len(records)} 条记录")
                    return records
                except Exception as e:
                    logger.warning(f"加载 Parquet 失败: {e}")
            else:
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records = data
                        # 统一字段名
                        for r in records:
                            if "text" not in r and "transcript" in r:
                                r["text"] = r["transcript"]
                    logger.info(f"从 {source} 加载 {len(records)} 条记录")

    return records


def extract_entities_from_record(record: dict, llm_client: QwenClient = None) -> dict:
    """
    从通话记录中提取实体

    Args:
        record: 通话记录
        llm_client: LLM 客户端

    Returns:
        实体字典
    """
    text = record.get("text", record.get("transcript", ""))
    category = record.get("category", "")

    entities = {
        "aspects": [],
        "concepts": [],
        "sentiment": "neutral"
    }

    # 基于关键词提取
    keywords_map = {
        "外卖": ["外卖", "配送", "骑手", "取餐"],
        "快递": ["快递", "包裹", "取件", "驿站"],
        "打车": ["打车", "网约车", "车牌", "司机"],
        "诈骗": ["转账", "验证码", "安全账户", "公安"],
        "工作": ["会议", "项目", "客户", "合同"],
        "生活": ["朋友", "家人", "聚会", "吃饭"]
    }

    for cat, words in keywords_map.items():
        for word in words:
            if word in text:
                entities["aspects"].append(cat)
                break

    # 如果有 LLM 客户端，使用 LLM 提取
    if llm_client and len(text) > 5:
        try:
            prompt = f"""从以下来电内容中提取关键实体和方面。

来电内容: {text}
分类: {category}

请提取:
- aspects: 涉及的方面/话题 (最多3个)
- concepts: 关键概念 (最多3个)
- sentiment: 情感倾向 (positive/negative/neutral)

只输出 JSON 格式:"""

            response = llm_client.generate(prompt)
            import json
            try:
                result = json.loads(response)
                entities.update(result)
            except:
                pass
        except Exception as e:
            logger.debug(f"LLM 提取失败: {e}")

    # 去重
    entities["aspects"] = list(set(entities["aspects"]))[:5]
    entities["concepts"] = list(set(entities["concepts"]))[:5]

    return entities


def build_all_indices(
    data_dir: Path,
    indices_dir: Path,
    config: dict,
    skip_vector: bool = False,
    skip_graph: bool = False,
    skip_community: bool = False,
    resume_file: str = "indices/graph_partial.jsonl"
):
    """
    构建所有索引

    Args:
        data_dir: 数据目录
        indices_dir: 索引目录
        config: 配置字典
        skip_vector: 跳过向量索引
        skip_graph: 跳过图谱
        skip_community: 跳过社区
        resume_file: 图谱构建断点续传文件
    """
    # 确保目录存在
    indices_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    records = load_call_records(data_dir)
    logger.info(f"加载 {len(records)} 条通话记录")

    if not records:
        logger.error("没有通话记录可处理")
        return

    # 加载 Embedding 模型
    embed_cfg = config.get("embedding", {})
    embedding = load_bge_embedding(
        model_name=embed_cfg.get("model", "BAAI/bge-large-zh-v1.5"),
        device=embed_cfg.get("device", "cpu"),
        batch_size=embed_cfg.get("batch_size", 32)
    )

    # ========== 1. 向量索引 ==========
    if not skip_vector:
        t0 = time.time()
        vector_dir = Path("d:/temp/call_rag_vector")

        indexer = VectorIndexer(
            embedder=embedding,
            persist_dir=str(vector_dir),
            collection_name="call_records"
        )
        indexer.build_from_records(records, doc_type="call")

        logger.info(f"[1/4] 向量索引构建完成: {indexer.count} 文档 (耗时: {time.time()-t0:.1f}s)")
    else:
        logger.info("[1/4] 跳过向量索引")

    # ========== 2. 知识图谱 ==========
    graph_path = indices_dir / "graph.pkl"
    resume_path = indices_dir / resume_file

    if not skip_graph:
        t0 = time.time()

        # 加载 LLM 客户端
        llm_client = None
        api_key = os.environ.get(config.get("qwen", {}).get("api_key_env", "QWEN_API_KEY"), "")
        if api_key:
            try:
                qwen_cfg = config.get("qwen", {})
                llm_client = QwenClient(
                    api_key=api_key,
                    model=qwen_cfg.get("model", "qwen-turbo"),
                    base_url=qwen_cfg.get("base_url"),
                    timeout=qwen_cfg.get("timeout", 30)
                )
            except Exception as e:
                logger.warning(f"LLM 客户端创建失败: {e}")

        # 构建图谱
        builder = GraphBuilder()

        # 检查断点续传
        processed_ids = set()
        if resume_path.exists():
            import json
            with open(resume_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_ids.add(data.get("id", ""))
                    except:
                        continue
            logger.info(f"断点续传: 已处理 {len(processed_ids)} 条记录")

        # 添加通话记录到图谱
        for i, record in enumerate(records):
            record_id = record.get("id", f"call_{i}")

            # 跳过已处理的记录
            if str(record_id) in processed_ids:
                continue

            # 提取实体
            entities = extract_entities_from_record(record, llm_client)

            # 添加到图谱
            builder.add_call_record(
                call_id=record_id,
                text=record.get("text", record.get("transcript", "")),
                category=record.get("category", "其他"),
                entities=entities
            )

            # 添加联系人信息
            if "caller_name" in record and record["caller_name"]:
                builder.add_contact(
                    contact_id=record_id,
                    name=record["caller_name"],
                    company=record.get("company"),
                    relation=record.get("relation", "unknown")
                )

            # 定期保存断点
            if (i + 1) % 100 == 0:
                logger.info(f"处理进度: {i+1}/{len(records)}")

        # 构建 NetworkX 图
        G = builder.build_networkx()

        if G:
            # 保存图谱
            builder.save(str(graph_path))
            logger.info(f"[2/4] 知识图谱构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边 (耗时: {time.time()-t0:.1f}s)")
        else:
            logger.warning("[2/4] 图谱构建失败")
    else:
        logger.info("[2/4] 跳过知识图谱")

    # ========== 3. 社区检测 ==========
    communities_path = indices_dir / "communities.json"

    if not skip_community:
        t0 = time.time()

        # 加载图谱
        try:
            builder = GraphBuilder()
            builder.load(str(graph_path))
            G = builder.build_networkx()
        except Exception as e:
            logger.error(f"加载图谱失败: {e}")
            G = None

        if G and G.number_of_nodes() > 0:
            # 社区检测
            communities = detect_communities(G, seed=42, min_size=3)
            annotate_community_ids(G, communities)
            builder.save(str(graph_path))

            logger.info(f"[3/4] 社区检测完成: {len(communities)} 个社区")

            # ========== 4. 社区摘要 ==========
            # 创建 call_id -> text 映射
            call_map = {
                record.get("id", f"call_{i}"): record.get("text", record.get("transcript", ""))
                for i, record in enumerate(records)
            }

            # 加载 LLM 客户端
            summary_llm = None
            api_key = os.environ.get(config.get("qwen", {}).get("api_key_env", "QWEN_API_KEY"), "")
            if api_key:
                try:
                    qwen_cfg = config.get("qwen", {})
                    summary_llm = QwenClient(
                        api_key=api_key,
                        model=qwen_cfg.get("model", "qwen-turbo"),
                        base_url=qwen_cfg.get("base_url"),
                        timeout=qwen_cfg.get("timeout", 30)
                    )
                except Exception as e:
                    logger.warning(f"LLM 客户端创建失败: {e}")

            # 生成社区摘要
            summaries = generate_community_summaries(G, communities, summary_llm, call_map)
            save_communities(summaries, str(communities_path))

            # 构建社区摘要向量索引
            summary_vector_dir = indices_dir / "summary_store"
            build_summary_index(summaries, embedding, str(summary_vector_dir))

            logger.info(f"[4/4] 社区摘要向量索引构建完成: {len(summaries)} 个摘要 (耗时: {time.time()-t0:.1f}s)")
        else:
            logger.warning("[3/4] 图谱为空，跳过社区检测")
    else:
        logger.info("[3/4] 跳过社区检测")
        logger.info("[4/4] 跳过社区摘要")


def main():
    parser = argparse.ArgumentParser(description="来电助手索引构建")
    parser.add_argument("--data-dir", "-d", default="data", help="数据目录")
    parser.add_argument("--indices-dir", "-i", default="indices", help="索引目录")
    parser.add_argument("--config", "-c", default=None, help="配置文件路径")
    parser.add_argument("--skip-vector", action="store_true", help="跳过向量索引")
    parser.add_argument("--skip-graph", action="store_true", help="跳过知识图谱")
    parser.add_argument("--skip-community", action="store_true", help="跳过社区检测")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("来电助手 Hybrid RAG 索引构建")
    logger.info("=" * 60)

    # 路径
    project_root = PROJECT_ROOT
    data_dir = project_root / args.data_dir
    indices_dir = project_root / args.indices_dir

    # 加载配置
    config = load_config()

    # 构建索引
    build_all_indices(
        data_dir=data_dir,
        indices_dir=indices_dir,
        config=config,
        skip_vector=args.skip_vector,
        skip_graph=args.skip_graph,
        skip_community=args.skip_community
    )

    logger.info("=" * 60)
    logger.info("索引构建完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
