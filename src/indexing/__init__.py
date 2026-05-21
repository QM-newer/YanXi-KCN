"""
索引构建模块
============
向量索引、知识图谱和社区检测的构建
参考 RAG-CITY 项目设计
"""

from src.indexing.vector_builder import VectorIndexer, build_vector_index, load_vector_index
from src.indexing.graph_builder import GraphBuilder
from src.indexing.embeddings import BGEEmbedder, load_bge_embedding
from src.indexing.community_builder import (
    detect_communities,
    annotate_community_ids,
    generate_community_summaries,
    save_communities,
    load_communities,
    build_summary_index
)

__all__ = [
    "VectorIndexer",
    "build_vector_index",
    "load_vector_index",
    "GraphBuilder",
    "BGEEmbedder",
    "load_bge_embedding",
    "detect_communities",
    "annotate_community_ids",
    "generate_community_summaries",
    "save_communities",
    "load_communities",
    "build_summary_index",
]
