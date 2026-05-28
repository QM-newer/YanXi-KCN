"""
图检索器
========
基于社区检测和摘要的图谱检索
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphRetriever:
    """
    图检索器

    检索策略:
    1. 基于查询关键词匹配社区摘要
    2. 返回社区内最相关的通话记录
    """

    def __init__(
        self,
        graph_path: str,
        communities_path: str,
        embedding=None,
        summary_db=None,
        top_k: int = 5
    ):
        """
        初始化图检索器

        Args:
            graph_path: 图谱文件路径 (.pkl)
            communities_path: 社区摘要文件路径 (.json)
            embedding: 嵌入模型
            summary_db: 社区摘要向量库
            top_k: 返回数量
        """
        self.graph_path = graph_path
        self.communities_path = communities_path
        self.embedding = embedding
        self.summary_db = summary_db
        self.top_k = top_k

        self.graph = None
        self.communities = None
        self._load()

    def _load(self):
        """加载图谱和社区数据"""
        # 加载图谱（支持 JSON 和 pickle 格式）
        graph_file = Path(self.graph_path)
        if graph_file.exists():
            try:
                # 根据扩展名或用 fallback 方式决定加载方式
                if graph_file.suffix == '.pkl':
                    # 先尝试 pickle 加载
                    try:
                        import pickle
                        with open(graph_file, 'rb') as f:
                            self.graph = pickle.load(f)
                        logger.info(f"加载图谱(pickle): {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
                    except (pickle.UnpicklingError, UnicodeDecodeError, EOFError):
                        # 回退到 JSON（可能文件内容就是 JSON 但扩展名是 .pkl）
                        with open(graph_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        self._build_graph_from_json(data)
                        logger.info(f"加载图谱(JSON回退): {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
                else:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self._build_graph_from_json(data)
                    logger.info(f"加载图谱(JSON): {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
            except Exception as e:
                logger.warning(f"图谱加载失败: {e}")

        # 加载社区 (支持列表和字典两种格式)
        if Path(self.communities_path).exists():
            try:
                with open(self.communities_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 如果是列表格式，转换为字典
                if isinstance(data, list):
                    self.communities = {str(c["community_id"]): c for c in data}
                else:
                    self.communities = data

                logger.info(f"加载社区: {len(self.communities)} 个")
            except Exception as e:
                logger.warning(f"社区加载失败: {e}")

    def _build_graph_from_json(self, data: dict) -> None:
        """从 JSON 数据构建 NetworkX 图"""
        import networkx as nx
        G = nx.Graph()
        for node in data.get('nodes', []):
            G.add_node(node['id'], **node.get('properties', {}))
        for rel in data.get('relations', []):
            G.add_edge(rel['source'], rel['target'])
        self.graph = G

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        图谱检索

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            文档列表
        """
        k = k or self.top_k
        docs = []

        if self.communities is None:
            logger.warning("社区数据未加载")
            return docs

        # 1. 优先尝试 summary_db 向量检索
        if self.summary_db is not None:
            try:
                summary_docs = self.summary_db.similarity_search(query, k=3)
                matched_communities = []
                for doc in summary_docs:
                    cid_val = doc.metadata.get("community_id")
                    if cid_val is not None:
                        matched_communities.append((cid_val, 1.0))
            except Exception as e:
                logger.warning(f"向量摘要检索失败: {e}")
                matched_communities = self._find_relevant_communities(query, k=3)
        else:
            matched_communities = self._find_relevant_communities(query, k=3)

        if not matched_communities:
            logger.info("未找到匹配社区，使用关键词匹配")
            return self._keyword_match(query, k=k)

        # 2. 从匹配社区收集文档
        for cid, score in matched_communities:
            community = self.communities.get(str(cid), {})
            nodes = community.get('nodes', [])
            categories = community.get('categories', [])
            summary = community.get('summary', '')

            # 如果社区有节点，收集节点文档
            if nodes:
                for node_id in nodes[:5]:  # 每个社区最多5个节点
                    doc = self._node_to_document(node_id, score)
                    if doc:
                        docs.append(doc)
            # 否则直接使用社区摘要
            elif summary:
                cats_str = ', '.join(categories) if categories else ''
                content = f"[社区{cid}] {cats_str}\n{summary}" if cats_str else summary
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "community_id": cid,
                        "categories": categories,
                        "score": score
                    }
                ))

            if len(docs) >= k:
                break

        logger.info(f"图谱检索返回 {len(docs)} 个文档")
        return docs

    @staticmethod
    def _tokenize(text: str) -> set:
        """分词：提取中文双字以上词组和英文单词"""
        import re
        tokens = set()
        # 中文连续序列 (2+ 字符) 
        for m in re.finditer(r'[\u4e00-\u9fff]{2,}', text):
            tokens.add(m.group())
        # 英文单词 (2+ 字符)
        for m in re.finditer(r'[a-zA-Z]{2,}', text):
            tokens.add(m.group())
        return tokens

    def _find_relevant_communities(self, query: str, k: int = 3) -> List[tuple]:
        """找到最相关的社区"""
        query_lower = query.lower()
        query_words = self._tokenize(query_lower)

        scored_communities = []

        for cid_str, community in self.communities.items():
            score = 0.0

            # 匹配摘要内容
            summary = community.get('summary', '').lower()
            if summary:
                # 关键词匹配
                for word in query_words:
                    if word in summary:
                        score += 1.0

                # 类别匹配 (categories 是列表)
                categories = community.get('categories', [])
                for cat in categories:
                    if cat.lower() in query_lower:
                        score += 2.0
                    for word in query_words:
                        if word in cat.lower():
                            score += 0.5

            # 样本通话匹配
            sample_calls = community.get('sample_calls', [])
            for call in sample_calls:
                call_content = call.lower() if isinstance(call, str) else ''
                for word in query_words:
                    if word in call_content:
                        score += 0.3

            if score > 0:
                try:
                    cid = int(cid_str)
                except (ValueError, TypeError):
                    cid = cid_str
                scored_communities.append((cid, score))

        # 按分数排序
        scored_communities.sort(key=lambda x: x[1], reverse=True)
        return scored_communities[:k]

    def _keyword_match(self, query: str, k: int = 5) -> List[Document]:
        """关键词匹配 - 从社区摘要中匹配"""
        docs = []
        query_lower = query.lower()
        query_words = self._tokenize(query_lower)

        if self.communities is None:
            return docs

        # 从社区摘要中匹配
        for cid_str, community in self.communities.items():
            score = 0.0

            # 匹配摘要内容
            summary = community.get('summary', '').lower()
            if summary:
                for word in query_words:
                    if word in summary:
                        score += 1.0

            # 匹配类别
            categories = community.get('categories', [])
            for cat in categories:
                if cat.lower() in query_lower:
                    score += 2.0
                for word in query_words:
                    if word in cat.lower():
                        score += 0.5

            if score > 0:
                try:
                    cid = int(cid_str)
                except (ValueError, TypeError):
                    cid = cid_str
                # 从摘要生成一个文档
                cats_str = ', '.join(categories) if categories else ''
                content = f"[社区{cid}] {cats_str}\n{summary}" if cats_str else summary
                doc = Document(
                    page_content=content,
                    metadata={
                        "community_id": cid,
                        "categories": categories,
                        "score": score
                    }
                )
                docs.append((doc, score))

        # 按分数排序
        docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in docs[:k]]

    def _node_to_document(self, node_id: str, community_score: float = 0.0) -> Optional[Document]:
        """将节点转换为 Document"""
        if self.graph is None:
            return None

        if node_id not in self.graph.nodes:
            return None

        data = self.graph.nodes[node_id]

        content = data.get('text', '')
        if not content:
            return None

        return Document(
            page_content=content,
            metadata={
                "node_id": node_id,
                "category": data.get('category', ''),
                "community_score": community_score
            }
        )

    def get_community_summary(self, community_id: int) -> Optional[str]:
        """获取社区摘要"""
        community = self.communities.get(str(community_id), {})
        return community.get('summary', '')


def load_graph_retriever(
    graph_path: str,
    communities_path: str,
    embedding=None,
    summary_db=None
) -> GraphRetriever:
    """工厂函数：加载图检索器"""
    return GraphRetriever(
        graph_path=graph_path,
        communities_path=communities_path,
        embedding=embedding,
        summary_db=summary_db
    )
