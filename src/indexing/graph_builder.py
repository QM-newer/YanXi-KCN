"""
知识图谱构建器
==============
构建来电领域的知识图谱
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KGNode:
    """知识图谱节点"""
    id: str
    type: str  # person, company, location, concept, call_type
    name: str
    properties: Dict[str, Any] = None


@dataclass
class KGRelation:
    """知识图谱关系"""
    source: str
    target: str
    relation: str  # IS_A, WORKS_AT, CALLED, MENTIONS
    properties: Dict[str, Any] = None


class GraphBuilder:
    """
    知识图谱构建器

    构建来电助手领域的实体关系图谱
    """

    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.relations: List[KGRelation] = []
        self._graph = None

    def add_call_record(
        self,
        call_id: str,
        text: str,
        category: str,
        entities: Dict[str, Any] = None
    ) -> None:
        """
        添加通话记录到图谱

        Args:
            call_id: 通话ID
            text: 通话文本
            category: 分类
            entities: 提取的实体
        """
        # 添加通话节点
        call_node = KGNode(
            id=f"call:{call_id}",
            type="call_record",
            name=f"通话记录_{category}",
            properties={"text": text, "category": category}
        )
        self.nodes[call_node.id] = call_node

        # 添加分类节点
        cat_id = f"category:{category}"
        if cat_id not in self.nodes:
            self.nodes[cat_id] = KGNode(
                id=cat_id,
                type="category",
                name=category
            )
        self.relations.append(KGRelation(
            source=call_node.id,
            target=cat_id,
            relation="BELONGS_TO"
        ))

        # 添加实体节点和关系
        if entities:
            for entity_type, entity_list in entities.items():
                if not isinstance(entity_list, list):
                    entity_list = [entity_list]

                for entity in entity_list:
                    entity_id = f"{entity_type}:{entity}"
                    if entity_id not in self.nodes:
                        self.nodes[entity_id] = KGNode(
                            id=entity_id,
                            type=entity_type,
                            name=entity
                        )
                    self.relations.append(KGRelation(
                        source=call_node.id,
                        target=entity_id,
                        relation="MENTIONS"
                    ))

    def add_contact(
        self,
        contact_id: str,
        name: str,
        company: str = None,
        relation: str = None
    ) -> None:
        """添加联系人到图谱"""
        person_id = f"person:{contact_id}"
        self.nodes[person_id] = KGNode(
            id=person_id,
            type="person",
            name=name,
            properties={"relation": relation}
        )

        if company:
            company_id = f"company:{company}"
            self.nodes[company_id] = KGNode(
                id=company_id,
                type="company",
                name=company
            )
            self.relations.append(KGRelation(
                source=person_id,
                target=company_id,
                relation="WORKS_AT"
            ))

    def build_networkx(self) -> Any:
        """构建NetworkX图"""
        try:
            import networkx as nx
        except ImportError:
            logger.error("networkx未安装，请运行: pip install networkx")
            return None

        G = nx.Graph()

        # 添加节点
        for node in self.nodes.values():
            G.add_node(node.id, type=node.type, name=node.name, **(node.properties or {}))

        # 添加边
        for rel in self.relations:
            G.add_edge(rel.source, rel.target, relation=rel.relation)

        self._graph = G
        logger.info(f"构建图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G

    def save(self, path: str) -> None:
        """保存图谱到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "name": n.name,
                    "properties": n.properties or {}
                }
                for n in self.nodes.values()
            ],
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "properties": r.properties or {}
                }
                for r in self.relations
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"图谱保存到 {path}")

    def load(self, path: str) -> None:
        """从文件加载图谱"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"图谱文件不存在: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes = {}
        for n in data.get("nodes", []):
            self.nodes[n["id"]] = KGNode(
                id=n["id"],
                type=n["type"],
                name=n["name"],
                properties=n.get("properties")
            )

        self.relations = [
            KGRelation(
                source=r["source"],
                target=r["target"],
                relation=r["relation"],
                properties=r.get("properties")
            )
            for r in data.get("relations", [])
        ]

        logger.info(f"加载图谱: {len(self.nodes)} 节点, {len(self.relations)} 关系")

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[str]:
        """获取节点的邻居"""
        if self._graph is None:
            self.build_networkx()

        if self._graph is None:
            return []

        try:
            import networkx as nx
            return list(nx.single_source_shortest_path_length(self._graph, node_id, cutoff=depth).keys())
        except:
            return []

    def query_by_type(self, node_type: str) -> List[KGNode]:
        """按类型查询节点"""
        return [n for n in self.nodes.values() if n.type == node_type]

    def query_by_relation(self, source_type: str, relation: str) -> List[tuple]:
        """按关系查询"""
        results = []
        for rel in self.relations:
            if rel.relation == relation:
                src_node = self.nodes.get(rel.source)
                tgt_node = self.nodes.get(rel.target)
                if src_node and tgt_node and src_node.type == source_type:
                    results.append((src_node, tgt_node))
        return results
