"""
图数据库模块

支持多种图数据库：
- Neo4j
- ArangoDB
- NetworkX (内存图)
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from arango import ArangoClient
except ImportError:
    ArangoClient = None

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    id: str
    label: str
    properties: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class GraphEdge:
    """图边数据结构"""
    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any]
    weight: float = 1.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class GraphDatabaseInterface(ABC):
    """图数据库接口"""
    
    @abstractmethod
    def add_node(self, node: GraphNode) -> bool:
        """添加节点"""
        pass
    
    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> bool:
        """添加边"""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """获取节点"""
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: str, relationship: Optional[str] = None) -> List[GraphNode]:
        """获取邻居节点"""
        pass
    
    @abstractmethod
    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[str]]:
        """查找路径"""
        pass
    
    @abstractmethod
    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """执行查询"""
        pass

class Neo4jGraphDB(GraphDatabaseInterface):
    """Neo4j图数据库实现"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", password: str = "password"):
        if GraphDatabase is None:
            raise ImportError("neo4j not installed. Install with: pip install neo4j")
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # 测试连接
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
    
    def add_node(self, node: GraphNode) -> bool:
        """添加节点到Neo4j"""
        try:
            with self.driver.session() as session:
                query = f"""
                MERGE (n:{node.label} {{id: $id}})
                SET n += $properties
                SET n.created_at = $created_at
                RETURN n
                """
                
                result = session.run(query, {
                    "id": node.id,
                    "properties": node.properties,
                    "created_at": node.created_at.isoformat()
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error adding node to Neo4j: {e}")
            return False
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """添加边到Neo4j"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (a {{id: $source_id}})
                MATCH (b {{id: $target_id}})
                MERGE (a)-[r:{edge.relationship}]->(b)
                SET r += $properties
                SET r.weight = $weight
                SET r.created_at = $created_at
                RETURN r
                """
                
                result = session.run(query, {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "properties": edge.properties,
                    "weight": edge.weight,
                    "created_at": edge.created_at.isoformat()
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error adding edge to Neo4j: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """从Neo4j获取节点"""
        try:
            with self.driver.session() as session:
                query = "MATCH (n {id: $id}) RETURN n, labels(n) as labels"
                result = session.run(query, {"id": node_id})
                record = result.single()
                
                if record:
                    node_data = dict(record["n"])
                    labels = record["labels"]
                    
                    # 移除内部属性
                    node_id = node_data.pop("id")
                    created_at_str = node_data.pop("created_at", None)
                    created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
                    
                    return GraphNode(
                        id=node_id,
                        label=labels[0] if labels else "Node",
                        properties=node_data,
                        created_at=created_at
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting node from Neo4j: {e}")
            return None
    
    def get_neighbors(self, node_id: str, relationship: Optional[str] = None) -> List[GraphNode]:
        """获取邻居节点"""
        try:
            with self.driver.session() as session:
                if relationship:
                    query = f"""
                    MATCH (n {{id: $id}})-[:{relationship}]-(neighbor)
                    RETURN neighbor, labels(neighbor) as labels
                    """
                else:
                    query = """
                    MATCH (n {id: $id})-(neighbor)
                    RETURN neighbor, labels(neighbor) as labels
                    """
                
                result = session.run(query, {"id": node_id})
                
                neighbors = []
                for record in result:
                    node_data = dict(record["neighbor"])
                    labels = record["labels"]
                    
                    neighbor_id = node_data.pop("id")
                    created_at_str = node_data.pop("created_at", None)
                    created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
                    
                    neighbors.append(GraphNode(
                        id=neighbor_id,
                        label=labels[0] if labels else "Node",
                        properties=node_data,
                        created_at=created_at
                    ))
                
                return neighbors
                
        except Exception as e:
            logger.error(f"Error getting neighbors from Neo4j: {e}")
            return []
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[str]]:
        """查找路径"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH path = (start {{id: $source_id}})-[*1..{max_depth}]-(end {{id: $target_id}})
                RETURN [node in nodes(path) | node.id] as path_nodes
                LIMIT 10
                """
                
                result = session.run(query, {
                    "source_id": source_id,
                    "target_id": target_id
                })
                
                paths = []
                for record in result:
                    paths.append(record["path_nodes"])
                
                return paths
                
        except Exception as e:
            logger.error(f"Error finding path in Neo4j: {e}")
            return []
    
    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """执行Cypher查询"""
        try:
            with self.driver.session() as session:
                result = session.run(query_str)
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Error executing query in Neo4j: {e}")
            return []

class NetworkXGraphDB(GraphDatabaseInterface):
    """NetworkX内存图数据库实现"""
    
    def __init__(self, directed: bool = True):
        if nx is None:
            raise ImportError("networkx not installed. Install with: pip install networkx")
        
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.nodes_data = {}  # 存储节点详细信息
        
        logger.info(f"NetworkX graph initialized (directed: {directed})")
    
    def add_node(self, node: GraphNode) -> bool:
        """添加节点到NetworkX"""
        try:
            self.graph.add_node(node.id, **node.properties)
            self.nodes_data[node.id] = node
            return True
            
        except Exception as e:
            logger.error(f"Error adding node to NetworkX: {e}")
            return False
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """添加边到NetworkX"""
        try:
            self.graph.add_edge(
                edge.source_id, 
                edge.target_id,
                relationship=edge.relationship,
                weight=edge.weight,
                **edge.properties
            )
            return True
            
        except Exception as e:
            logger.error(f"Error adding edge to NetworkX: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """从NetworkX获取节点"""
        return self.nodes_data.get(node_id)
    
    def get_neighbors(self, node_id: str, relationship: Optional[str] = None) -> List[GraphNode]:
        """获取邻居节点"""
        try:
            if node_id not in self.graph:
                return []
            
            neighbors = []
            for neighbor_id in self.graph.neighbors(node_id):
                if relationship:
                    edge_data = self.graph.get_edge_data(node_id, neighbor_id)
                    if edge_data and edge_data.get('relationship') == relationship:
                        if neighbor_id in self.nodes_data:
                            neighbors.append(self.nodes_data[neighbor_id])
                else:
                    if neighbor_id in self.nodes_data:
                        neighbors.append(self.nodes_data[neighbor_id])
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Error getting neighbors from NetworkX: {e}")
            return []
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[str]]:
        """查找路径"""
        try:
            if source_id not in self.graph or target_id not in self.graph:
                return []
            
            # 使用NetworkX的所有简单路径算法
            paths = list(nx.all_simple_paths(
                self.graph, source_id, target_id, cutoff=max_depth
            ))
            
            return paths[:10]  # 限制返回路径数量
            
        except Exception as e:
            logger.error(f"Error finding path in NetworkX: {e}")
            return []
    
    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """执行简单查询（基于节点属性）"""
        try:
            # 简化的查询实现，支持基本的属性匹配
            results = []
            
            # 解析简单的查询格式："label:value"
            if ":" in query_str:
                key, value = query_str.split(":", 1)
                for node_id, node_data in self.nodes_data.items():
                    if node_data.properties.get(key) == value:
                        results.append({
                            "node_id": node_id,
                            "node": asdict(node_data)
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing query in NetworkX: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取图统计信息"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else False
        }

class GraphDatabase:
    """图数据库统一接口"""
    
    def __init__(self, db_type: str = "networkx", **kwargs):
        self.db_type = db_type
        
        if db_type == "neo4j":
            self.db = Neo4jGraphDB(**kwargs)
        elif db_type == "networkx":
            self.db = NetworkXGraphDB(**kwargs)
        elif db_type == "arangodb":
            if ArangoClient is None:
                raise ImportError("python-arango not installed")
            # TODO: 实现ArangoDB支持
            raise NotImplementedError("ArangoDB support coming soon")
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        logger.info(f"GraphDatabase initialized with type: {db_type}")
    
    def add_node(self, node: GraphNode) -> bool:
        """添加节点"""
        return self.db.add_node(node)
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """添加边"""
        return self.db.add_edge(edge)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """获取节点"""
        return self.db.get_node(node_id)
    
    def get_neighbors(self, node_id: str, relationship: Optional[str] = None) -> List[GraphNode]:
        """获取邻居节点"""
        return self.db.get_neighbors(node_id, relationship)
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[str]]:
        """查找路径"""
        return self.db.find_path(source_id, target_id, max_depth)
    
    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """执行查询"""
        return self.db.query(query_str)
    
    def build_knowledge_graph(self, entities: List[Dict[str, Any]], 
                            relationships: List[Dict[str, Any]]) -> bool:
        """构建知识图谱"""
        try:
            # 添加实体节点
            for entity in entities:
                node = GraphNode(
                    id=entity["id"],
                    label=entity.get("type", "Entity"),
                    properties=entity.get("properties", {})
                )
                self.add_node(node)
            
            # 添加关系边
            for rel in relationships:
                edge = GraphEdge(
                    source_id=rel["source"],
                    target_id=rel["target"],
                    relationship=rel["type"],
                    properties=rel.get("properties", {}),
                    weight=rel.get("weight", 1.0)
                )
                self.add_edge(edge)
            
            logger.info(f"Built knowledge graph with {len(entities)} entities and {len(relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {"db_type": self.db_type}
        
        if hasattr(self.db, 'get_stats'):
            stats.update(self.db.get_stats())
        
        return stats
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 尝试添加和查询一个测试节点
            test_node = GraphNode(
                id="health_check_node",
                label="Test",
                properties={"test": True}
            )
            
            self.add_node(test_node)
            result = self.get_node("health_check_node")
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Graph database health check failed: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self.db, 'close'):
            self.db.close()