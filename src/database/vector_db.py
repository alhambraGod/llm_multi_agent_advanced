"""
向量数据库模块

支持多种向量数据库：
- Chroma
- Pinecone
- Weaviate
- Qdrant
- FAISS
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    import pinecone
except ImportError:
    pinecone = None

try:
    import weaviate
except ImportError:
    weaviate = None

try:
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    qdrant_client = None

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)

@dataclass
class VectorDocument:
    """向量文档数据结构"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class VectorDatabaseInterface(ABC):
    """向量数据库接口"""
    
    @abstractmethod
    def add_documents(self, documents: List[VectorDocument]) -> bool:
        """添加文档"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VectorDocument, float]]:
        """搜索相似文档"""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def update_document(self, document: VectorDocument) -> bool:
        """更新文档"""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """获取文档"""
        pass

class ChromaVectorDB(VectorDatabaseInterface):
    """Chroma向量数据库实现"""
    
    def __init__(self, collection_name: str = "fitness_knowledge", 
                 persist_directory: str = "./chroma_db"):
        if chromadb is None:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        logger.info(f"ChromaVectorDB initialized with collection: {collection_name}")
    
    def add_documents(self, documents: List[VectorDocument]) -> bool:
        """添加文档到Chroma"""
        try:
            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            metadatas = [{
                **doc.metadata,
                "content": doc.content,
                "timestamp": doc.timestamp.isoformat()
            } for doc in documents]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to Chroma")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VectorDocument, float]]:
        """在Chroma中搜索相似文档"""
        try:
            where_clause = filters if filters else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            documents = []
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                content = metadata.pop('content', '')
                timestamp_str = metadata.pop('timestamp', None)
                timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
                
                doc = VectorDocument(
                    id=doc_id,
                    content=content,
                    embedding=results['embeddings'][0][i] if results['embeddings'] else [],
                    metadata=metadata,
                    timestamp=timestamp
                )
                
                score = results['distances'][0][i]
                documents.append((doc, 1 - score))  # 转换为相似度分数
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching in Chroma: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """从Chroma删除文档"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from Chroma")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from Chroma: {e}")
            return False
    
    def update_document(self, document: VectorDocument) -> bool:
        """更新Chroma中的文档"""
        try:
            # Chroma不支持直接更新，需要先删除再添加
            self.delete_documents([document.id])
            return self.add_documents([document])
            
        except Exception as e:
            logger.error(f"Error updating document in Chroma: {e}")
            return False
    
    def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """从Chroma获取文档"""
        try:
            results = self.collection.get(ids=[document_id])
            
            if not results['ids']:
                return None
            
            metadata = results['metadatas'][0]
            content = metadata.pop('content', '')
            timestamp_str = metadata.pop('timestamp', None)
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
            
            return VectorDocument(
                id=document_id,
                content=content,
                embedding=results['embeddings'][0] if results['embeddings'] else [],
                metadata=metadata,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Error getting document from Chroma: {e}")
            return None

class FAISSVectorDB(VectorDatabaseInterface):
    """FAISS向量数据库实现"""
    
    def __init__(self, dimension: int = 768, index_type: str = "IVFFlat", 
                 nlist: int = 100, persist_path: str = "./faiss_index"):
        if faiss is None:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")
        
        self.dimension = dimension
        self.persist_path = persist_path
        self.documents = {}  # 存储文档元数据
        
        # 创建FAISS索引
        if index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        # 尝试加载已存在的索引
        self._load_index()
        
        logger.info(f"FAISSVectorDB initialized with dimension: {dimension}")
    
    def _load_index(self):
        """加载已存在的索引"""
        try:
            if os.path.exists(f"{self.persist_path}.index"):
                self.index = faiss.read_index(f"{self.persist_path}.index")
                
            if os.path.exists(f"{self.persist_path}.metadata"):
                with open(f"{self.persist_path}.metadata", 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                    
            logger.info("Loaded existing FAISS index")
            
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    
    def _save_index(self):
        """保存索引"""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            faiss.write_index(self.index, f"{self.persist_path}.index")
            
            with open(f"{self.persist_path}.metadata", 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2, default=str)
                
            logger.info("Saved FAISS index")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def add_documents(self, documents: List[VectorDocument]) -> bool:
        """添加文档到FAISS"""
        try:
            embeddings = np.array([doc.embedding for doc in documents]).astype('float32')
            
            # 如果索引需要训练
            if not self.index.is_trained:
                self.index.train(embeddings)
            
            # 添加向量
            start_id = self.index.ntotal
            self.index.add(embeddings)
            
            # 存储文档元数据
            for i, doc in enumerate(documents):
                self.documents[str(start_id + i)] = {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "timestamp": doc.timestamp.isoformat()
                }
            
            self._save_index()
            logger.info(f"Added {len(documents)} documents to FAISS")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VectorDocument, float]]:
        """在FAISS中搜索相似文档"""
        try:
            query_vector = np.array([query_embedding]).astype('float32')
            scores, indices = self.index.search(query_vector, top_k)
            
            documents = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue
                    
                doc_data = self.documents.get(str(idx))
                if doc_data:
                    # 应用过滤器
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if doc_data["metadata"].get(key) != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    doc = VectorDocument(
                        id=doc_data["id"],
                        content=doc_data["content"],
                        embedding=query_embedding,  # 简化处理
                        metadata=doc_data["metadata"],
                        timestamp=datetime.fromisoformat(doc_data["timestamp"])
                    )
                    
                    # FAISS返回的是距离，转换为相似度
                    similarity = 1 / (1 + scores[0][i])
                    documents.append((doc, similarity))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching in FAISS: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """FAISS不支持直接删除，需要重建索引"""
        logger.warning("FAISS does not support direct deletion. Consider rebuilding index.")
        return False
    
    def update_document(self, document: VectorDocument) -> bool:
        """FAISS不支持直接更新"""
        logger.warning("FAISS does not support direct updates. Consider rebuilding index.")
        return False
    
    def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """从FAISS获取文档"""
        try:
            for idx, doc_data in self.documents.items():
                if doc_data["id"] == document_id:
                    return VectorDocument(
                        id=doc_data["id"],
                        content=doc_data["content"],
                        embedding=[],  # FAISS不存储原始向量
                        metadata=doc_data["metadata"],
                        timestamp=datetime.fromisoformat(doc_data["timestamp"])
                    )
            return None
            
        except Exception as e:
            logger.error(f"Error getting document from FAISS: {e}")
            return None

class VectorDatabase:
    """向量数据库统一接口"""
    
    def __init__(self, db_type: str = "chroma", **kwargs):
        self.db_type = db_type
        
        if db_type == "chroma":
            self.db = ChromaVectorDB(**kwargs)
        elif db_type == "faiss":
            self.db = FAISSVectorDB(**kwargs)
        elif db_type == "pinecone":
            if pinecone is None:
                raise ImportError("pinecone not installed")
            # TODO: 实现Pinecone支持
            raise NotImplementedError("Pinecone support coming soon")
        elif db_type == "weaviate":
            if weaviate is None:
                raise ImportError("weaviate not installed")
            # TODO: 实现Weaviate支持
            raise NotImplementedError("Weaviate support coming soon")
        elif db_type == "qdrant":
            if qdrant_client is None:
                raise ImportError("qdrant-client not installed")
            # TODO: 实现Qdrant支持
            raise NotImplementedError("Qdrant support coming soon")
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        logger.info(f"VectorDatabase initialized with type: {db_type}")
    
    def add_documents(self, documents: List[VectorDocument]) -> bool:
        """添加文档"""
        return self.db.add_documents(documents)
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VectorDocument, float]]:
        """搜索相似文档"""
        return self.db.search(query_embedding, top_k, filters)
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        return self.db.delete_documents(document_ids)
    
    def update_document(self, document: VectorDocument) -> bool:
        """更新文档"""
        return self.db.update_document(document)
    
    def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """获取文档"""
        return self.db.get_document(document_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {
            "db_type": self.db_type,
            "total_documents": 0
        }
        
        if hasattr(self.db, 'collection'):
            stats["total_documents"] = self.db.collection.count()
        elif hasattr(self.db, 'index'):
            stats["total_documents"] = self.db.index.ntotal
        
        return stats
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 尝试执行一个简单的搜索
            dummy_embedding = [0.0] * 768  # 假设768维
            self.search(dummy_embedding, top_k=1)
            return True
        except Exception as e:
            logger.error(f"Vector database health check failed: {e}")
            return False