"""
检索器 - 负责从向量数据库中检索相关文档
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from langchain.schema import Document
from langchain.retrievers import (
    VectorStoreRetriever,
    ContextualCompressionRetriever,
    EnsembleRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

class Retriever:
    """
    企业级检索器
    支持多种检索策略和优化技术
    """
    
    def __init__(self, config: Dict[str, Any], vector_db):
        """
        初始化检索器
        
        Args:
            config: 配置参数
            vector_db: 向量数据库实例
        """
        self.config = config
        self.vector_db = vector_db
        self.logger = logging.getLogger(__name__)
        
        # 检索策略
        self.retrieval_strategy = config.get('strategy', 'similarity')
        self.use_compression = config.get('use_compression', True)
        self.use_multi_query = config.get('use_multi_query', False)
        
        # 性能参数
        self.max_concurrent_queries = config.get('max_concurrent_queries', 10)
        self.timeout = config.get('timeout', 30.0)
        
        # 缓存
        self.query_cache = {}
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1小时
        
        # 检索器实例
        self.base_retriever = None
        self.compressed_retriever = None
        self.multi_query_retriever = None
    
    async def initialize(self):
        """初始化检索器"""
        try:
            # 获取向量存储
            vector_store = await self.vector_db.get_vector_store()
            
            # 基础检索器
            self.base_retriever = VectorStoreRetriever(
                vectorstore=vector_store,
                search_type=self.retrieval_strategy,
                search_kwargs={
                    'k': self.config.get('top_k', 10),
                    'score_threshold': self.config.get('score_threshold', 0.5)
                }
            )
            
            # 压缩检索器
            if self.use_compression:
                compressor = LLMChainExtractor.from_llm(
                    llm=self.vector_db.get_llm()
                )
                self.compressed_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.base_retriever
                )
            
            # 多查询检索器
            if self.use_multi_query:
                self.multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=self.base_retriever,
                    llm=self.vector_db.get_llm()
                )
            
            self.logger.info("检索器初始化完成")
            
        except Exception as e:
            self.logger.error(f"检索器初始化失败: {e}")
            raise
    
    async def retrieve(self, 
                      query: str, 
                      k: int = 5, 
                      context: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回文档数量
            context: 上下文信息
            
        Returns:
            相关文档列表
        """
        try:
            # 检查缓存
            cache_key = self._get_cache_key(query, k, context)
            if cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    self.logger.debug(f"使用缓存结果: {query[:50]}...")
                    return cache_entry['documents']
            
            # 预处理查询
            processed_query = await self._preprocess_query(query, context)
            
            # 选择检索策略
            documents = await self._execute_retrieval(processed_query, k, context)
            
            # 后处理文档
            processed_documents = await self._postprocess_documents(documents, query, context)
            
            # 缓存结果
            self.query_cache[cache_key] = {
                'documents': processed_documents,
                'timestamp': datetime.now().timestamp()
            }
            
            self.logger.info(f"检索到 {len(processed_documents)} 个相关文档")
            return processed_documents
            
        except Exception as e:
            self.logger.error(f"文档检索失败: {e}")
            return []
    
    async def _preprocess_query(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """
        预处理查询文本
        
        Args:
            query: 原始查询
            context: 上下文信息
            
        Returns:
            处理后的查询
        """
        processed_query = query.strip()
        
        # 添加上下文信息
        if context:
            user_profile = context.get('user_profile', {})
            if user_profile:
                # 根据用户画像调整查询
                fitness_level = user_profile.get('fitness_level', '')
                goals = user_profile.get('goals', [])
                
                if fitness_level:
                    processed_query += f" (适合{fitness_level}水平)"
                
                if goals:
                    processed_query += f" (目标: {', '.join(goals)})"
        
        return processed_query
    
    async def _execute_retrieval(self, 
                                query: str, 
                                k: int, 
                                context: Optional[Dict[str, Any]]) -> List[Document]:
        """
        执行检索
        
        Args:
            query: 查询文本
            k: 返回文档数量
            context: 上下文信息
            
        Returns:
            检索到的文档
        """
        try:
            # 根据配置选择检索器
            if self.use_multi_query and self.multi_query_retriever:
                documents = await asyncio.wait_for(
                    self._async_retrieve(self.multi_query_retriever, query),
                    timeout=self.timeout
                )
            elif self.use_compression and self.compressed_retriever:
                documents = await asyncio.wait_for(
                    self._async_retrieve(self.compressed_retriever, query),
                    timeout=self.timeout
                )
            else:
                documents = await asyncio.wait_for(
                    self._async_retrieve(self.base_retriever, query),
                    timeout=self.timeout
                )
            
            # 限制返回数量
            return documents[:k]
            
        except asyncio.TimeoutError:
            self.logger.warning(f"检索超时: {query[:50]}...")
            return []
        except Exception as e:
            self.logger.error(f"检索执行失败: {e}")
            return []
    
    async def _async_retrieve(self, retriever, query: str) -> List[Document]:
        """异步检索包装器"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, retriever.get_relevant_documents, query)
    
    async def _postprocess_documents(self, 
                                   documents: List[Document], 
                                   query: str, 
                                   context: Optional[Dict[str, Any]]) -> List[Document]:
        """
        后处理文档
        
        Args:
            documents: 原始文档列表
            query: 查询文本
            context: 上下文信息
            
        Returns:
            处理后的文档列表
        """
        if not documents:
            return documents
        
        # 去重
        unique_documents = self._deduplicate_documents(documents)
        
        # 重新排序
        ranked_documents = await self._rerank_documents(unique_documents, query, context)
        
        # 添加检索元数据
        for i, doc in enumerate(ranked_documents):
            doc.metadata.update({
                'retrieval_rank': i,
                'retrieval_timestamp': datetime.now().isoformat(),
                'query': query[:100]  # 截断查询以节省空间
            })
        
        return ranked_documents
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """去除重复文档"""
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    async def _rerank_documents(self, 
                              documents: List[Document], 
                              query: str, 
                              context: Optional[Dict[str, Any]]) -> List[Document]:
        """
        重新排序文档
        
        Args:
            documents: 文档列表
            query: 查询文本
            context: 上下文信息
            
        Returns:
            重新排序的文档列表
        """
        # 简单的重排序策略：基于文档新鲜度和相关性
        def score_document(doc: Document) -> float:
            base_score = getattr(doc, 'score', 0.5)
            
            # 时间衰减因子
            processed_at = doc.metadata.get('processed_at', 0)
            current_time = datetime.now().timestamp()
            time_diff = current_time - processed_at
            time_decay = max(0.1, 1.0 - (time_diff / (30 * 24 * 3600)))  # 30天衰减
            
            # 文档类型权重
            file_type = doc.metadata.get('file_type', '')
            type_weight = {
                '.pdf': 1.0,
                '.docx': 0.9,
                '.md': 0.8,
                '.txt': 0.7,
                '.html': 0.6
            }.get(file_type, 0.5)
            
            return base_score * time_decay * type_weight
        
        # 排序
        scored_docs = [(doc, score_document(doc)) for doc in documents]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]
    
    def _get_cache_key(self, query: str, k: int, context: Optional[Dict[str, Any]]) -> str:
        """生成缓存键"""
        context_str = str(sorted(context.items())) if context else ""
        return f"{hash(query)}_{k}_{hash(context_str)}"
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """检查缓存是否有效"""
        current_time = datetime.now().timestamp()
        return (current_time - cache_entry['timestamp']) < self.cache_ttl
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.base_retriever:
                return False
            
            # 执行简单的检索测试
            test_docs = await self.retrieve("健康检查测试", k=1)
            return True
            
        except Exception as e:
            self.logger.error(f"检索器健康检查失败: {e}")
            return False
    
    async def cleanup(self):
        """清理资源"""
        try:
            self.query_cache.clear()
            self.logger.info("检索器资源清理完成")
        except Exception as e:
            self.logger.error(f"检索器资源清理失败: {e}")