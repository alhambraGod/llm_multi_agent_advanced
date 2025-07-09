"""
RAG引擎 - 检索增强生成的核心实现
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime

# 向量数据库
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document

# LLM
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 本地模块
from .document_processor import DocumentProcessor
from .retriever import Retriever
from .generator import Generator
from ..database.vector_db import VectorDatabase

class RAGEngine:
    """
    企业级RAG引擎
    整合文档处理、向量检索和生成功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RAG引擎
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.document_processor = DocumentProcessor(config.get('document_processor', {}))
        self.vector_db = VectorDatabase(config.get('vector_db', {}))
        self.retriever = Retriever(config.get('retriever', {}), self.vector_db)
        self.generator = Generator(config.get('generator', {}))
        
        # 嵌入模型
        embedding_config = config.get('embeddings', {})
        if embedding_config.get('provider') == 'openai':
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=embedding_config.get('api_key')
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            )
        
        # 性能监控
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'total_documents_processed': 0
        }
    
    async def initialize(self):
        """初始化RAG引擎"""
        try:
            await self.vector_db.initialize()
            await self.retriever.initialize()
            await self.generator.initialize()
            self.logger.info("RAG引擎初始化完成")
        except Exception as e:
            self.logger.error(f"RAG引擎初始化失败: {e}")
            raise
    
    async def add_documents(self, file_paths: List[str]) -> bool:
        """
        添加文档到知识库
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            是否成功
        """
        try:
            # 处理文档
            documents = await self.document_processor.process_documents(file_paths)
            
            if not documents:
                self.logger.warning("没有成功处理的文档")
                return False
            
            # 生成嵌入向量
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # 批量添加到向量数据库
            await self.vector_db.add_texts(texts, metadatas, self.embeddings)
            
            self.metrics['total_documents_processed'] += len(documents)
            self.logger.info(f"成功添加 {len(documents)} 个文档块到知识库")
            
            return True
            
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            return False
    
    async def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行RAG查询
        
        Args:
            question: 用户问题
            context: 上下文信息
            
        Returns:
            查询结果
        """
        start_time = asyncio.get_event_loop().time()
        self.metrics['total_queries'] += 1
        
        try:
            # 检索相关文档
            retrieved_docs = await self.retriever.retrieve(
                question, 
                k=self.config.get('top_k', 5),
                context=context
            )
            
            if not retrieved_docs:
                self.logger.warning(f"未找到相关文档: {question}")
                return {
                    'answer': '抱歉，我没有找到相关信息来回答您的问题。',
                    'sources': [],
                    'confidence': 0.0,
                    'response_time': asyncio.get_event_loop().time() - start_time
                }
            
            # 生成回答
            answer_result = await self.generator.generate(
                question=question,
                context_docs=retrieved_docs,
                user_context=context
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(retrieved_docs, answer_result)
            
            # 准备响应
            response = {
                'answer': answer_result['answer'],
                'sources': [{
                    'content': doc.page_content[:200] + '...',
                    'metadata': doc.metadata,
                    'score': getattr(doc, 'score', 0.0)
                } for doc in retrieved_docs],
                'confidence': confidence,
                'response_time': asyncio.get_event_loop().time() - start_time,
                'reasoning': answer_result.get('reasoning', ''),
                'suggestions': answer_result.get('suggestions', [])
            }
            
            self.metrics['successful_queries'] += 1
            self._update_avg_response_time(response['response_time'])
            
            return response
            
        except Exception as e:
            self.logger.error(f"RAG查询失败: {e}")
            self.metrics['failed_queries'] += 1
            
            return {
                'answer': '抱歉，处理您的问题时出现了错误。',
                'sources': [],
                'confidence': 0.0,
                'response_time': asyncio.get_event_loop().time() - start_time,
                'error': str(e)
            }
    
    async def batch_query(self, questions: List[str], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        批量查询
        
        Args:
            questions: 问题列表
            context: 上下文信息
            
        Returns:
            查询结果列表
        """
        tasks = [self.query(question, context) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'answer': f'处理问题时出现错误: {result}',
                    'sources': [],
                    'confidence': 0.0,
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _calculate_confidence(self, docs: List[Document], answer_result: Dict[str, Any]) -> float:
        """
        计算回答置信度
        
        Args:
            docs: 检索到的文档
            answer_result: 生成的回答结果
            
        Returns:
            置信度分数 (0-1)
        """
        if not docs:
            return 0.0
        
        # 基于检索分数的置信度
        retrieval_scores = [getattr(doc, 'score', 0.0) for doc in docs]
        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        
        # 基于生成质量的置信度
        generation_confidence = answer_result.get('confidence', 0.5)
        
        # 综合置信度
        confidence = (avg_retrieval_score * 0.6 + generation_confidence * 0.4)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _update_avg_response_time(self, response_time: float):
        """更新平均响应时间"""
        total_successful = self.metrics['successful_queries']
        if total_successful == 1:
            self.metrics['avg_response_time'] = response_time
        else:
            self.metrics['avg_response_time'] = (
                (self.metrics['avg_response_time'] * (total_successful - 1) + response_time) / total_successful
            )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.metrics,
            'success_rate': (
                self.metrics['successful_queries'] / self.metrics['total_queries'] 
                if self.metrics['total_queries'] > 0 else 0.0
            ),
            'vector_db_stats': await self.vector_db.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查各组件状态
            vector_db_healthy = await self.vector_db.health_check()
            retriever_healthy = await self.retriever.health_check()
            generator_healthy = await self.generator.health_check()
            
            overall_healthy = all([vector_db_healthy, retriever_healthy, generator_healthy])
            
            return {
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'components': {
                    'vector_db': 'healthy' if vector_db_healthy else 'unhealthy',
                    'retriever': 'healthy' if retriever_healthy else 'unhealthy',
                    'generator': 'healthy' if generator_healthy else 'unhealthy'
                },
                'metrics': await self.get_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """清理资源"""
        try:
            await self.vector_db.close()
            await self.retriever.cleanup()
            await self.generator.cleanup()
            self.logger.info("RAG引擎资源清理完成")
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")