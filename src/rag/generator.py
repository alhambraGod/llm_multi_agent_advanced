"""
生成器 - 负责基于检索到的文档生成回答
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json

# LLM相关
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.callbacks import AsyncCallbackHandler

# 本地模块
from ..utils.prompt_templates import PromptTemplateManager

class StreamingCallbackHandler(AsyncCallbackHandler):
    """流式输出回调处理器"""
    
    def __init__(self):
        self.tokens = []
        self.is_streaming = False
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """处理新token"""
        self.tokens.append(token)
        self.is_streaming = True
    
    def get_response(self) -> str:
        """获取完整响应"""
        return ''.join(self.tokens)
    
    def reset(self):
        """重置状态"""
        self.tokens = []
        self.is_streaming = False

class Generator:
    """
    企业级生成器
    支持多种LLM模型和生成策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化生成器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # LLM配置
        self.model_provider = config.get('provider', 'openai')
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        
        # 生成策略
        self.generation_strategy = config.get('strategy', 'standard')
        self.use_streaming = config.get('use_streaming', False)
        self.enable_reasoning = config.get('enable_reasoning', True)
        
        # 提示词管理器
        self.prompt_manager = PromptTemplateManager(config.get('prompts', {}))
        
        # LLM实例
        self.llm = None
        self.chat_llm = None
        
        # 回调处理器
        self.streaming_handler = StreamingCallbackHandler()
        
        # 性能监控
        self.metrics = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'avg_generation_time': 0.0,
            'total_tokens_generated': 0
        }
    
    async def initialize(self):
        """初始化生成器"""
        try:
            # 初始化LLM
            if self.model_provider == 'openai':
                self.llm = OpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    streaming=self.use_streaming,
                    callbacks=[self.streaming_handler] if self.use_streaming else None
                )
                
                self.chat_llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    streaming=self.use_streaming,
                    callbacks=[self.streaming_handler] if self.use_streaming else None
                )
            
            elif self.model_provider == 'anthropic':
                self.chat_llm = ChatAnthropic(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            # 初始化提示词模板
            await self.prompt_manager.initialize()
            
            self.logger.info(f"生成器初始化完成 - 模型: {self.model_provider}/{self.model_name}")
            
        except Exception as e:
            self.logger.error(f"生成器初始化失败: {e}")
            raise
    
    async def generate(self, 
                      question: str, 
                      context_docs: List[Document], 
                      user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成回答
        
        Args:
            question: 用户问题
            context_docs: 上下文文档
            user_context: 用户上下文
            
        Returns:
            生成结果
        """
        start_time = asyncio.get_event_loop().time()
        self.metrics['total_generations'] += 1
        
        try:
            # 准备上下文
            context = self._prepare_context(context_docs, user_context)
            
            # 选择生成策略
            if self.generation_strategy == 'reasoning':
                result = await self._generate_with_reasoning(question, context, user_context)
            elif self.generation_strategy == 'multi_step':
                result = await self._generate_multi_step(question, context, user_context)
            else:
                result = await self._generate_standard(question, context, user_context)
            
            # 后处理
            processed_result = await self._postprocess_result(result, question, context_docs)
            
            # 更新指标
            generation_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(generation_time, processed_result)
            
            self.metrics['successful_generations'] += 1
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"生成失败: {e}")
            self.metrics['failed_generations'] += 1
            
            return {
                'answer': '抱歉，生成回答时出现了错误。',
                'confidence': 0.0,
                'reasoning': '',
                'suggestions': [],
                'error': str(e)
            }
    
    def _prepare_context(self, 
                        context_docs: List[Document], 
                        user_context: Optional[Dict[str, Any]]) -> str:
        """
        准备上下文信息
        
        Args:
            context_docs: 上下文文档
            user_context: 用户上下文
            
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 添加文档上下文
        for i, doc in enumerate(context_docs[:5]):  # 限制文档数量
            source = doc.metadata.get('source', '未知来源')
            context_parts.append(f"文档 {i+1} (来源: {source}):\n{doc.page_content}\n")
        
        # 添加用户上下文
        if user_context:
            user_profile = user_context.get('user_profile', {})
            if user_profile:
                context_parts.append(f"用户信息: {json.dumps(user_profile, ensure_ascii=False)}")
        
        return "\n".join(context_parts)
    
    async def _generate_standard(self, 
                               question: str, 
                               context: str, 
                               user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        标准生成策略
        
        Args:
            question: 问题
            context: 上下文
            user_context: 用户上下文
            
        Returns:
            生成结果
        """
        # 获取提示词模板
        template = await self.prompt_manager.get_template('fitness_qa')
        
        # 构建提示词
        prompt = template.format(
            question=question,
            context=context,
            user_info=self._format_user_info(user_context)
        )
        
        # 生成回答
        if self.chat_llm:
            messages = [
                SystemMessage(content="你是一个专业的运动健身助手，请基于提供的信息回答用户问题。"),
                HumanMessage(content=prompt)
            ]
            response = await self.chat_llm.agenerate([messages])
            answer = response.generations[0][0].text
        else:
            response = await self.llm.agenerate([prompt])
            answer = response.generations[0][0].text
        
        return {
            'answer': answer.strip(),
            'confidence': 0.8,  # 标准策略的默认置信度
            'reasoning': '',
            'suggestions': []
        }
    
    async def _generate_with_reasoning(self, 
                                     question: str, 
                                     context: str, 
                                     user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        带推理的生成策略
        
        Args:
            question: 问题
            context: 上下文
            user_context: 用户上下文
            
        Returns:
            生成结果
        """
        # 第一步：分析问题
        analysis_template = await self.prompt_manager.get_template('question_analysis')
        analysis_prompt = analysis_template.format(
            question=question,
            context=context
        )
        
        analysis_response = await self.chat_llm.agenerate([
            [SystemMessage(content="分析用户问题的关键要素和意图。"),
             HumanMessage(content=analysis_prompt)]
        ])
        analysis = analysis_response.generations[0][0].text
        
        # 第二步：推理回答
        reasoning_template = await self.prompt_manager.get_template('reasoning_qa')
        reasoning_prompt = reasoning_template.format(
            question=question,
            context=context,
            analysis=analysis,
            user_info=self._format_user_info(user_context)
        )
        
        reasoning_response = await self.chat_llm.agenerate([
            [SystemMessage(content="基于分析结果，进行逐步推理并生成回答。"),
             HumanMessage(content=reasoning_prompt)]
        ])
        reasoning_result = reasoning_response.generations[0][0].text
        
        # 解析推理结果
        try:
            parsed_result = json.loads(reasoning_result)
            return {
                'answer': parsed_result.get('answer', ''),
                'confidence': parsed_result.get('confidence', 0.7),
                'reasoning': parsed_result.get('reasoning', ''),
                'suggestions': parsed_result.get('suggestions', [])
            }
        except json.JSONDecodeError:
            # 如果解析失败，返回原始结果
            return {
                'answer': reasoning_result,
                'confidence': 0.6,
                'reasoning': analysis,
                'suggestions': []
            }
    
    async def _generate_multi_step(self, 
                                 question: str, 
                                 context: str, 
                                 user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        多步骤生成策略
        
        Args:
            question: 问题
            context: 上下文
            user_context: 用户上下文
            
        Returns:
            生成结果
        """
        steps = []
        
        # 步骤1：理解问题
        understanding_prompt = f"请理解以下问题的核心需求：{question}"
        understanding = await self._single_generation(understanding_prompt)
        steps.append(f"问题理解: {understanding}")
        
        # 步骤2：分析上下文
        context_analysis_prompt = f"基于以下上下文信息，分析与问题相关的关键点：\n{context}"
        context_analysis = await self._single_generation(context_analysis_prompt)
        steps.append(f"上下文分析: {context_analysis}")
        
        # 步骤3：生成回答
        final_prompt = f"""
        基于以下信息生成最终回答：
        问题: {question}
        问题理解: {understanding}
        上下文分析: {context_analysis}
        用户信息: {self._format_user_info(user_context)}
        
        请提供专业、准确、个性化的回答。
        """
        
        final_answer = await self._single_generation(final_prompt)
        
        return {
            'answer': final_answer,
            'confidence': 0.85,
            'reasoning': ' -> '.join(steps),
            'suggestions': await self._generate_suggestions(question, final_answer)
        }
    
    async def _single_generation(self, prompt: str) -> str:
        """单次生成"""
        if self.chat_llm:
            response = await self.chat_llm.agenerate([
                [HumanMessage(content=prompt)]
            ])
            return response.generations[0][0].text.strip()
        else:
            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text.strip()
    
    async def _generate_suggestions(self, question: str, answer: str) -> List[str]:
        """
        生成相关建议
        
        Args:
            question: 原问题
            answer: 生成的回答
            
        Returns:
            建议列表
        """
        suggestion_prompt = f"""
        基于以下问答对，生成3-5个相关的后续建议或问题：
        问题: {question}
        回答: {answer}
        
        请以JSON格式返回建议列表：["建议1", "建议2", "建议3"]
        """
        
        try:
            suggestions_text = await self._single_generation(suggestion_prompt)
            suggestions = json.loads(suggestions_text)
            return suggestions if isinstance(suggestions, list) else []
        except:
            return []
    
    def _format_user_info(self, user_context: Optional[Dict[str, Any]]) -> str:
        """格式化用户信息"""
        if not user_context or 'user_profile' not in user_context:
            return "无特定用户信息"
        
        profile = user_context['user_profile']
        info_parts = []
        
        if 'fitness_level' in profile:
            info_parts.append(f"健身水平: {profile['fitness_level']}")
        if 'goals' in profile:
            info_parts.append(f"健身目标: {', '.join(profile['goals'])}")
        if 'age' in profile:
            info_parts.append(f"年龄: {profile['age']}")
        if 'gender' in profile:
            info_parts.append(f"性别: {profile['gender']}")
        
        return "; ".join(info_parts) if info_parts else "无特定用户信息"
    
    async def _postprocess_result(self, 
                                result: Dict[str, Any], 
                                question: str, 
                                context_docs: List[Document]) -> Dict[str, Any]:
        """
        后处理生成结果
        
        Args:
            result: 原始生成结果
            question: 原问题
            context_docs: 上下文文档
            
        Returns:
            处理后的结果
        """
        # 添加元数据
        result['metadata'] = {
            'generation_strategy': self.generation_strategy,
            'model_provider': self.model_provider,
            'model_name': self.model_name,
            'context_doc_count': len(context_docs),
            'generated_at': datetime.now().isoformat()
        }
        
        # 质量检查
        quality_score = await self._assess_quality(result['answer'], question)
        result['quality_score'] = quality_score
        
        # 调整置信度
        if quality_score < 0.5:
            result['confidence'] *= 0.8
        
        return result
    
    async def _assess_quality(self, answer: str, question: str) -> float:
        """
        评估回答质量
        
        Args:
            answer: 生成的回答
            question: 原问题
            
        Returns:
            质量分数 (0-1)
        """
        # 简单的质量评估指标
        score = 0.5  # 基础分数
        
        # 长度检查
        if 50 <= len(answer) <= 1000:
            score += 0.2
        
        # 关键词匹配
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words & answer_words) / len(question_words) if question_words else 0
        score += overlap * 0.2
        
        # 结构检查
        if '。' in answer or '！' in answer or '？' in answer:
            score += 0.1
        
        return min(score, 1.0)
    
    def _update_metrics(self, generation_time: float, result: Dict[str, Any]):
        """更新性能指标"""
        # 更新平均生成时间
        total_successful = self.metrics['successful_generations'] + 1
        if total_successful == 1:
            self.metrics['avg_generation_time'] = generation_time
        else:
            self.metrics['avg_generation_time'] = (
                (self.metrics['avg_generation_time'] * (total_successful - 1) + generation_time) / total_successful
            )
        
        # 估算生成的token数量
        estimated_tokens = len(result.get('answer', '')) // 4  # 粗略估算
        self.metrics['total_tokens_generated'] += estimated_tokens
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.llm and not self.chat_llm:
                return False
            
            # 执行简单的生成测试
            test_result = await self._single_generation("健康检查测试")
            return len(test_result) > 0
            
        except Exception as e:
            self.logger.error(f"生成器健康检查失败: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.metrics,
            'success_rate': (
                self.metrics['successful_generations'] / self.metrics['total_generations']
                if self.metrics['total_generations'] > 0 else 0.0
            ),
            'timestamp': datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """清理资源"""
        try:
            if self.streaming_handler:
                self.streaming_handler.reset()
            self.logger.info("生成器资源清理完成")
        except Exception as e:
            self.logger.error(f"生成器资源清理失败: {e}")