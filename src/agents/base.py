"""
智能体基础类

定义了所有智能体的通用接口和基础功能：
- 生命周期管理
- 消息处理
- 状态管理
- 错误处理
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..utils.logger import get_logger
from ..core.events import EventBus
from ..rag.engine import RAGEngine
from ..multimodal.processor import MultimodalProcessor

logger = get_logger(__name__)

class AgentStatus(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class AgentMessage:
    """智能体消息"""
    message_id: str
    sender_id: str
    receiver_id: str
    content: Any
    message_type: str = "text"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentCapability:
    """智能体能力描述"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """智能体基础类"""
    
    def __init__(self, 
                 agent_id: str,
                 name: str = None,
                 description: str = None,
                 config: Dict[str, Any] = None):
        """初始化智能体
        
        Args:
            agent_id: 智能体唯一标识
            name: 智能体名称
            description: 智能体描述
            config: 配置参数
        """
        self.agent_id = agent_id
        self.name = name or agent_id
        self.description = description or f"智能体 {agent_id}"
        self.config = config or {}
        
        # 状态管理
        self.status = AgentStatus.IDLE
        self.last_activity = datetime.now()
        self.error_count = 0
        self.max_errors = self.config.get("max_errors", 5)
        
        # 消息处理
        self.message_queue = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}
        self.response_cache: Dict[str, Any] = {}
        
        # 组件
        self.event_bus: Optional[EventBus] = None
        self.rag_engine: Optional[RAGEngine] = None
        self.multimodal_processor: Optional[MultimodalProcessor] = None
        
        # 能力注册
        self.capabilities: Dict[str, AgentCapability] = {}
        
        # 性能指标
        self.metrics = {
            "total_messages": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "avg_response_time": 0.0
        }
        
        logger.info(f"智能体 {self.agent_id} 初始化完成")
    
    async def initialize(self) -> None:
        """初始化智能体"""
        try:
            logger.info(f"初始化智能体 {self.agent_id}...")
            
            # 初始化组件
            await self._initialize_components()
            
            # 注册能力
            await self._register_capabilities()
            
            # 注册消息处理器
            await self._register_message_handlers()
            
            # 启动消息处理循环
            asyncio.create_task(self._message_processing_loop())
            
            # 调用子类初始化
            await self._agent_initialize()
            
            self.status = AgentStatus.IDLE
            self.last_activity = datetime.now()
            
            logger.info(f"智能体 {self.agent_id} 初始化成功")
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 初始化失败: {e}")
            self.status = AgentStatus.ERROR
            raise
    
    async def stop(self) -> None:
        """停止智能体"""
        try:
            logger.info(f"停止智能体 {self.agent_id}...")
            
            self.status = AgentStatus.STOPPED
            
            # 调用子类停止逻辑
            await self._agent_stop()
            
            logger.info(f"智能体 {self.agent_id} 已停止")
            
        except Exception as e:
            logger.error(f"停止智能体 {self.agent_id} 失败: {e}")
            raise
    
    async def send_message(self, message: AgentMessage) -> None:
        """发送消息
        
        Args:
            message: 要发送的消息
        """
        try:
            if self.event_bus:
                await self.event_bus.publish(f"agent.message.{message.receiver_id}", {
                    "message": message,
                    "sender_id": self.agent_id
                })
            else:
                logger.warning(f"智能体 {self.agent_id} 未连接事件总线，无法发送消息")
                
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 发送消息失败: {e}")
            raise
    
    async def receive_message(self, message: AgentMessage) -> None:
        """接收消息
        
        Args:
            message: 接收到的消息
        """
        try:
            await self.message_queue.put(message)
            self.metrics["total_messages"] += 1
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 接收消息失败: {e}")
            raise
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求
        
        Args:
            request: 请求数据
            
        Returns:
            处理结果
        """
        start_time = datetime.now()
        
        try:
            self.status = AgentStatus.BUSY
            self.last_activity = datetime.now()
            
            # 检查缓存
            cache_key = self._generate_cache_key(request)
            if cache_key in self.response_cache:
                logger.info(f"智能体 {self.agent_id} 命中缓存")
                return self.response_cache[cache_key]
            
            # 处理请求
            response = await self._process_request(request)
            
            # 缓存响应
            self.response_cache[cache_key] = response
            
            # 更新指标
            self.metrics["successful_responses"] += 1
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_avg_response_time(response_time)
            
            self.status = AgentStatus.IDLE
            
            return response
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 处理请求失败: {e}")
            
            self.error_count += 1
            self.metrics["failed_responses"] += 1
            
            if self.error_count >= self.max_errors:
                self.status = AgentStatus.ERROR
                logger.error(f"智能体 {self.agent_id} 错误次数过多，状态设为ERROR")
            else:
                self.status = AgentStatus.IDLE
            
            raise
    
    def register_capability(self, capability: AgentCapability) -> None:
        """注册能力
        
        Args:
            capability: 能力描述
        """
        self.capabilities[capability.name] = capability
        logger.info(f"智能体 {self.agent_id} 注册能力: {capability.name}")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """获取能力列表
        
        Returns:
            能力列表
        """
        return list(self.capabilities.values())
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息
        
        Returns:
            状态信息
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "last_activity": self.last_activity,
            "error_count": self.error_count,
            "capabilities": [cap.name for cap in self.capabilities.values()],
            "metrics": self.metrics
        }
    
    # 抽象方法，子类必须实现
    @abstractmethod
    async def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求的具体实现
        
        Args:
            request: 请求数据
            
        Returns:
            处理结果
        """
        pass
    
    @abstractmethod
    async def _agent_initialize(self) -> None:
        """智能体特定的初始化逻辑"""
        pass
    
    @abstractmethod
    async def _agent_stop(self) -> None:
        """智能体特定的停止逻辑"""
        pass
    
    @abstractmethod
    async def _register_capabilities(self) -> None:
        """注册智能体能力"""
        pass
    
    # 内部方法
    async def _initialize_components(self) -> None:
        """初始化组件"""
        # 初始化RAG引擎
        if self.config.get("enable_rag", True):
            from ..rag.engine import RAGEngine
            self.rag_engine = RAGEngine(self.config.get("rag", {}))
            await self.rag_engine.initialize()
        
        # 初始化多模态处理器
        if self.config.get("enable_multimodal", True):
            from ..multimodal.processor import MultimodalProcessor
            self.multimodal_processor = MultimodalProcessor(self.config.get("multimodal", {}))
            await self.multimodal_processor.initialize()
    
    async def _register_message_handlers(self) -> None:
        """注册消息处理器"""
        self.message_handlers["text"] = self._handle_text_message
        self.message_handlers["image"] = self._handle_image_message
        self.message_handlers["audio"] = self._handle_audio_message
        self.message_handlers["video"] = self._handle_video_message
    
    async def _message_processing_loop(self) -> None:
        """消息处理循环"""
        while self.status != AgentStatus.STOPPED:
            try:
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._handle_message(message)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"智能体 {self.agent_id} 消息处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """处理消息"""
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                logger.warning(f"智能体 {self.agent_id} 不支持消息类型: {message.message_type}")
                
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 处理消息失败: {e}")
    
    async def _handle_text_message(self, message: AgentMessage) -> None:
        """处理文本消息"""
        # 子类可以重写此方法
        pass
    
    async def _handle_image_message(self, message: AgentMessage) -> None:
        """处理图像消息"""
        # 子类可以重写此方法
        pass
    
    async def _handle_audio_message(self, message: AgentMessage) -> None:
        """处理音频消息"""
        # 子类可以重写此方法
        pass
    
    async def _handle_video_message(self, message: AgentMessage) -> None:
        """处理视频消息"""
        # 子类可以重写此方法
        pass
    
    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        import json
        
        # 移除时间戳等动态字段
        cache_data = {k: v for k, v in request.items() 
                     if k not in ["timestamp", "request_id"]}
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _update_avg_response_time(self, response_time: float) -> None:
        """更新平均响应时间"""
        total_responses = self.metrics["successful_responses"] + self.metrics["failed_responses"]
        if total_responses > 0:
            current_avg = self.metrics["avg_response_time"]
            self.metrics["avg_response_time"] = (
                (current_avg * (total_responses - 1) + response_time) / total_responses
            )